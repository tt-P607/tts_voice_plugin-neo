"""GPT-SoVITS API v2 的 HTTP 客户端。

负责会话复用、模型权重切换与 ``/tts`` 请求构建。权重切换和合成请求在同一把
锁内串行执行，避免并发调用时音色串线；权重切换失败会直接中止本次合成，
而不是继续用未知的模型状态推理。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any, Final, Literal

import aiohttp

from src.app.plugin_system.api.log_api import get_logger

from ..config import TTSAdvancedSection
from ..protocol import PCMStream
from .styles import StyleProfile

logger = get_logger("tts_voice_plugin-neo.gsv")

_CONNECTION_LIMIT: Final[int] = 20
_RESPONSE_CHUNK_SIZE: Final[int] = 1024 * 1024

WeightType = Literal["gpt", "sovits"]


class GSVError(RuntimeError):
    """GPT-SoVITS 请求失败。"""


class GSVClient:
    """面向单个 GPT-SoVITS 服务实例的异步客户端。"""

    def __init__(self, timeout: int) -> None:
        """初始化客户端。

        Args:
            timeout: 单次请求的总超时秒数。
        """
        self._timeout = timeout
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()
        self._loaded_weights: dict[WeightType, str] = {}
        self._active_stream_tasks: set[asyncio.Task[Any]] = set()
        self._closing = False

    @property
    def timeout(self) -> int:
        """当前生效的请求超时秒数。"""
        return self._timeout

    def set_timeout(self, timeout: int) -> None:
        """更新超时配置。

        已建立的会话沿用旧超时，新会话在下次创建时生效。

        Args:
            timeout: 新的超时秒数。
        """
        self._timeout = timeout

    async def _get_session(self) -> aiohttp.ClientSession:
        """返回可复用的 aiohttp 会话，必要时重建。"""
        if self._closing:
            raise GSVError("TTS HTTP 客户端正在关闭")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout),
                connector=aiohttp.TCPConnector(limit=_CONNECTION_LIMIT),
            )
        return self._session

    async def close(self) -> None:
        """取消在途 PCM 流并关闭底层 HTTP 会话。"""
        self._closing = True
        current_task = asyncio.current_task()
        active_tasks = [
            task
            for task in self._active_stream_tasks
            if task is not current_task and not task.done()
        ]
        for task in active_tasks:
            task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _switch_weights(
        self,
        base_url: str,
        weights_path: str,
        weight_type: WeightType,
    ) -> None:
        """切换一类模型权重。

        Args:
            base_url: GPT-SoVITS 服务根地址（无尾斜杠）。
            weights_path: 权重文件路径；为空表示不切换。
            weight_type: 权重类别。

        Raises:
            GSVError: 切换请求失败或返回非 200。
        """
        if not weights_path or self._loaded_weights.get(weight_type) == weights_path:
            return

        session = await self._get_session()
        try:
            async with session.get(
                f"{base_url}/set_{weight_type}_weights",
                params={"weights_path": weights_path},
            ) as response:
                if response.status != 200:
                    detail = await response.text()
                    raise GSVError(
                        f"切换 {weight_type} 模型失败: {response.status} - {detail}"
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            raise GSVError(f"切换 {weight_type} 模型网络异常: {error}") from error

        self._loaded_weights[weight_type] = weights_path
        logger.info(f"成功切换 {weight_type} 模型为: {weights_path}")

    async def _prepare_weights(self, style: StyleProfile, base_url: str) -> None:
        """按风格切换 GPT 与 SoVITS 权重。

        Args:
            style: 目标风格。
            base_url: 服务根地址。

        Raises:
            GSVError: 任一权重切换失败。
        """
        await self._switch_weights(base_url, style.gpt_weights, "gpt")
        await self._switch_weights(base_url, style.sovits_weights, "sovits")

    @staticmethod
    def _build_payload(
        style: StyleProfile,
        text: str,
        text_language: str,
        advanced: TTSAdvancedSection,
        streaming_mode: int | None,
    ) -> dict[str, Any]:
        """构造 ``/tts`` 请求体。

        Args:
            style: 已应用调用级覆盖的风格。
            text: 待合成文本。
            text_language: 最终语言代码。
            advanced: 高级推理参数。
            streaming_mode: 流式等级；``None`` 表示非流式请求。

        Returns:
            使用全局推理参数与当前风格音色配置的请求体字典。
        """
        payload: dict[str, Any] = advanced.model_dump()
        payload.update(
            {
                "text": text,
                "text_lang": text_language,
                "ref_audio_path": style.refer_wav_path,
                "prompt_text": style.prompt_text,
                "prompt_lang": style.prompt_language,
                "speed_factor": style.speed_factor,
            }
        )
        if style.aux_refer_wav_paths:
            payload["aux_ref_audio_paths"] = list(style.aux_refer_wav_paths)
        if streaming_mode is not None:
            payload["streaming_mode"] = streaming_mode
        else:
            payload["streaming_mode"] = False
            payload["media_type"] = "wav"
        return payload

    @staticmethod
    def _tts_url(base_url: str) -> str:
        """拼出 ``/tts`` 端点地址。"""
        return base_url if base_url.endswith("/tts") else f"{base_url}/tts"

    async def synthesize(
        self,
        style: StyleProfile,
        text: str,
        text_language: str,
        advanced: TTSAdvancedSection,
    ) -> bytes:
        """合成完整音频。

        Args:
            style: 已应用调用级覆盖的风格。
            text: 待合成文本。
            text_language: 最终语言代码。
            advanced: 高级推理参数。

        Returns:
            完整的音频字节。

        Raises:
            GSVError: 权重切换失败、请求失败或返回空音频。
        """
        base_url = style.server_url.rstrip("/")
        payload = self._build_payload(style, text, text_language, advanced, None)

        async with self._lock:
            await self._prepare_weights(style, base_url)
            session = await self._get_session()
            try:
                async with session.post(self._tts_url(base_url), json=payload) as response:
                    if response.status != 200:
                        detail = await response.text()
                        raise GSVError(f"TTS 合成失败: {response.status} - {detail}")
                    audio = bytearray()
                    async for chunk in response.content.iter_chunked(_RESPONSE_CHUNK_SIZE):
                        audio.extend(chunk)
            except asyncio.TimeoutError as error:
                raise GSVError("TTS 请求超时") from error
            except aiohttp.ClientError as error:
                raise GSVError(f"TTS 网络异常: {error}") from error

        if not audio:
            raise GSVError("TTS 返回空音频")
        logger.debug(f"接收音频数据 {len(audio)} 字节")
        return bytes(audio)

    async def synthesize_stream(
        self,
        style: StyleProfile,
        text: str,
        text_language: str,
        advanced: TTSAdvancedSection,
        streaming_mode: int,
        chunk_size: int,
    ) -> AsyncGenerator[bytes, None]:
        """流式合成音频，边接收边产出。

        首块为 WAV header，后续为 raw PCM 数据块。

        Args:
            style: 已应用调用级覆盖的风格。
            text: 待合成文本。
            text_language: 最终语言代码。
            advanced: 高级推理参数。
            streaming_mode: 传递给 GPT-SoVITS 的流式等级。
            chunk_size: 每次产出的字节块大小。

        Yields:
            音频字节块。

        Raises:
            GSVError: 权重切换失败或流式请求失败。
        """
        base_url = style.server_url.rstrip("/")
        payload = self._build_payload(
            style, text, text_language, advanced, streaming_mode
        )

        async with self._lock:
            await self._prepare_weights(style, base_url)
            session = await self._get_session()
            logger.info(f"开始流式 TTS 合成: {text[:50]}")
            try:
                async with session.post(
                    self._tts_url(base_url), json=payload
                ) as response:
                    if response.status != 200:
                        detail = await response.text()
                        raise GSVError(
                            f"流式 TTS 合成失败: {response.status} - {detail}"
                        )
                    async for chunk in response.content.iter_chunked(chunk_size):
                        if chunk:
                            yield chunk
            except asyncio.TimeoutError as error:
                raise GSVError("流式 TTS 请求超时") from error
            except aiohttp.ClientError as error:
                raise GSVError(f"流式 TTS 网络异常: {error}") from error
            logger.info("流式 TTS 合成完成")

    @asynccontextmanager
    async def open_pcm_stream(
        self,
        style: StyleProfile,
        text: str,
        text_language: str,
        advanced: TTSAdvancedSection,
        streaming_mode: int,
        streaming_chunk_seconds: float,
        sample_steps: int,
        cfg_rate: float,
        chunk_size: int,
    ) -> AsyncIterator[PCMStream]:
        """打开 GPT-SoVITS V5 raw PCM 流并持有请求锁至流关闭。"""
        if streaming_mode != 2:
            raise ValueError("PCM 流只支持 GPT-SoVITS V5 streaming_mode=2")

        base_url = style.server_url.rstrip("/")
        payload = self._build_payload(
            style, text, text_language, advanced, streaming_mode
        )
        payload.update(
            {
                "streaming_chunk_seconds": streaming_chunk_seconds,
                "media_type": "raw",
                "batch_size": 1,
                "sample_steps": sample_steps,
                "cfg_rate": cfg_rate,
            }
        )

        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("V5 PCM 流必须在 asyncio task 中运行")
        if self._closing:
            raise GSVError("TTS HTTP 客户端正在关闭")
        self._active_stream_tasks.add(task)
        try:
            async with self._lock:
                await self._prepare_weights(style, base_url)
                session = await self._get_session()
                request_started = perf_counter()
                logger.info("V5 PCM 请求开始")
                try:
                    async with session.post(
                        self._tts_url(base_url), json=payload
                    ) as response:
                        if response.status != 200:
                            raise GSVError(f"V5 PCM 流请求失败: HTTP {response.status}")
                        if response.content_type != "audio/raw":
                            raise GSVError(
                                f"V5 PCM 流 Content-Type 无效: {response.content_type}"
                            )
                        try:
                            sample_rate = int(response.headers["X-Audio-Sample-Rate"])
                            channels = int(response.headers["X-Audio-Channels"])
                            response_mode = int(response.headers["X-Streaming-Mode"])
                        except (KeyError, ValueError) as error:
                            raise GSVError("V5 PCM 流缺少有效的格式响应头") from error
                        if sample_rate != 48000 or channels != 1 or response_mode != 2:
                            raise GSVError(
                                "V5 PCM 流格式不匹配: "
                                f"sample_rate={sample_rate}, channels={channels}, "
                                f"streaming_mode={response_mode}"
                            )

                        async def chunks() -> AsyncGenerator[bytes, None]:
                            total_bytes = 0
                            first_pcm = True
                            async for chunk in response.content.iter_chunked(chunk_size):
                                if chunk:
                                    total_bytes += len(chunk)
                                    if first_pcm:
                                        logger.info(
                                            "V5 PCM 首块到达: "
                                            f"elapsed={perf_counter() - request_started:.3f}s, "
                                            f"bytes={len(chunk)}"
                                        )
                                        first_pcm = False
                                    yield chunk
                            logger.info(
                                "V5 PCM 网络 EOF: "
                                f"elapsed={perf_counter() - request_started:.3f}s, "
                                f"bytes={total_bytes}"
                            )

                        chunk_iterator = chunks()
                        stream = PCMStream(
                            sample_rate=sample_rate,
                            channels=channels,
                            sample_format="s16le",
                            chunks=chunk_iterator,
                        )
                        try:
                            yield stream
                        finally:
                            await chunk_iterator.aclose()
                except TimeoutError as error:
                    raise GSVError("V5 PCM 流请求超时") from error
                except aiohttp.ClientError as error:
                    raise GSVError(f"V5 PCM 流网络异常: {error}") from error
        finally:
            self._active_stream_tasks.discard(task)


__all__ = ["GSVClient", "GSVError", "WeightType"]
