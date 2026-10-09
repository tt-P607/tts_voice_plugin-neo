"""TTS 核心服务。

对外暴露语音合成能力，内部把职责委托给四个协作模块：

- :mod:`.styles`  风格配置 → 强类型 :class:`~.styles.StyleProfile`
- :mod:`.gsv_client`  GPT-SoVITS HTTP 通信与权重切换
- :mod:`.effects`  pedalboard 音频后处理
- :mod:`.audio`  WAV 拼接、时长估算与临时文件

本类只做参数解析、语言决策和流程编排。
"""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import BasePlugin, BaseService

from ..config import TTSVoiceConfig
from ..language import normalize_language_code
from ..protocol import PCMStream
from . import audio, effects
from .gsv_client import GSVClient, GSVError
from .styles import StyleProfile, StyleRegistry, load_styles

logger = get_logger("tts_voice_plugin-neo.service")

# 合成前从文本中剔除的括号注释（动作描写、旁白等无法语音化的内容）。
_BRACKET_PATTERN = re.compile(r"[\(（\[【].*?[\)）\]】]")
_LOG_TEXT_LIMIT = 80


class TTSService(BaseService):
    """GPT-SoVITS 语音合成服务。"""

    name: str = "tts"
    description: str = "GPT-SoVITS 语音合成服务"

    def __init__(self, plugin: BasePlugin) -> None:
        """初始化服务并装载风格配置。

        Args:
            plugin: 所属插件实例。

        Raises:
            ValueError: 风格配置为空或存在重名。
        """
        super().__init__(plugin)
        self.styles: StyleRegistry = load_styles(self.config)
        self._client = GSVClient(self.config.tts.timeout)
        logger.info(f"TTS 服务已加载风格: {self.styles.names}")

    @property
    def config(self) -> TTSVoiceConfig:
        """当前插件配置。"""
        return cast(TTSVoiceConfig, self.plugin.config)

    def refresh_config(self) -> None:
        """在配置变更后重新装载风格与客户端参数。

        Raises:
            ValueError: 新配置的风格列表为空或存在重名。
        """
        self.styles = load_styles(self.config)
        self._client.set_timeout(self.config.tts.timeout)
        logger.debug(f"TTS 服务热更加载风格: {self.styles.names}")

    async def close(self) -> None:
        """释放服务持有的网络资源。"""
        await self._client.close()

    def get_available_styles(self) -> list[str]:
        """返回可用语音风格名称列表。"""
        return self.styles.names

    # ------------------------------------------------------------------
    # 参数解析
    # ------------------------------------------------------------------

    def _clean_text(self, text: str, *, limit_length: bool = True) -> str:
        """剔除括号注释，完整 WAV 路径按配置上限截断。

        Args:
            text: 原始文本。
            limit_length: 是否应用普通合成的文本长度上限。

        Returns:
            清洗后的文本，可能为空串。
        """
        cleaned = _BRACKET_PATTERN.sub("", text)
        if limit_length:
            cleaned = cleaned[: self.config.tts.max_text_length]
        return cleaned.strip()

    def _resolve_language(self, style: StyleProfile, language_hint: str | None) -> str:
        """决定最终发送给 GPT-SoVITS 的语言代码。

        调用方传入的 ``language_hint`` 优先于风格默认语言；两者都会经过
        归一化，因为模型可能给出 ``chinese`` / ``zh-CN`` 之类的非法写法。

        Args:
            style: 已解析的风格。
            language_hint: 调用方指定的语言，可为空。

        Returns:
            合法的 GPT-SoVITS 语言代码。
        """
        raw = language_hint or style.text_language
        code, kind = normalize_language_code(raw, default="zh")
        if kind == "fallback":
            logger.warning(f"无法识别的语言代码 {raw!r}，已回退为 zh")
        elif kind in {"alias", "fuzzy", "normalized"}:
            logger.info(f"语言代码 {raw!r} 通过 {kind} 规整为 {code}")
        return code

    def _resolve_effects(
        self,
        audio_effects: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """决定生效的效果器链。

        调用方传入的效果链优先；未传入时使用配置中启用的手动效果链。

        Args:
            audio_effects: 调用方传入的效果器链，可为空。

        Returns:
            效果器描述列表，可能为空。
        """
        if audio_effects:
            return audio_effects
        configured = self.config.audio_effects
        if configured.enabled and configured.chain:
            return [
                {"type": item.type, "params": dict(item.params)}
                for item in configured.chain
            ]
        return []

    def _prepare(
        self,
        text: str,
        style_hint: str,
        language_hint: str | None,
        speed_factor: float | None,
        aux_refer_wav_paths: list[str] | None,
        *,
        limit_length: bool = True,
    ) -> tuple[StyleProfile, str, str] | None:
        """完成风格选择、文本清洗与语言决策。

        Args:
            text: 原始文本。
            style_hint: 期望的风格名。
            language_hint: 期望的语言代码。
            speed_factor: 覆盖语速。
            aux_refer_wav_paths: 覆盖辅助参考音频。
            limit_length: 是否应用普通合成的文本长度上限。

        Returns:
            ``(风格, 清洗后文本, 语言代码)``；文本清洗后为空时返回 ``None``。

        Raises:
            LookupError: 没有任何可用风格。
        """
        style = self.styles.resolve(style_hint)
        if style_hint and style.name != style_hint:
            logger.warning(f"风格 {style_hint!r} 不存在，回退到 {style.name!r}")

        clean_text = self._clean_text(text, limit_length=limit_length)
        if not clean_text:
            return None

        language = self._resolve_language(style, language_hint)
        style = style.override(
            speed_factor=speed_factor,
            aux_refer_wav_paths=aux_refer_wav_paths,
        )
        return style, clean_text, language

    # ------------------------------------------------------------------
    # 合成入口
    # ------------------------------------------------------------------

    async def generate_voice_bytes(
        self,
        text: str,
        style_hint: str = "default",
        language_hint: str | None = None,
        speed_factor: float | None = None,
        audio_effects: list[dict[str, Any]] | None = None,
        aux_refer_wav_paths: list[str] | None = None,
    ) -> bytes | None:
        """合成语音并返回 WAV 字节。

        Args:
            text: 待合成文本。
            style_hint: 风格名称；不存在时回退到首个风格。
            language_hint: 语言代码；为空时使用风格默认语言。
            speed_factor: 语速因子；为空时使用风格默认语速。
            audio_effects: 效果器链；为空时使用配置的手动效果链。
            aux_refer_wav_paths: 辅助参考音频；为空时使用风格配置值。

        Returns:
            WAV 字节；文本为空或合成失败时返回 ``None``。
        """
        prepared = self._prepare(
            text, style_hint, language_hint, speed_factor, aux_refer_wav_paths
        )
        if prepared is None:
            return None
        style, clean_text, language = prepared

        logger.info(
            f"开始 TTS 合成 | 风格: {style.name} | 语言: {language} | "
            f"语速: {style.speed_factor} | "
            f"参考音频: {Path(style.refer_wav_path).name or '(无)'} | "
            f"辅助音频: {len(style.aux_refer_wav_paths)} | "
            f"文本: {clean_text[:_LOG_TEXT_LIMIT]}"
        )

        try:
            audio_data = await self._client.synthesize(
                style, clean_text, language, self.config.tts_advanced
            )
        except GSVError as error:
            logger.error(f"TTS 合成失败: {error}")
            return None

        chain = self._resolve_effects(audio_effects)
        if not chain:
            return audio_data
        try:
            return await effects.apply_effects(audio_data, chain)
        except ValueError as error:
            logger.error(f"音频效果器配置无效: {error}")
            return audio_data

    async def generate_voice_stream(
        self,
        text: str,
        style_hint: str = "default",
        language_hint: str | None = None,
        chunk_size: int = 4096,
        aux_refer_wav_paths: list[str] | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """流式合成语音，边接收边产出音频块。

        流式路径不支持效果器后处理——效果器需要完整音频才能渲染。

        Args:
            text: 待合成文本。
            style_hint: 风格名称。
            language_hint: 语言代码。
            chunk_size: 每次产出的字节块大小。
            aux_refer_wav_paths: 辅助参考音频。

        Yields:
            音频字节块。

        Raises:
            GSVError: 权重切换或流式请求失败。
        """
        prepared = self._prepare(text, style_hint, language_hint, None, aux_refer_wav_paths)
        if prepared is None:
            return
        style, clean_text, language = prepared

        async for chunk in self._client.synthesize_stream(
            style,
            clean_text,
            language,
            self.config.tts_advanced,
            self.config.tts_streaming.streaming_mode,
            chunk_size,
        ):
            yield chunk

    @asynccontextmanager
    async def open_pcm_stream(
        self,
        text: str,
        style_hint: str,
        language_hint: str | None,
        speed_factor: float | None,
        aux_refer_wav_paths: list[str] | None,
    ) -> AsyncIterator[PCMStream]:
        """打开 V5 PCM 流并将请求参数解析为最终合成风格。"""
        prepared = self._prepare(
            text, style_hint, language_hint, speed_factor, aux_refer_wav_paths,
            limit_length=False,
        )
        if prepared is None:
            raise ValueError("text 不能为空")
        style, clean_text, language = prepared
        streaming = self.config.tts_streaming

        async with self._client.open_pcm_stream(
            style=style,
            text=clean_text,
            text_language=language,
            advanced=self.config.tts_advanced,
            streaming_mode=streaming.streaming_mode,
            streaming_chunk_seconds=streaming.streaming_chunk_seconds,
            sample_steps=streaming.sample_steps,
            cfg_rate=streaming.cfg_rate,
            chunk_size=streaming.chunk_size,
        ) as stream:
            yield stream

    # ------------------------------------------------------------------
    # 音频工具（转发到 audio 模块，便于组件通过 Service 单点访问）
    # ------------------------------------------------------------------

    @staticmethod
    def to_wsl_path(path: str) -> str:
        """把 Windows 绝对路径转换为 WSL 挂载路径。"""
        return audio.to_wsl_path(path)

    @staticmethod
    def sanitize_file_name(file_name: str | None) -> str:
        """生成安全的 WAV 文件名。"""
        return audio.sanitize_file_name(file_name)

    @staticmethod
    async def write_temp_audio(audio_data: bytes, file_name: str | None = None) -> Path:
        """把音频写入插件数据目录的唯一临时文件。"""
        return await audio.write_temp_audio(audio_data, file_name)

    @staticmethod
    async def merge_audio_bytes(
        audio_list: list[bytes],
        pause_duration: float = 0.0,
    ) -> bytes | None:
        """把多段 WAV 合并为一段，可在段间插入静音。"""
        return await audio.merge_audio(audio_list, pause_duration)


__all__ = ["TTSService"]
