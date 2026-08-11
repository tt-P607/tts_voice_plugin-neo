"""TTS 语音合成 Action。

通过 LLM Tool Calling 或关键词触发 GPT-SoVITS 语音合成并发送语音消息。
支持段落级参数控制（风格/语言/语速）、语音条与文件两种发送模式，以及多段合并。

``execute`` 声明了全部可选参数；哪些参数真正暴露给模型由插件配置决定——
:meth:`TTSVoiceAction.to_schema` 会按 :attr:`exposed_optional_params` 裁剪，
避免为每种开关组合各写一个子类。
"""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, cast

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.send_api import send_file, send_voice
from src.app.plugin_system.base import BaseAction, BasePlugin

from .. import prompts
from ..language import LANGUAGE_HELP_TEXT
from ..protocol import TTSPluginLike
from ..services import audio

if TYPE_CHECKING:
    from src.app.plugin_system.types import ChatStream

    from ..services.tts_service import TTSService

logger = get_logger("tts_voice_plugin-neo.action")

# 中文语速约 5 字/秒，用于 WAV 解析失败时的时长兜底估算。
_CHARS_PER_SECOND = 5.0
# 单次调用允许的最大段落数，防止模型一次提交过多合成任务。
_MAX_SEGMENTS = 32
_DEFAULT_PAUSE_DURATION = 0.3
# 仅在对应配置开关打开时才暴露给模型的参数。
_GATED_PARAMS: frozenset[str] = frozenset(
    {"speed_factor", "audio_effects", "aux_refer_wav_paths", "merge_voice", "pause_duration"}
)


class TTSVoiceAction(BaseAction):
    """把文本合成为语音并发送到当前聊天流。"""

    name: str = "tts_voice_action"
    associated_types: list[str] = ["voice"]
    description: str = prompts.ACTION_DESCRIPTION
    primary_action: bool = False

    exposed_optional_params: ClassVar[frozenset[str]] = frozenset()
    """当前配置下额外暴露给模型的可选参数名集合，由插件在装配时写入。"""

    def __init__(self, chat_stream: "ChatStream", plugin: "BasePlugin") -> None:
        """初始化 Action。

        Args:
            chat_stream: 当前聊天流。
            plugin: 所属插件实例。
        """
        super().__init__(chat_stream, plugin)
        self.tts_plugin: TTSPluginLike = cast(TTSPluginLike, plugin)

    @property
    def tts_service(self) -> "TTSService | None":
        """插件持有的 TTS Service，未初始化时为 ``None``。"""
        return self.tts_plugin.tts_service

    @classmethod
    def to_schema(cls) -> dict[str, Any]:
        """生成 Tool Schema，并剔除当前配置未开放的可选参数。

        Returns:
            OpenAI Tool Calling 格式的 schema。
        """
        schema = super().to_schema()
        properties = schema["function"]["parameters"]["properties"]
        for name in _GATED_PARAMS - cls.exposed_optional_params:
            properties.pop(name, None)
        return schema

    # ------------------------------------------------------------------
    # 激活判定
    # ------------------------------------------------------------------

    async def go_activate(self) -> bool:
        """判断本次是否激活语音合成。

        满足随机概率、关键词命中或 LLM 判定任一条件即激活。

        Returns:
            是否激活。
        """
        if await self._random_activation(0.25):
            logger.info("TTSVoiceAction 随机激活成功 (25%)")
            return True

        keywords = self.tts_plugin.config.plugin.keywords
        if keywords and await self._keyword_match(keywords):
            logger.info("TTSVoiceAction 关键词激活成功")
            return True

        if await self._llm_judge_activation():
            logger.info("TTSVoiceAction LLM 判断激活成功")
            return True

        logger.debug("TTSVoiceAction 所有激活条件均未满足")
        return False

    # ------------------------------------------------------------------
    # 合成与发送
    # ------------------------------------------------------------------

    @staticmethod
    def _send_interval(duration: float) -> float:
        """按上一条语音的时长计算下一条的发送间隔，模拟真人节奏。

        Args:
            duration: 上一条语音时长（秒）。

        Returns:
            建议的等待秒数。
        """
        if duration < 3.0:
            return 0.8
        if duration < 10.0:
            return 1.5
        if duration < 30.0:
            return 2.5
        return 3.5

    async def _synthesize_segments(
        self,
        segments: list[dict[str, Any]],
        voice_style: str,
        text_language: str | None,
        speed_factor: float | None,
        audio_effects: list[dict[str, Any]] | None,
        aux_refer_wav_paths: list[str] | None,
    ) -> list[bytes]:
        """串行合成所有段落。

        GPT-SoVITS 本身是 GPU 串行推理，并发提交不会加速，反而增加权重切换竞争。

        Args:
            segments: 段落列表，每段含 ``text`` 与可选的段级参数。
            voice_style: 全局默认风格。
            text_language: 全局默认语言。
            speed_factor: 全局默认语速。
            audio_effects: 效果器链。
            aux_refer_wav_paths: 辅助参考音频。

        Returns:
            成功合成的音频字节列表。
        """
        service = self.tts_service
        if service is None:
            return []

        audio_list: list[bytes] = []
        for index, segment in enumerate(segments, start=1):
            result = await service.generate_voice_bytes(
                text=segment["text"],
                style_hint=segment.get("voice_style") or voice_style,
                language_hint=segment.get("text_language") or text_language,
                speed_factor=(
                    segment["speed_factor"]
                    if segment.get("speed_factor") is not None
                    else speed_factor
                ),
                audio_effects=audio_effects,
                aux_refer_wav_paths=aux_refer_wav_paths,
            )
            if result:
                audio_list.append(result)
            else:
                logger.error(f"第 {index} 段语音合成失败")
        return audio_list

    async def _send_voice_sequence(
        self,
        audio_list: list[bytes],
        texts: list[str],
    ) -> tuple[bool, str]:
        """逐条发送语音条，条间插入自适应间隔。

        Args:
            audio_list: 音频字节列表。
            texts: 对应的原始文本，用于时长兜底估算。

        Returns:
            ``(是否成功, 结果描述)``。
        """
        previous_duration = 0.0
        for index, audio_bytes in enumerate(audio_list):
            if index > 0:
                interval = self._send_interval(previous_duration)
                logger.debug(f"第 {index + 1} 段发送前等待 {interval:.1f}s")
                await asyncio.sleep(interval)

            await send_voice(
                voice_data=base64.b64encode(audio_bytes).decode("utf-8"),
                stream_id=self.chat_stream.stream_id,
            )
            previous_duration = audio.estimate_duration(audio_bytes)
            if previous_duration <= 0 and index < len(texts):
                previous_duration = len(texts[index]) / _CHARS_PER_SECOND
            logger.info(
                f"第 {index + 1}/{len(audio_list)} 段语音发送成功"
                f"（时长约 {previous_duration:.1f}s）"
            )

        total_length = sum(len(text) for text in texts)
        return True, f"成功发送 {len(audio_list)} 段语音，总文本长度: {total_length} 字符"

    async def _send_merged_voice(
        self,
        audio_list: list[bytes],
        texts: list[str],
        pause_duration: float,
    ) -> tuple[bool, str]:
        """把多段音频合并为一条语音条发送。

        Args:
            audio_list: 音频字节列表。
            texts: 对应的原始文本。
            pause_duration: 段间静音秒数。

        Returns:
            ``(是否成功, 结果描述)``。
        """
        merged = await audio.merge_audio(audio_list, pause_duration)
        if not merged:
            return False, "音频拼接失败"

        await send_voice(
            voice_data=base64.b64encode(merged).decode("utf-8"),
            stream_id=self.chat_stream.stream_id,
        )
        logger.info(
            f"拼接语音发送成功，包含 {len(audio_list)} 段，"
            f"段间停顿 {pause_duration}s，时长约 {audio.estimate_duration(merged):.1f}s"
        )
        total_length = sum(len(text) for text in texts)
        return (
            True,
            f"成功拼接 {len(audio_list)} 段为一条语音，总文本长度: {total_length} 字符",
        )

    async def _send_as_file(
        self,
        audio_list: list[bytes],
        texts: list[str],
        pause_duration: float,
        file_name: str | None,
    ) -> tuple[bool, str]:
        """把多段音频合并后以文件发送。

        Args:
            audio_list: 音频字节列表。
            texts: 对应的原始文本。
            pause_duration: 段间静音秒数。
            file_name: 自定义文件名；为空时使用时间戳。

        Returns:
            ``(是否成功, 结果描述)``。
        """
        merged = await audio.merge_audio(audio_list, pause_duration)
        if not merged:
            return False, "音频合并失败"

        requested_name = file_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = await audio.write_temp_audio(merged, requested_name)
        wsl_mode = self.tts_plugin.config.tts.wsl_mode
        send_path = audio.to_wsl_path(str(file_path)) if wsl_mode else file_path.as_posix()
        try:
            await send_file(
                file_path=send_path,
                stream_id=self.chat_stream.stream_id,
                file_name=file_path.name,
            )
        finally:
            file_path.unlink(missing_ok=True)

        total_length = sum(len(text) for text in texts)
        logger.info(f"合并音频文件发送成功，包含 {len(audio_list)} 段，{len(merged)} 字节")
        return (
            True,
            f"成功合并 {len(audio_list)} 段并以文件发送，总文本长度: {total_length} 字符",
        )

    # ------------------------------------------------------------------
    # 执行入口
    # ------------------------------------------------------------------

    async def execute(
        self,
        tts_segments: Annotated[list[dict[str, Any]], prompts.TTS_SEGMENTS_HINT],
        send_mode: Annotated[str, prompts.SEND_MODE_HINT] = "voice",
        voice_style: Annotated[str, prompts.VOICE_STYLE_HINT] = "default",
        text_language: Annotated[str | None, LANGUAGE_HELP_TEXT] = None,
        file_name: Annotated[str | None, prompts.FILE_NAME_HINT] = None,
        speed_factor: Annotated[float | None, prompts.SPEED_FACTOR_HINT] = None,
        audio_effects: Annotated[
            list[dict[str, Any]] | None, prompts.AUDIO_EFFECTS_HINT
        ] = None,
        aux_refer_wav_paths: Annotated[
            list[str] | None, prompts.AUX_REFER_WAV_PATHS_HINT
        ] = None,
        merge_voice: Annotated[bool, prompts.MERGE_VOICE_HINT] = False,
        pause_duration: Annotated[float | None, prompts.PAUSE_DURATION_HINT] = None,
    ) -> tuple[bool, str]:
        """合成并发送语音。

        Args:
            tts_segments: 待合成的段落列表。
            send_mode: 发送模式，``voice`` 或 ``file``。
            voice_style: 全局默认语音风格。
            text_language: 全局默认语言。
            file_name: file 模式下的自定义文件名。
            speed_factor: 全局默认语速因子。
            audio_effects: 音频效果器链。
            aux_refer_wav_paths: 辅助参考音频路径列表。
            merge_voice: voice 模式下是否合并为一条语音条。
            pause_duration: 合并时的段间静音秒数。

        Returns:
            ``(是否成功, 结果描述)``。
        """
        service = self.tts_service
        if service is None:
            logger.error("TTSService 未初始化，跳过本次合成")
            return False, "TTSService 未注册或初始化失败"

        if send_mode not in {"voice", "file"}:
            return False, "send_mode 只支持 voice 或 file"
        if speed_factor is not None and not 0.5 <= speed_factor <= 2.0:
            return False, "speed_factor 必须位于 0.5 到 2.0 之间"
        if pause_duration is not None and not 0.0 <= pause_duration <= 10.0:
            return False, "pause_duration 必须位于 0 到 10 秒之间"
        if len(tts_segments) > _MAX_SEGMENTS:
            return False, f"单次最多支持 {_MAX_SEGMENTS} 个语音段落"

        segments = [
            {**segment, "text": str(segment.get("text", "")).strip()}
            for segment in tts_segments
            if str(segment.get("text", "")).strip()
        ]
        if not segments:
            logger.warning("段落列表为空，跳过本次合成")
            return False, "文本列表为空"

        texts = [segment["text"] for segment in segments]
        logger.info(
            f"接收到 {len(segments)} 段文本 | 发送模式: {send_mode} | "
            f"全局风格: {voice_style} | 合并发送: {merge_voice}"
        )

        audio_list = await self._synthesize_segments(
            segments,
            voice_style,
            text_language,
            speed_factor,
            audio_effects,
            aux_refer_wav_paths,
        )
        if not audio_list:
            return False, "所有语音段均合成失败"

        effective_pause = (
            pause_duration if pause_duration is not None else _DEFAULT_PAUSE_DURATION
        )
        if send_mode == "file":
            return await self._send_as_file(
                audio_list, texts, effective_pause, file_name
            )
        if merge_voice and len(audio_list) > 1:
            return await self._send_merged_voice(audio_list, texts, effective_pause)
        return await self._send_voice_sequence(audio_list, texts)


__all__ = ["TTSVoiceAction"]
