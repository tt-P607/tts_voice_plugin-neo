"""tts_voice_plugin-neo 的 TTS Provider 适配层。

通过结构协议接入 TTS Registry，不直接导入 ``tts_http_server`` 的源码。
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncGenerator
from typing import Any

from src.app.plugin_system.api.log_api import get_logger

from . import prompts
from .language import LANGUAGE_HELP_TEXT
from .protocol import (
    ParameterGuide,
    ProviderCapabilities,
    SynthesisRequestLike,
    SynthesisResponse,
)
from .services.audio import estimate_duration
from .services.tts_service import TTSService

logger = get_logger("tts_voice_plugin-neo.provider")

_MIN_CHUNK_SIZE = 512
_MAX_CHUNK_SIZE = 1024 * 1024


class TTSVoiceProvider:
    """把插件持有的 TTSService 暴露为 Registry Provider。"""

    provider_name = "tts_voice_plugin-neo"

    def __init__(self, tts_service: TTSService) -> None:
        """初始化 Provider。

        Args:
            tts_service: 插件持有的 TTS Service 实例。
        """
        self.tts_service = tts_service

    @staticmethod
    def _parse_speed(raw_value: Any) -> float | None:
        """解析并校验可选语速。

        Args:
            raw_value: 原始输入值。

        Returns:
            语速因子；未提供时返回 ``None``。

        Raises:
            ValueError: 值不是数字或超出 0.5 ~ 2.0。
        """
        if raw_value is None or raw_value == "":
            return None
        try:
            speed = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError("speed 必须是数字") from error
        if not 0.5 <= speed <= 2.0:
            raise ValueError("speed 必须位于 0.5 到 2.0 之间")
        return speed

    @staticmethod
    def _parse_aux_refs(raw_value: Any) -> list[str] | None:
        """解析 options/markers 中的辅助参考音频路径列表。

        兼容元素为 JSON 字符串的调用方（部分模型会把列表编码成 JSON 字符串）。

        Args:
            raw_value: 原始输入值。

        Returns:
            路径列表；未提供或为空时返回 ``None``。

        Raises:
            ValueError: 结构不是字符串列表，或包含无效 JSON。
        """
        if raw_value is None:
            return None
        if not isinstance(raw_value, list):
            raise ValueError("aux_refer_wav_paths 必须是字符串列表")
        paths: list[str] = []
        for item in raw_value:
            parsed = item
            if isinstance(item, str):
                stripped = item.strip()
                if stripped.startswith("[") or stripped.startswith('"'):
                    try:
                        import json

                        parsed = json.loads(stripped)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            "aux_refer_wav_paths 包含无效 JSON 字符串"
                        ) from error
                else:
                    paths.append(stripped)
                    continue
            if not isinstance(parsed, str):
                raise ValueError("aux_refer_wav_paths 每一项都必须是字符串")
            stripped = parsed.strip()
            if stripped:
                paths.append(stripped)
        return paths or None

    @staticmethod
    def _parse_effects(raw_value: Any) -> list[dict[str, Any]] | None:
        """解析 options 中的效果器列表。

        兼容元素为 JSON 字符串的调用方。

        Args:
            raw_value: 原始输入值。

        Returns:
            效果器描述列表；未提供或为空时返回 ``None``。

        Raises:
            ValueError: 结构不是对象列表，或包含无效 JSON。
        """
        if raw_value is None:
            return None
        if not isinstance(raw_value, list):
            raise ValueError("effects 必须是对象列表")

        effects: list[dict[str, Any]] = []
        for item in raw_value:
            parsed = item
            if isinstance(item, str):
                try:
                    parsed = json.loads(item)
                except json.JSONDecodeError as error:
                    raise ValueError("effects 包含无效 JSON 字符串") from error
            if not isinstance(parsed, dict):
                raise ValueError("effects 每一项都必须是对象")
            effects.append(dict(parsed))
        return effects or None

    async def synthesize(self, request: SynthesisRequestLike) -> SynthesisResponse:
        """合成完整音频并返回结构化响应。

        Args:
            request: 规范化的合成请求。

        Returns:
            带 base64 音频的合成响应。

        Raises:
            ValueError: 文本为空或参数非法。
            RuntimeError: 合成未产出音频。
        """
        text = str(request.text or "").strip()
        if not text:
            raise ValueError("text 不能为空")

        options = request.options if isinstance(request.options, dict) else {}
        markers = request.markers if isinstance(request.markers, dict) else {}

        def _pick(*keys: str) -> Any:
            """按优先级从 options / markers 中取第一个非空值。"""
            for key in keys:
                value = options.get(key)
                if value not in (None, ""):
                    return value
                value = markers.get(key)
                if value not in (None, ""):
                    return value
            return None

        language = _pick("language")
        audio_bytes = await self.tts_service.generate_voice_bytes(
            text=text,
            style_hint=str(_pick("style") or "default").strip(),
            language_hint=str(language).strip().lower() if language else None,
            speed_factor=self._parse_speed(_pick("speed")),
            audio_effects=self._parse_effects(_pick("effects", "audio_effects")),
            aux_refer_wav_paths=self._parse_aux_refs(
                _pick("aux_refer_wav_paths", "aux_refer_wav_paths")
            ),
        )
        if not audio_bytes:
            raise RuntimeError("TTS 合成失败，未生成音频数据")

        duration = estimate_duration(audio_bytes)
        return SynthesisResponse(
            audio_base64=base64.b64encode(audio_bytes).decode("utf-8"),
            mime_type="audio/wav",
            format="wav",
            duration_ms=int(duration * 1000) if duration > 0 else None,
            text=text,
            provider=self.provider_name,
        )

    async def synthesize_stream(
        self,
        text: str,
        style_hint: str = "default",
        language_hint: str | None = None,
        chunk_size: int | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """按配置启用流式合成并逐块返回音频。

        Args:
            text: 待合成文本。
            style_hint: 风格名称。
            language_hint: 语言代码。
            chunk_size: 每次产出的字节块大小；为空时使用配置值。

        Yields:
            音频字节块。

        Raises:
            RuntimeError: 流式合成未启用。
            ValueError: ``chunk_size`` 超出允许范围。
        """
        streaming = self.tts_service.config.tts_streaming
        if not streaming.enabled:
            raise RuntimeError("流式合成未启用")

        effective_chunk_size = chunk_size or streaming.chunk_size
        if not _MIN_CHUNK_SIZE <= effective_chunk_size <= _MAX_CHUNK_SIZE:
            raise ValueError(
                f"chunk_size 必须位于 {_MIN_CHUNK_SIZE} 到 {_MAX_CHUNK_SIZE} 之间"
            )

        async for chunk in self.tts_service.generate_voice_stream(
            text=text,
            style_hint=style_hint,
            language_hint=language_hint,
            chunk_size=effective_chunk_size,
        ):
            yield chunk

    def get_capabilities(self) -> ProviderCapabilities:
        """返回当前 Provider 的动态参数能力。

        Returns:
            按当前配置与风格列表生成的能力描述。
        """
        styles = self.tts_service.get_available_styles()
        if styles:
            style_guide = ParameterGuide(
                description=prompts.VOICE_STYLE_HINT
                + "\n\n【当前可用语音风格】必须从以下列表选择：\n"
                + "\n".join(f"  - '{name}'" for name in styles),
                param_type="string",
                default=styles[0],
                valid_values=styles,
            )
        else:
            style_guide = ParameterGuide(
                description=prompts.VOICE_STYLE_HINT
                + "\n\n当前没有可用的 TTS 风格。",
                param_type="string",
                default="default",
            )

        plugin_config = self.tts_service.config.plugin
        return ProviderCapabilities(
            style_guide=style_guide,
            language_guide=ParameterGuide(
                description=LANGUAGE_HELP_TEXT,
                param_type="string",
                default="zh",
            ),
            speed_guide=ParameterGuide(
                description=prompts.SPEED_FACTOR_HINT,
                param_type="number",
                default=1.0,
                min_value=0.5,
                max_value=2.0,
            )
            if plugin_config.llm_speed_control
            else None,
            effects_guide=ParameterGuide(
                description=prompts.AUDIO_EFFECTS_HINT,
                param_type="array",
            )
            if plugin_config.llm_audio_effects
            else None,
            aux_refer_wav_paths_guide=ParameterGuide(
                description=prompts.AUX_REFER_WAV_PATHS_HINT,
                param_type="array",
            ),
        )


__all__ = ["TTSVoiceProvider"]
