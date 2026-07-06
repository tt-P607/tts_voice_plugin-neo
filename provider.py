"""tts_voice_plugin-neo 的 TTS Provider 适配层。

使 tts_voice_plugin-neo 能够作为 provider 注册到 tts_http_server 中。
支持标准合成（返回完整音频）和流式合成（边接收边 yield 字节块）两种模式。
"""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.log_api import get_logger

if TYPE_CHECKING:
    from .services.tts_service import TTSService
    from plugins.tts_http_server.protocol import (
        TTSCapabilities,
        TTSSynthesisRequest,
        TTSSynthesisResponse,
    )

logger = get_logger("tts_voice_plugin-neo.provider")


class TTSVoiceProvider:
    """tts_voice_plugin-neo 的 TTS Provider 实现。

    实现了标准 TTSProvider 协议（synthesize），同时额外提供
    synthesize_stream 方法供 anima_chatter 的流式播放路径使用。
    """

    provider_name = "tts_voice_plugin-neo"

    def __init__(self, tts_service: "TTSService") -> None:
        """初始化 Provider。

        Args:
            tts_service: tts_voice_plugin-neo 的核心服务实例
        """
        self.tts_service = tts_service

    async def synthesize(self, request: "TTSSynthesisRequest") -> "TTSSynthesisResponse":
        """实现 tts_http_server 期望的合成接口（返回完整音频）。

        Args:
            request: TTSSynthesisRequest 实例 (来自 tts_http_server.protocol)

        Returns:
            TTSSynthesisResponse 实例
        """
        from plugins.tts_http_server.protocol import TTSSynthesisResponse

        text = request.text
        style_hint = request.options.get("style") or request.markers.get("style") or "default"

        # 语言提示：options.language > markers.language > None（让 service 走配置/自动检测）。
        # 接受 zh / en / ja / yue / auto 等取值，由下层 _normalize_language_code 校验。
        language_hint_raw = (
            request.options.get("language")
            or request.markers.get("language")
        )
        language_hint = (
            str(language_hint_raw).strip().lower() if language_hint_raw else None
        ) or None

        speed_raw = request.options.get("speed") or request.markers.get("speed")
        speed_factor: float | None = float(speed_raw) if speed_raw is not None else None
        effects_raw = (
            request.options.get("effects")
            or request.markers.get("effects")
            or request.options.get("audio_effects")
            or request.markers.get("audio_effects")
        )
        audio_effects: list[dict[str, Any]] | None = None
        if isinstance(effects_raw, list):
            # LLM 可能传 JSON 字符串而非 dict（tool call 序列化导致），
            # 逐项解析确保每个效果器是 dict
            import json

            audio_effects = []
            for item in effects_raw:
                if isinstance(item, dict):
                    audio_effects.append(item)
                elif isinstance(item, str):
                    try:
                        parsed = json.loads(item)
                        if isinstance(parsed, dict):
                            audio_effects.append(parsed)
                    except Exception:
                        pass
            if not audio_effects:
                audio_effects = None

        audio_bytes = await self.tts_service.generate_voice_bytes(
            text=text,
            style_hint=style_hint,
            language_hint=language_hint,
            speed_factor=speed_factor,
            audio_effects=audio_effects,
        )

        if not audio_bytes:
            raise RuntimeError("TTS 合成失败，未生成音频数据")

        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return TTSSynthesisResponse(
            audio_base64=audio_base64,
            mime_type="audio/wav",
            format="wav",
            text=text,
            provider=self.provider_name,
        )

    async def synthesize_stream(
        self,
        text: str,
        style_hint: str = "default",
        chunk_size: int = 4096,
    ) -> AsyncGenerator[bytes, None]:
        """流式合成接口，直接 yield GSV 返回的原始字节块。

        anima_chatter 的 SayAction 检测到此方法存在时会走流式路径，
        边接收边通过 sounddevice 播放，降低首字节延迟。

        Args:
            text: 要合成的文本
            style_hint: 风格名称提示
            chunk_size: 每次从 GSV 读取的字节块大小

        Yields:
            音频字节块（WAV 格式）
        """
        async for chunk in self.tts_service.generate_voice_stream(
            text=text,
            style_hint=style_hint,
            chunk_size=chunk_size,
        ):
            yield chunk

    def get_capabilities(self) -> "TTSCapabilities | None":
        """返回 GSV Provider 的完整参数定义（含类型、默认值、取值范围）。

        动态读取当前配置的可用风格列表、语言说明、语速 / 效果器开关，
        返回完整的参数元数据，让上游消费方（如 anima_chatter）的 action
        schema 能完全动态地复制参数定义，实现零硬编码。
        """

        from plugins.tts_http_server.protocol import TTSCapabilities, TTSParameterGuide

        from .actions.tts_action import _AUDIO_EFFECTS_HINT, _SPEED_FACTOR_HINT
        from .language import LANGUAGE_HELP_TEXT

        # ── style 参数：动态读取可用风格列表 ──
        available_styles = self.tts_service.get_available_styles()
        if available_styles:
            style_lines: list[str] = []
            for style_name in available_styles:
                style_cfg = self.tts_service.tts_styles.get(style_name, {})
                display_name = style_cfg.get("name", style_name)
                if display_name and display_name != style_name:
                    style_lines.append(f"  - '{style_name}' ({display_name})")
                else:
                    style_lines.append(f"  - '{style_name}'")
            style_desc = (
                "TTS 语音风格。请根据对话内容的实际情感选择相应风格，"
                "具体可用风格请参考下方的【当前可用语音风格】列表。"
                "如未提供则使用默认风格。\n\n"
                "【当前可用语音风格】（必须从以下列表中选择，传入字面量）：\n"
                + "\n".join(style_lines)
            )
            style_guide = TTSParameterGuide(
                description=style_desc,
                param_type="string",
                default="default",
                required=False,
                valid_values=available_styles,  # 提供枚举列表
            )
        else:
            style_guide = TTSParameterGuide(
                description="TTS 语音风格。当前无可用风格配置，使用默认值 'default'。",
                param_type="string",
                default="default",
                required=False,
            )

        # ── language 参数：完整定义 ──
        language_guide = TTSParameterGuide(
            description=LANGUAGE_HELP_TEXT,
            param_type="string",
            default="zh",  # 默认中文
            required=False,
        )

        # ── speed 参数：根据配置决定是否暴露 ──
        cfg = getattr(self.tts_service, "_config", None)
        speed_guide = None
        effects_guide = None
        if cfg is not None:
            if getattr(cfg.plugin, "llm_speed_control", False):
                speed_guide = TTSParameterGuide(
                    description=_SPEED_FACTOR_HINT,
                    param_type="number",
                    default=1.0,
                    required=False,
                    min_value=0.5,
                    max_value=2.0,
                )
            if getattr(cfg.plugin, "llm_audio_effects", False):
                effects_guide = TTSParameterGuide(
                    description=_AUDIO_EFFECTS_HINT,
                    param_type="array",
                    default=None,  # 不填则不应用效果器
                    required=False,
                )

        return TTSCapabilities(
            style_guide=style_guide,
            language_guide=language_guide,
            speed_guide=speed_guide,
            effects_guide=effects_guide,
        )
