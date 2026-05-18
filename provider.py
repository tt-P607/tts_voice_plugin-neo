"""tts_voice_plugin-neo 的 TTS Provider 适配层。

使 tts_voice_plugin-neo 能够作为 provider 注册到 tts_http_server 中。
支持标准合成（返回完整音频）和流式合成（边接收边 yield 字节块）两种模式。
"""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from src.app.plugin_system.api.log_api import get_logger

if TYPE_CHECKING:
    from .services.tts_service import TTSService
    from plugins.tts_http_server.protocol import TTSSynthesisRequest, TTSSynthesisResponse

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

        audio_bytes = await self.tts_service.generate_voice_bytes(
            text=text,
            style_hint=style_hint,
            language_hint=language_hint,
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
