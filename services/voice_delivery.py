"""通过框架媒体缓存发送语音消息。"""

from __future__ import annotations

import base64
from typing import Literal

from src.app.plugin_system.api.send_api import send_media, send_voice


async def send_voice_audio(
    audio_bytes: bytes,
    stream_id: str,
    context_text: str | None = None,
    context_source: Literal["text", "asr"] = "text",
) -> bool:
    """发送可回查的语音媒体，并返回实际发送结果。

    Args:
        audio_bytes: WAV 音频字节。
        stream_id: 目标聊天流。
        context_text: 原始合成文本；为空时不附加语音上下文。
        context_source: 使用合成文本或识别音频作为上下文。

    Returns:
        缓存与平台发送均成功时返回 True。
    """
    voice_data = base64.b64encode(audio_bytes).decode("utf-8")
    if context_text and context_source == "asr":
        return await send_media("voice", voice_data, stream_id, context_mode="description")
    return await send_voice(
        voice_data=voice_data, stream_id=stream_id, processed_plain_text=context_text,
    )


__all__ = ["send_voice_audio"]