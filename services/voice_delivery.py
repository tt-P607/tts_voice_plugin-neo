"""语音消息的 Base64 与本地文件 URL 发送。"""

from __future__ import annotations

import base64
from typing import Literal

from src.app.plugin_system.api.media_api import recognize_media
from src.app.plugin_system.api.send_api import send_custom, send_media, send_voice

from . import audio


async def send_voice_audio(
    audio_bytes: bytes,
    stream_id: str,
    use_base64: bool,
    wsl_mode: bool,
    context_text: str | None = None,
    context_source: Literal["text", "asr"] = "text",
) -> None:
    """按配置选择 Base64 或本地文件 URL 发送语音。

    Args:
        audio_bytes: WAV 音频字节。
        stream_id: 目标聊天流。
        use_base64: 是否直接发送 Base64 数据。
        wsl_mode: URL 模式下是否转换为 WSL 路径。
        context_text: 原始合成文本；为空时不附加语音上下文。
        context_source: 使用合成文本或识别音频作为上下文。
    """
    voice_data = base64.b64encode(audio_bytes).decode("utf-8")
    if use_base64:
        if context_text and context_source == "asr":
            await send_media("voice", voice_data, stream_id, context_mode="description")
        elif context_text:
            await send_voice(voice_data=voice_data, stream_id=stream_id, processed_plain_text=context_text)
        else:
            await send_voice(voice_data=voice_data, stream_id=stream_id)
        return

    if context_text and context_source == "asr":
        recognized = await recognize_media(voice_data, "voice")
        context_text = f"[语音:{recognized.strip()}]" if recognized and recognized.strip() else None

    file_path = await audio.write_temp_audio(audio_bytes)
    send_path = audio.to_wsl_path(str(file_path)) if wsl_mode else file_path.as_posix()
    voice_url = f"file://{send_path}" if wsl_mode else file_path.resolve().as_uri()
    try:
        await send_custom(
            content=voice_url,
            message_type="voiceurl",
            stream_id=stream_id,
            processed_plain_text=context_text or "[语音]",
        )
    finally:
        file_path.unlink(missing_ok=True)


__all__ = ["send_voice_audio"]