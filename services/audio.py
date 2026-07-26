"""WAV 音频的拼接、时长估算与临时文件落盘。

这里的函数都不依赖插件配置，只处理字节与路径；涉及磁盘或 CPU 的操作
统一通过 :func:`asyncio.to_thread` 暴露异步版本，避免阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import tempfile
import wave
from pathlib import Path
from typing import Final

from src.app.plugin_system.api.log_api import get_logger

logger = get_logger("tts_voice_plugin-neo.audio")

DATA_DIR: Final[Path] = Path("data/tts_voice_plugin-neo")
_FILE_NAME_STEM_LIMIT: Final[int] = 80


def estimate_duration(audio_data: bytes) -> float:
    """估算 WAV 音频时长。

    Args:
        audio_data: WAV 格式音频字节。

    Returns:
        时长秒数；无法解析时返回 ``0.0``。
    """
    try:
        with wave.open(io.BytesIO(audio_data)) as wav_file:  # type: ignore[arg-type]
            rate = wav_file.getframerate()
            if rate > 0:
                return wav_file.getnframes() / rate
    except (wave.Error, EOFError):
        logger.debug("音频时长估算失败：不是可解析的 WAV 数据")
    return 0.0


def to_wsl_path(path: str) -> str:
    """把 Windows 绝对路径转换为 WSL 挂载路径。

    Args:
        path: 原始路径字符串。

    Returns:
        WSL 形式的路径；非盘符路径原样返回（仅统一分隔符）。
    """
    normalized = path.replace("\\", "/")
    if len(normalized) >= 2 and normalized[1] == ":":
        return f"/mnt/{normalized[0].lower()}{normalized[2:]}"
    return normalized


def sanitize_file_name(file_name: str | None) -> str:
    """生成安全的 WAV 文件名，剥离目录成分与非法字符。

    Args:
        file_name: 调用方期望的文件名，可为空。

    Returns:
        以 ``.wav`` 结尾的安全文件名。
    """
    stem = Path((file_name or "").strip()).name
    if stem.lower().endswith(".wav"):
        stem = stem[:-4]
    stem = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", stem, flags=re.UNICODE).strip("_.")
    if not stem:
        stem = "tts_voice"
    return f"{stem[:_FILE_NAME_STEM_LIMIT]}.wav"


def _write_temp_audio(audio_data: bytes, file_name: str | None) -> Path:
    """在插件数据目录写入唯一临时 WAV 文件。

    Args:
        audio_data: 音频字节。
        file_name: 期望的文件名，可为空。

    Returns:
        写入后的文件路径。
    """
    data_dir = DATA_DIR.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{Path(sanitize_file_name(file_name)).stem}_"
    descriptor, raw_path = tempfile.mkstemp(prefix=prefix, suffix=".wav", dir=data_dir)
    with os.fdopen(descriptor, "wb") as output:
        output.write(audio_data)
    return Path(raw_path)


async def write_temp_audio(audio_data: bytes, file_name: str | None = None) -> Path:
    """异步写入临时 WAV 文件。

    Args:
        audio_data: 音频字节。
        file_name: 期望的文件名，可为空。

    Returns:
        写入后的文件路径。
    """
    return await asyncio.to_thread(_write_temp_audio, audio_data, file_name)


def _merge_audio(audio_list: list[bytes], pause_duration: float) -> bytes:
    """同步拼接多段 WAV，可在段间插入静音。

    Args:
        audio_list: 多段 WAV 字节，至少两段。
        pause_duration: 段间静音秒数。

    Returns:
        拼接后的 WAV 字节。
    """
    output = io.BytesIO()
    with wave.open(output, "wb") as out_wav:
        silence_frames = b""
        for index, audio_bytes in enumerate(audio_list):
            with wave.open(io.BytesIO(audio_bytes)) as in_wav:  # type: ignore[arg-type]
                if index == 0:
                    out_wav.setparams(in_wav.getparams())
                    if pause_duration > 0:
                        frame_count = int(in_wav.getframerate() * pause_duration)
                        silence_frames = b"\x00" * (
                            frame_count * in_wav.getsampwidth() * in_wav.getnchannels()
                        )
                elif silence_frames:
                    out_wav.writeframes(silence_frames)
                out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
    return output.getvalue()


async def merge_audio(
    audio_list: list[bytes],
    pause_duration: float = 0.0,
) -> bytes | None:
    """把多段 WAV 合并为一段，可在段间插入静音。

    Args:
        audio_list: 多段 WAV 字节。
        pause_duration: 段间静音秒数，``0`` 表示直接拼接。

    Returns:
        合并后的 WAV 字节；列表为空或合并失败时返回 ``None``。
    """
    if not audio_list:
        return None
    if len(audio_list) == 1:
        return audio_list[0]
    try:
        return await asyncio.to_thread(_merge_audio, audio_list, pause_duration)
    except (wave.Error, EOFError, ValueError) as error:
        logger.error(f"音频合并失败: {error}")
        return None


__all__ = [
    "DATA_DIR",
    "estimate_duration",
    "merge_audio",
    "sanitize_file_name",
    "to_wsl_path",
    "write_temp_audio",
]
