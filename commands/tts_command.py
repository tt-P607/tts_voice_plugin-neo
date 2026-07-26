"""TTS 语音合成命令。

提供 ``/tts`` 手动入口，支持语音条与音频文件两种发送模式。
合成在后台任务中进行，命令本身立即返回，避免阻塞事件处理。
"""

from __future__ import annotations

import base64
from datetime import datetime
from typing import cast

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.send_api import send_file, send_text, send_voice
from src.app.plugin_system.base import BaseCommand, cmd_route
from src.app.plugin_system.types import PermissionLevel
from src.kernel.concurrency import get_task_manager

from ..language import resolve_language_token
from ..protocol import TTSPluginLike
from ..services import audio

logger = get_logger("tts_voice_plugin-neo.command")

_HELP_TEXT = (
    "请提供要转换为语音的文本哦！\n"
    "用法：/tts <文本> [风格] [语言]           ← 语音条\n"
    "      /tts file <文本> [风格] [语言]  ← 音频文件（无时长限制）\n"
    "语言代码：zh 中文 / all_zh 纯中文 / en 英文\n"
    "         ja 日文 / all_ja 纯日文 / yue 粤语 / all_yue 纯粤语\n"
    "         zh_en 中英混合 / auto 自动 / auto_yue 自动粤语\n"
    "（不填语言默认 zh，可用中文替代：纯中文/日文 等）"
)
_SYNTHESIS_FAILED = "❌ 语音合成失败，请检查服务状态或配置。"
_UNEXPECTED_ERROR = "❌ 语音合成时发生了意想不到的错误，请查看日志。"


class TTSVoiceCommand(BaseCommand):
    """通过 ``/tts`` 手动触发语音合成。"""

    name: str = "tts"
    description: str = (
        "使用 GPT-SoVITS 将文本转换为语音并发送，"
        "用法：/tts <文本> [风格] [语言] 或 /tts file <文本> [风格] [语言]"
    )
    permission_level: PermissionLevel = PermissionLevel.OPERATOR

    @property
    def tts_plugin(self) -> TTSPluginLike:
        """所属插件实例。"""
        return cast(TTSPluginLike, self.plugin)

    @staticmethod
    def _parse_words(
        words: tuple[str, ...],
        available_styles: set[str],
    ) -> tuple[str, str, str]:
        """从词序列尾部剥离可选的风格与语言参数。

        Args:
            words: 原始词序列，可能包含空串。
            available_styles: 当前可用的风格名集合。

        Returns:
            ``(文本, 风格名, 语言代码)``；文本为空表示参数缺失。
        """
        tokens = [word for word in words if word]
        language = "zh"
        style = "default"

        if tokens:
            matched = resolve_language_token(tokens[-1])
            if matched is not None:
                language = matched
                tokens = tokens[:-1]

        if tokens and tokens[-1] in available_styles:
            style = tokens[-1]
            tokens = tokens[:-1]

        return " ".join(tokens), style, language

    async def _dispatch(
        self,
        words: tuple[str, ...],
        as_file: bool,
    ) -> tuple[bool, str]:
        """校验参数并提交后台合成任务。

        Args:
            words: 命令原始词序列。
            as_file: 是否以音频文件发送。

        Returns:
            ``(是否成功, 结果描述)``。
        """
        service = self.tts_plugin.tts_service
        if service is None:
            await send_text("❌ TTSService 未初始化，请检查插件配置。", stream_id=self.stream_id)
            return False, "TTSService 未注册或初始化失败"

        text, style, language = self._parse_words(words, set(service.get_available_styles()))
        if not text:
            await send_text(_HELP_TEXT, stream_id=self.stream_id)
            return False, "缺少文本参数"

        plugin = self.tts_plugin
        stream_id = self.stream_id
        wsl_mode = plugin.config.tts.wsl_mode
        purpose = "file" if as_file else "voice"
        task_id = ""

        async def _run() -> None:
            """在后台完成合成与发送，结束后注销任务登记。"""
            try:
                audio_bytes = await service.generate_voice_bytes(text, style, language)
                if not audio_bytes:
                    await send_text(_SYNTHESIS_FAILED, stream_id=stream_id)
                    return
                if as_file:
                    await self._send_file(audio_bytes, stream_id, wsl_mode)
                else:
                    await send_voice(
                        voice_data=base64.b64encode(audio_bytes).decode("utf-8"),
                        stream_id=stream_id,
                    )
            except Exception as error:
                logger.error(f"后台 TTS {purpose} 任务出错: {error}")
                await send_text(_UNEXPECTED_ERROR, stream_id=stream_id)
            finally:
                plugin.discard_command_task(task_id)

        task_info = get_task_manager().create_task(
            _run(),
            name=f"tts_{purpose}_cmd",
            daemon=True,
            metadata={
                "plugin": "tts_voice_plugin-neo",
                "purpose": f"command_{purpose}",
                "stream_id": stream_id,
            },
        )
        task_id = task_info.task_id
        plugin.register_command_task(task_id)
        return True, f"TTS {purpose} 任务已提交"

    @staticmethod
    async def _send_file(audio_bytes: bytes, stream_id: str, wsl_mode: bool) -> None:
        """把音频写入临时文件并发送，发送后删除。

        Args:
            audio_bytes: 音频字节。
            stream_id: 目标聊天流。
            wsl_mode: 是否把路径转换为 WSL 挂载形式。
        """
        file_path = await audio.write_temp_audio(
            audio_bytes, datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        send_path = audio.to_wsl_path(str(file_path)) if wsl_mode else str(file_path)
        try:
            await send_file(
                file_path=send_path,
                stream_id=stream_id,
                file_name=file_path.name,
            )
        finally:
            file_path.unlink(missing_ok=True)

    @cmd_route()
    async def handle_tts(
        self,
        w0: str = "", w1: str = "", w2: str = "", w3: str = "",
        w4: str = "", w5: str = "", w6: str = "", w7: str = "",
    ) -> tuple[bool, str]:
        """以语音条发送合成结果（``/tts <文本> [风格] [语言]``）。

        Returns:
            ``(是否成功, 结果描述)``。
        """
        return await self._dispatch((w0, w1, w2, w3, w4, w5, w6, w7), as_file=False)

    @cmd_route("file")
    async def handle_tts_file(
        self,
        w0: str = "", w1: str = "", w2: str = "", w3: str = "",
        w4: str = "", w5: str = "", w6: str = "", w7: str = "",
    ) -> tuple[bool, str]:
        """以音频文件发送合成结果（``/tts file <文本> [风格] [语言]``）。

        Returns:
            ``(是否成功, 结果描述)``。
        """
        return await self._dispatch((w0, w1, w2, w3, w4, w5, w6, w7), as_file=True)


__all__ = ["TTSVoiceCommand"]
