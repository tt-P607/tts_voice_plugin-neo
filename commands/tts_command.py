"""TTS 语音合成命令。

提供 /tts 命令，用户通过命令手动触发 GPT-SoVITS 语音合成。
支持语音条（/tts）和文件（/tts file）两种发送模式。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, cast

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.send_api import send_file, send_text, send_voice
from src.app.plugin_system.base import BaseCommand, cmd_route
from src.app.plugin_system.types import PermissionLevel
from src.kernel.concurrency import get_task_manager

if TYPE_CHECKING:
    from src.app.plugin_system.base import BasePlugin

    from ..plugin import TTSVoicePlugin
    from ..services.tts_service import TTSService

logger = get_logger("tts_voice_plugin-neo.command")

# 语言代码映射（GPT-SoVITS 原生代码 + 中文别名）
_LANG_MAP: dict[str, str] = {
    "zh": "zh",         "中文": "zh",     "中": "zh",
    "en": "en",         "英文": "en",     "英": "en",
    "ja": "ja",         "日文": "ja",     "日语": "ja", "日": "ja",
    "yue": "yue",       "粤语": "yue",   "粤": "yue",
    "ko": "ko",         "韩文": "ko",     "韩语": "ko", "韩": "ko",
    "all_zh": "all_zh", "纯中文": "all_zh",
    "all_ja": "all_ja", "纯日文": "all_ja",
    "all_yue": "all_yue", "纯粤语": "all_yue",
    "all_ko": "all_ko", "纯韩文": "all_ko",
    "zh_en": "zh_en",   "中英混合": "zh_en",
    "auto": "auto",     "自动": "auto",
    "auto_yue": "auto_yue", "自动粤语": "auto_yue",
}

_HELP_TEXT = (
    "请提供要转换为语音的文本哦！\n"
    "用法：/tts <文本> [风格] [语言]           ← 语音条\n"
    "      /tts file <文本> [风格] [语言]  ← 音频文件（无时长限制）\n"
    "语言代码：zh 中文 / all_zh 纯中文 / en 英文\n"
    "         ja 日文 / all_ja 纯日文 / yue 粤语 / all_yue 纯粤语\n"
    "         zh_en 中英混合 / auto 自动 / auto_yue 自动粤语\n"
    "（不填语言默认 zh，可用中文替代：纯中文/日文 等）"
)


class TTSVoiceCommand(BaseCommand):
    """通过 /tts 命令手动触发 TTS 语音合成。

    支持两种发送模式：
    - /tts <文本>            → 以语音条发送
    - /tts file <文本>       → 以音频文件发送（无时长限制）
    """

    command_name: str = "tts"
    command_description: str = (
        "使用GPT-SoVITS将文本转换为语音并发送，"
        "用法：/tts <文本> [风格] [语言] 或 /tts file <文本> [风格] [语言]"
    )
    permission_level: PermissionLevel = PermissionLevel.OPERATOR

    def _get_tts_service(self) -> "TTSService | None":
        """获取 TTSService 实例。

        Returns:
            TTSService 实例，未初始化时返回 None
        """
        return cast("TTSVoicePlugin", self.plugin).tts_service  # type: ignore[attr-defined]

    def _parse_words(
        self,
        available_styles: set[str],
        *words_args: str,
    ) -> tuple[list[str], str, str]:
        """解析词列表，提取文本、风格和语言。

        Args:
            available_styles: 可用风格名称集合
            *words_args: 原始词列表（含空字符串）

        Returns:
            (text_words, style_hint, language_hint) 三元组
        """
        words = [w for w in words_args if w]
        language_hint = "zh"
        style_hint = "default"

        if words and words[-1].lower() in _LANG_MAP:
            language_hint = _LANG_MAP[words[-1].lower()]
            words = words[:-1]

        if words and words[-1] in available_styles:
            style_hint = words[-1]
            words = words[:-1]

        return words, style_hint, language_hint

    @cmd_route()
    async def handle_tts(
        self,
        w0: str = "", w1: str = "", w2: str = "", w3: str = "",
        w4: str = "", w5: str = "", w6: str = "", w7: str = "",
    ) -> tuple[bool, str]:
        """以语音条发送 TTS 合成结果（/tts <文本> [风格] [语言]）。

        立即回复提示后在后台任务中生成并发送，避免事件超时。

        Returns:
            (是否成功, 结果描述)
        """
        tts_service = self._get_tts_service()
        if not tts_service:
            await send_text("❌ TTSService 未初始化，请检查插件配置。", stream_id=self.stream_id)
            return False, "TTSService 未注册或初始化失败"

        words, style_hint, language_hint = self._parse_words(
            set(tts_service.tts_styles.keys()), w0, w1, w2, w3, w4, w5, w6, w7
        )

        if not words:
            await send_text(_HELP_TEXT, stream_id=self.stream_id)
            return False, "缺少文本参数"

        text_to_speak = " ".join(words)
        stream_id = self.stream_id

        async def _do_tts_voice() -> None:
            try:
                audio_b64 = await tts_service.generate_voice(text_to_speak, style_hint, language_hint)
                if audio_b64:
                    await send_voice(voice_data=audio_b64, stream_id=stream_id)
                else:
                    await send_text("❌ 语音合成失败，请检查服务状态或配置。", stream_id=stream_id)
            except Exception as e:
                logger.error(f"后台 TTS 语音任务出错: {e}")
                await send_text("❌ 语音合成时发生了意想不到的错误，请查看日志。", stream_id=stream_id)

        get_task_manager().create_task(_do_tts_voice(), name="tts_voice_cmd")
        return True, "TTS 语音任务已提交"

    @cmd_route("file")
    async def handle_tts_file(
        self,
        w0: str = "", w1: str = "", w2: str = "", w3: str = "",
        w4: str = "", w5: str = "", w6: str = "", w7: str = "",
    ) -> tuple[bool, str]:
        """以音频文件发送 TTS 合成结果，不受时长限制（/tts file <文本> [风格] [语言]）。

        立即回复提示后在后台任务中生成并发送，避免事件超时。

        Returns:
            (是否成功, 结果描述)
        """
        tts_service = self._get_tts_service()
        if not tts_service:
            await send_text("❌ TTSService 未初始化，请检查插件配置。", stream_id=self.stream_id)
            return False, "TTSService 未注册或初始化失败"

        words, style_hint, language_hint = self._parse_words(
            set(tts_service.tts_styles.keys()), w0, w1, w2, w3, w4, w5, w6, w7
        )

        if not words:
            await send_text(_HELP_TEXT, stream_id=self.stream_id)
            return False, "缺少文本参数"

        text_to_speak = " ".join(words)
        stream_id = self.stream_id
        wsl_mode: bool = getattr(getattr(self.plugin.config, "tts", None), "wsl_mode", False)

        async def _do_tts_file() -> None:
            try:
                audio_bytes = await tts_service.generate_voice_bytes(text_to_speak, style_hint, language_hint)
                if not audio_bytes:
                    await send_text("❌ 语音合成失败，请检查服务状态或配置。", stream_id=stream_id)
                    return

                data_dir = os.path.abspath(os.path.join("data", "tts_voice_plugin-neo"))
                os.makedirs(data_dir, exist_ok=True)
                file_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
                file_path = os.path.join(data_dir, file_name)

                if wsl_mode:
                    path = file_path.replace("\\", "/")
                    if len(path) >= 2 and path[1] == ":":
                        send_path = f"/mnt/{path[0].lower()}{path[2:]}"
                    else:
                        send_path = path
                else:
                    send_path = file_path

                try:
                    with open(file_path, "wb") as f:
                        f.write(audio_bytes)
                    await send_file(file_path=send_path, stream_id=stream_id, file_name=file_name)
                except Exception as e:
                    logger.error(f"发送语音文件时出错: {e}")
                    await send_text("❌ 发送语音文件时发生错误，请查看日志。", stream_id=stream_id)
                finally:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
            except Exception as e:
                logger.error(f"后台 TTS 文件任务出错: {e}")
                await send_text("❌ 语音合成时发生了意想不到的错误，请查看日志。", stream_id=stream_id)

        get_task_manager().create_task(_do_tts_file(), name="tts_file_cmd")
        return True, "TTS 文件任务已提交"

