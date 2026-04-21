"""TTS 语音合成命令。

提供 /tts 命令，用户通过命令手动触发 GPT-SoVITS 语音合成。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.send_api import send_text, send_voice
from src.app.plugin_system.base import BaseCommand, cmd_route
from src.app.plugin_system.types import PermissionLevel

if TYPE_CHECKING:
    from src.app.plugin_system.base import BasePlugin

    from ..plugin import TTSVoicePlugin
    from ..services.tts_service import TTSService

logger = get_logger("tts_voice_plugin.command")


class TTSVoiceCommand(BaseCommand):
    """通过 /tts 命令手动触发 TTS 语音合成。"""

    command_name: str = "tts"
    command_description: str = "使用GPT-SoVITS将文本转换为语音并发送，用法：/tts <文本> [风格] [语言]"
    permission_level: PermissionLevel = PermissionLevel.OPERATOR

    @cmd_route()
    async def handle_tts(
        self,
        w0: str = "", w1: str = "", w2: str = "", w3: str = "",
        w4: str = "", w5: str = "", w6: str = "", w7: str = "",
    ) -> tuple[bool, str]:
        """TTS 语音合成（多词文本请使用引号，如 /tts "长文本" zh）。

        解析 `/tts <文本> [风格] [语言]`：
        - 风格：若末尾词为已知风格名则识别为风格，默认 "default"
        - 语言：若最后一个词为语言代码（zh/en/ja/mix/auto 等）则识别为语言，默认 "zh"
        - 其余词拼合为合成文本

        Returns:
            (是否成功, 结果描述)
        """
        words = [w for w in [w0, w1, w2, w3, w4, w5, w6, w7] if w]

        if not words:
            await send_text(
                "请提供要转换为语音的文本哦！\n"
                "用法：/tts <文本> [风格] [语言]\n"
                "语言代码：zh 中文 / all_zh 纯中文 / en 英文\n"
                "         ja 日文 / all_ja 纯日文 / yue 粤语 / all_yue 纯粤语\n"
                "         zh_en 中英混合 / auto 自动 / auto_yue 自动粤语\n"
                "（不填语言默认 zh，可用中文替代：纯中文/日文 等）",
                stream_id=self.stream_id,
            )
            return False, "缺少文本参数"

        tts_service: TTSService | None = cast("TTSVoicePlugin", self.plugin).tts_service
        if not tts_service:
            await send_text("❌ TTSService 未初始化，请检查插件配置。", stream_id=self.stream_id)
            return False, "TTSService 未注册或初始化失败"

        available_styles = set(tts_service.tts_styles.keys())
        # 语言代码：GPT-SoVITS 原生代码 + 中文别名
        lang_map: dict[str, str] = {
            # 通用代码（混合模式）
            "zh": "zh",         "中文": "zh",     "中": "zh",
            "en": "en",         "英文": "en",     "英": "en",
            "ja": "ja",         "日文": "ja",     "日语": "ja", "日": "ja",
            "yue": "yue",       "粤语": "yue",   "粤": "yue",
            "ko": "ko",         "韩文": "ko",     "韩语": "ko", "韩": "ko",
            # 纯语言代码（all_ 前缀）
            "all_zh": "all_zh", "纯中文": "all_zh",

            "all_ja": "all_ja", "纯日文": "all_ja",
            "all_yue": "all_yue", "纯粤语": "all_yue",
            "all_ko": "all_ko", "纯韩文": "all_ko",
            # 混合/自动
            "zh_en": "zh_en",   "中英混合": "zh_en",
            "auto": "auto",     "自动": "auto",
            "auto_yue": "auto_yue", "自动粤语": "auto_yue",
        }

        language_hint = "zh"  # 默认语言
        style_hint = "default"  # 默认风格

        # 检测语言代码（最后一个词）
        if words and words[-1].lower() in lang_map:
            language_hint = lang_map[words[-1].lower()]
            words = words[:-1]

        # 检测风格名（去掉语言后的最后一个词）
        if words and words[-1] in available_styles:
            style_hint = words[-1]
            words = words[:-1]

        text_to_speak = " ".join(words)

        if not text_to_speak:
            await send_text("请提供要转换为语音的文本内容哦！", stream_id=self.stream_id)
            return False, "文本内容为空"

        try:
            audio_b64 = await tts_service.generate_voice(text_to_speak, style_hint, language_hint)

            if audio_b64:
                await send_voice(voice_data=audio_b64, stream_id=self.stream_id)
                return True, "语音发送成功"
            else:
                await send_text("❌ 语音合成失败，请检查服务状态或配置。", stream_id=self.stream_id)
                return False, "语音合成失败"

        except Exception as e:
            logger.error(f"执行 /tts 命令时出错: {e}")
            await send_text("❌ 语音合成时发生了意想不到的错误，请查看日志。", stream_id=self.stream_id)
            return False, "命令执行异常"

