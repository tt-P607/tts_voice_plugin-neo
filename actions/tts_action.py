"""TTS 语音合成 Action。

通过 LLM Tool Calling 或关键词自动触发 GPT-SoVITS 语音合成并发送语音消息。
支持多段语音顺序发送（voice 模式）和合并为文件发送（file 模式）。
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import TYPE_CHECKING, Annotated

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.send_api import send_file, send_voice
from src.core.components.base.action import BaseAction

if TYPE_CHECKING:
    from src.core.components.base.plugin import BasePlugin
    from src.core.models.stream import ChatStream

    from ..services.tts_service import TTSService

logger = get_logger("tts_voice_plugin-neo.action")


class TTSVoiceAction(BaseAction):
    """通过关键词或规划器自动触发 TTS 语音合成。

    支持分段语音顺序发送（voice 模式）和合并为音频文件发送（file 模式）。
    """

    action_name: str = "tts_voice_action"
    action_description: str = (
        "将文本转换为语音并发送。\n"
        "【发送模式选择规则】\n"
        "- voice 模式：逐条发送语音消息，每条语音最长只能约 1 分钟（QQ 硬限制），"
        "适合单条或总时长不超过 2～3 分钟的多段语音。\n"
        "- file 模式：将所有段合并为一个音频文件发送，可绕过 1 分钟限制，"
        "适合较长内容（如故事、长段独白等）。\n"
        "注意：这是纯语音合成，只能说话，不能唱歌！"
    )

    primary_action: bool = False

    def __init__(self, chat_stream: "ChatStream", plugin: "BasePlugin") -> None:
        """初始化 TTS 动作组件。

        Args:
            chat_stream: 聊天流实例
            plugin: 所属插件实例
        """
        super().__init__(chat_stream, plugin)
        self.tts_service: TTSService | None = getattr(self.plugin, "tts_service", None)

    # ------------------------------------------------------------------
    # 激活判定
    # ------------------------------------------------------------------

    async def go_activate(self) -> bool:
        """判断此 Action 是否应该被激活。

        满足以下任一条件即可激活：
        1. 25% 随机概率
        2. 匹配预设关键词
        3. LLM 判断当前场景适合发送语音

        Returns:
            是否激活
        """
        # 条件 1：随机激活
        if await self._random_activation(0.25):
            logger.info("TTSVoiceAction 随机激活成功 (25%)")
            return True

        # 条件 2：关键词激活
        keywords = [
            "发语音", "语音", "说句话", "用语音说", "听你", "听声音",
            "想你", "想听声音", "讲个话", "说段话", "念一下", "读一下",
            "用嘴说", "说", "能发语音吗", "亲口",
        ]
        if await self._keyword_match(keywords):
            logger.info("TTSVoiceAction 关键词激活成功")
            return True

        # 条件 3：LLM 判断激活
        if await self._llm_judge_activation():
            logger.info("TTSVoiceAction LLM 判断激活成功")
            return True

        logger.debug("TTSVoiceAction 所有激活条件均未满足，不激活")
        return False

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    @staticmethod
    def _to_wsl_path(win_path: str) -> str:
        """将 Windows 绝对路径转换为 WSL 挂载路径。

        例如：``C:\\foo\\bar`` → ``/mnt/c/foo/bar``

        Args:
            win_path: Windows 绝对路径

        Returns:
            WSL 格式的绝对路径
        """
        path = win_path.replace("\\", "/")
        if len(path) >= 2 and path[1] == ":":
            drive = path[0].lower()
            path = f"/mnt/{drive}{path[2:]}"
        return path

    async def execute(
        self,
        tts_voice_texts: Annotated[
            list[str],
            (
                "需要转换为语音并发送的文本列表，按发送顺序排列。\n"
                "- 发单条语音：传单元素列表，如 [\"你好！\"]\n"
                "- 发多条语音：按顺序传多元素列表，如 [\"第一段\", \"第二段\", \"第三段\"]\n"
                "【情感表达要求】：\n"
                "1. 善用标点符号传递情绪：感叹号表达惊讶兴奋、问号表达疑问好奇、省略号表达犹豫思考。\n"
                "2. 灵活使用语气词增强真实感：如惊讶(诶咦哇呀啊)、思考(嗯唔额)、撒娇(嘛呐嘻)。\n"
                "3. 避免不能辅助语气的符号：不要用括号标注动作、特殊符号(♪☆)等无法语音化的内容。"
            ),
        ],
        send_mode: Annotated[
            str,
            (
                "发送模式（默认 voice）：\n"
                "  voice — 逐条发送语音消息，每段独立一条。QQ 限制单条语音最长约 1 分钟，"
                "适合短语音或总时长不超过 2～3 分钟的分段内容。\n"
                "  file — 将所有片段合并为一个音频文件发送，可突破 1 分钟限制，"
                "适合故事、长段独白等较长内容。"
            ),
        ] = "voice",
        voice_style: Annotated[
            str,
            (
                "语音的风格。请根据对话内容的实际情感选择相应风格，"
                "具体可用风格请参考插件配置中的 tts_styles 列表。如未提供则使用默认风格。"
            ),
        ] = "default",
        text_language: Annotated[
            str | None,
            (
                "语音合成的语言模式，根据文本内容选择。只填代码本身，不填括号内的说明文字。\n"
                "混合模式（文本中包含多种语言或外来词时选此类）：\n"
                "  zh — 中文为主（夹杂英文）  en — 英文为主（夹杂其他语言）\n"
                "  ja — 日文为主（夹杂英文）  yue — 粤语（夹杂英文）\n"
                "  ko — 韩文（夹杂英文）      auto — 自动识别多语种\n"
                "  auto_yue — 自动识别（含粤语优先）\n"
                "纯语言模式（文本仅含单一语言时优先选此类，推理效果更好）：\n"
                "  all_zh — 纯中文  all_ja — 纯日文  all_yue — 纯粤语  all_ko — 纯韩文\n"
                "不填则沿用风格配置中的默认语言。"
            ),
        ] = None,
        file_name: Annotated[
            str | None,
            (
                "file 模式下发送的文件名（可选，仅 send_mode=file 时生效）。\n"
                "不填时默认使用时间戳命名（如 20260504_151230.wav）。\n"
                "填写时只需填文件名，不需要带扩展名，如 '晚安故事'。"
            ),
        ] = None,
    ) -> tuple[bool, str]:
        """执行 TTS 语音合成并发送。

        Args:
            tts_voice_texts: 要合成的文本列表
            send_mode: 发送模式，"voice" 或 "file"
            voice_style: 语音风格名称
            text_language: 语言模式
            file_name: file 模式下的自定义文件名

        Returns:
            (是否成功, 结果描述)
        """
        try:
            if not self.tts_service:
                logger.error("TTSService 未注册或初始化失败，静默处理。")
                return False, "TTSService 未注册或初始化失败"

            texts = [t.strip() for t in tts_voice_texts if t.strip()]
            if not texts:
                logger.warning("文本列表为空，静默处理。")
                return False, "文本列表为空"

            logger.info(
                f"接收到 {len(texts)} 段文本，发送模式: {send_mode}, 风格: {voice_style}"
            )

            if send_mode == "file":
                return await self._execute_file_mode(texts, voice_style, text_language, file_name)
            return await self._execute_voice_mode(texts, voice_style, text_language)

        except Exception as e:
            logger.error(f"语音合成过程中发生未知错误: {e!s}")
            return False, f"语音合成出错: {e!s}"

    async def _execute_voice_mode(
        self,
        texts: list[str],
        voice_style: str,
        text_language: str | None,
    ) -> tuple[bool, str]:
        """并行合成各段语音，按顺序逐条发送。

        Args:
            texts: 文本段列表
            voice_style: 语音风格
            text_language: 语言模式

        Returns:
            (是否成功, 结果描述)
        """
        tasks = [
            self.tts_service.generate_voice(  # type: ignore[union-attr]
                text=text,
                style_hint=voice_style,
                language_hint=text_language,
            )
            for text in texts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error(f"第 {i + 1} 段语音合成失败: {result}")
                continue
            if not isinstance(result, str):
                logger.error(f"第 {i + 1} 段语音合成返回空数据")
                continue
            await send_voice(voice_data=result, stream_id=self.chat_stream.stream_id)
            success_count += 1
            logger.info(f"第 {i + 1}/{len(texts)} 段语音发送成功")

        if success_count == 0:
            return False, "所有语音段均合成失败"
        total_len = sum(len(t) for t in texts)
        return True, f"成功发送 {success_count}/{len(texts)} 段语音，总文本长度: {total_len} 字符"

    async def _execute_file_mode(
        self,
        texts: list[str],
        voice_style: str,
        text_language: str | None,
        custom_file_name: str | None = None,
    ) -> tuple[bool, str]:
        """并行合成各段语音，合并后以文件形式发送。

        Args:
            texts: 文本段列表
            voice_style: 语音风格
            text_language: 语言模式
            custom_file_name: 自定义文件名，为 None 时使用时间戳命名

        Returns:
            (是否成功, 结果描述)
        """
        tasks = [
            self.tts_service.generate_voice_bytes(  # type: ignore[union-attr]
                text=text,
                style_hint=voice_style,
                language_hint=text_language,
            )
            for text in texts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        audio_list: list[bytes] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error(f"第 {i + 1} 段语音合成失败: {result}")
                continue
            if not isinstance(result, bytes):
                logger.error(f"第 {i + 1} 段语音合成返回空数据")
                continue
            audio_list.append(result)

        if not audio_list:
            return False, "所有语音段均合成失败"

        merged = self.tts_service.merge_audio_bytes(audio_list)  # type: ignore[union-attr]
        if not merged:
            return False, "音频合并失败"

        data_dir = os.path.abspath(os.path.join("data", "tts_voice_plugin-neo"))
        os.makedirs(data_dir, exist_ok=True)
        if custom_file_name:
            base = custom_file_name.removesuffix(".wav").removesuffix(".WAV")
            file_name = base + ".wav"
        else:
            file_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
        file_path = os.path.join(data_dir, file_name)

        # WSL 路径转换：Bot(Win) + napcat(WSL) 跨环境时启用
        cfg = getattr(self.plugin, "config", None)
        wsl_mode: bool = getattr(getattr(cfg, "tts", None), "wsl_mode", False)
        send_path = self._to_wsl_path(file_path) if wsl_mode else file_path

        try:
            with open(file_path, "wb") as f:
                f.write(merged)

            await send_file(
                file_path=send_path,
                stream_id=self.chat_stream.stream_id,
                file_name=file_name,
            )
            logger.info(f"合并音频文件发送成功，包含 {len(audio_list)} 段，大小: {len(merged)} 字节")
            total_len = sum(len(t) for t in texts)
            return True, f"成功合并 {len(audio_list)} 段并以文件发送，总文本长度: {total_len} 字符"
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)
