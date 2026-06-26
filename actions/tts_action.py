"""TTS 语音合成 Action。

通过 LLM Tool Calling 或关键词自动触发 GPT-SoVITS 语音合成并发送语音消息。
支持多段语音顺序发送（voice 模式）和合并为文件发送（file 模式）。
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import wave
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.send_api import send_file, send_voice
from src.core.components.base.action import BaseAction

from ..language import LANGUAGE_HELP_TEXT

if TYPE_CHECKING:
    from src.core.components.base.plugin import BasePlugin
    from src.core.models.stream import ChatStream

    from ..config import TTSVoiceConfig
    from ..services.tts_service import TTSService

logger = get_logger("tts_voice_plugin-neo.action")

# 中文语速约 5 字/秒，30 秒对应约 150 字
_CHARS_PER_SECOND = 5.0


class TTSVoiceAction(BaseAction):
    """通过关键词或规划器自动触发 TTS 语音合成。

    支持分段语音顺序发送（voice 模式）和合并为音频文件发送（file 模式）。
    """

    action_name: str = "tts_voice_action"
    associated_types: list[str] = ["voice"]
    action_description: str = (
        "将文本转换为语音并发送。\n"
        "【发送模式】默认使用 voice 模式发送语音条，超出单条时长限制时分多条发送。"
        "file 模式仅在内容超长或用户明确要求时使用。\n"
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

        # 条件 2：关键词激活（从配置读取）
        cfg: TTSVoiceConfig | None = getattr(self.plugin, "config", None)  # type: ignore[assignment]
        keywords = cfg.plugin.keywords if cfg else []
        if keywords and await self._keyword_match(keywords):
            logger.info("TTSVoiceAction 关键词激活成功")
            return True

        # 条件 3：LLM 判断激活
        if await self._llm_judge_activation():
            logger.info("TTSVoiceAction LLM 判断激活成功")
            return True

        logger.debug("TTSVoiceAction 所有激活条件均未满足，不激活")
        return False

    # ------------------------------------------------------------------
    # 工具方法
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

    @staticmethod
    def _estimate_audio_duration_from_wav(audio_data: bytes) -> float:
        """从 WAV 字节数据估算音频时长（秒）。

        Args:
            audio_data: WAV 格式音频字节

        Returns:
            音频时长秒数，解析失败时根据文本长度估算
        """
        try:
            with wave.open(io.BytesIO(audio_data)) as wf:  # type: ignore[arg-type]
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return frames / rate
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _compute_send_interval(duration: float) -> float:
        """根据音频时长计算发送间隔，模拟真人语音消息的发送节奏。

        规则：
        - 时长 < 3s：间隔 0.8s
        - 时长 3~10s：间隔 1.5s
        - 时长 10~30s：间隔 2.5s
        - 时长 > 30s：间隔 3.5s

        Args:
            duration: 音频时长（秒）

        Returns:
            建议的发送间隔（秒）
        """
        if duration < 3.0:
            return 0.8
        if duration < 10.0:
            return 1.5
        if duration < 30.0:
            return 2.5
        return 3.5

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    async def execute(
        self,
        tts_voice_texts: Annotated[
            list[str],
            (
                "需要转换为语音并发送的文本列表，按发送顺序排列。\n"
                "【分段规则 — 极其重要】：\n"
                "- 默认不分段！能放一条语音里说完的内容就不要拆开，直接传单元素列表\n"
                "- 只有内容确实很长（说出来会超过 30 秒）时才拆分为多段\n"
                "- 拆分时在语义转折、话题切换或情绪变化的自然断点处分开，"
                "不要在句子中间生硬截断\n"
                "- 错误示范：[\"嗯...\", \"我想想\", \"好吧\"] ← 太碎了！\n"
                "- 正确示范：[\"嗯...我想想，好吧，那我就跟你说说这件事的来龙去脉吧。\"] ← 合为一段\n"
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
                "  voice — 发送语音条消息，单条最长约 1 分钟，超出时自动分为多条语音条依次发送。"
                "日常对话默认使用此模式。\n"
                "  file — 合并为单个音频文件发送，无时长限制。"
                "适用于内容超长或需要正式发送的场景，仅在用户明确要求时使用。"
            ),
        ] = "voice",
        voice_style: Annotated[
            str,
            (
                "语音的风格。请根据对话内容的实际情感选择相应风格，"
                "具体可用风格请参考下方的【当前可用语音风格】列表。如未提供则使用默认风格。"
            ),
        ] = "default",
        text_language: Annotated[
            str | None,
            LANGUAGE_HELP_TEXT,
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
        return await self._do_execute(tts_voice_texts, send_mode, voice_style, text_language, file_name)

    async def _do_execute(
        self,
        tts_voice_texts: list[str],
        send_mode: str = "voice",
        voice_style: str = "default",
        text_language: str | None = None,
        file_name: str | None = None,
        speed_factor: float | None = None,
        audio_effects: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, str]:
        """实际执行 TTS 语音合成的内部方法。

        Args:
            tts_voice_texts: 要合成的文本列表
            send_mode: 发送模式
            voice_style: 语音风格名称
            text_language: 语言模式
            file_name: file 模式下的自定义文件名
            speed_factor: 语速因子，None 时使用配置默认值
            audio_effects: 音频效果器链，None 时使用配置的效果器链

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
                + (f", 语速: {speed_factor}" if speed_factor is not None else "")
                + (f", 效果器: {[e.get('type', '?') for e in audio_effects]}" if audio_effects else "")
            )

            if send_mode == "file":
                return await self._execute_file_mode(
                    texts, voice_style, text_language, file_name, speed_factor, audio_effects,
                )
            return await self._execute_voice_mode(texts, voice_style, text_language, speed_factor, audio_effects)

        except Exception as e:
            logger.error(f"语音合成过程中发生未知错误: {e!s}")
            return False, f"语音合成出错: {e!s}"

    async def _execute_voice_mode(
        self,
        texts: list[str],
        voice_style: str,
        text_language: str | None,
        speed_factor: float | None = None,
        audio_effects: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, str]:
        """并行合成各段语音，按顺序逐条发送，每条之间添加自适应间隔。

        间隔时长根据上一条语音的实际时长动态计算，模拟真人发语音的节奏。

        Args:
            texts: 文本段列表
            voice_style: 语音风格
            text_language: 语言模式
            speed_factor: 语速因子，None 时使用配置默认值
            audio_effects: 音频效果器链，None 时使用配置的效果器链

        Returns:
            (是否成功, 结果描述)
        """
        # 使用 generate_voice_bytes 获取原始字节，以便计算时长
        tasks = [
            self.tts_service.generate_voice_bytes(  # type: ignore[union-attr]
                text=text,
                style_hint=voice_style,
                language_hint=text_language,
                speed_factor=speed_factor,
                audio_effects=audio_effects,
            )
            for text in texts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        prev_duration: float = 0.0
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error(f"第 {i + 1} 段语音合成失败: {result}")
                continue
            if not isinstance(result, bytes):
                logger.error(f"第 {i + 1} 段语音合成返回空数据")
                continue

            # 发送前添加自适应间隔（第一条不等待）
            if success_count > 0:
                interval = self._compute_send_interval(prev_duration)
                logger.debug(f"第 {i + 1} 段发送前等待 {interval:.1f}s（上一段时长 {prev_duration:.1f}s）")
                await asyncio.sleep(interval)

            # 转 base64 发送
            voice_b64 = base64.b64encode(result).decode("utf-8")
            await send_voice(voice_data=voice_b64, stream_id=self.chat_stream.stream_id)
            success_count += 1

            # 记录当前段时长，供下一段计算间隔
            prev_duration = self._estimate_audio_duration_from_wav(result)
            if prev_duration <= 0:
                prev_duration = len(texts[i]) / _CHARS_PER_SECOND

            logger.info(f"第 {i + 1}/{len(texts)} 段语音发送成功（时长约 {prev_duration:.1f}s）")

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
        speed_factor: float | None = None,
        audio_effects: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, str]:
        """并行合成各段语音，合并后以文件形式发送。

        Args:
            texts: 文本段列表
            voice_style: 语音风格
            text_language: 语言模式
            custom_file_name: 自定义文件名，为 None 时使用时间戳命名
            speed_factor: 语速因子，None 时使用配置默认值
            audio_effects: 音频效果器链，None 时使用配置的效果器链

        Returns:
            (是否成功, 结果描述)
        """
        tasks = [
            self.tts_service.generate_voice_bytes(  # type: ignore[union-attr]
                text=text,
                style_hint=voice_style,
                language_hint=text_language,
                speed_factor=speed_factor,
                audio_effects=audio_effects,
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


# ------------------------------------------------------------------
# Annotated 提示词常量（避免重复）
# ------------------------------------------------------------------

_TTS_VOICE_TEXTS_HINT = (
    "需要转换为语音并发送的文本列表，按发送顺序排列。\n"
    "【分段规则 — 极其重要】：\n"
    "- 默认不分段！能放一条语音里说完的内容就不要拆开，直接传单元素列表\n"
    "- 只有内容确实很长（说出来会超过 30 秒）时才拆分为多段\n"
    "- 拆分时在语义转折、话题切换或情绪变化的自然断点处分开，"
    "不要在句子中间生硬截断\n"
    "- 错误示范：[\"嗯...\", \"我想想\", \"好吧\"] ← 太碎了！\n"
    "- 正确示范：[\"嗯...我想想，好吧，那我就跟你说说这件事的来龙去脉吧。\"] ← 合为一段\n"
    "【情感表达要求】：\n"
    "1. 善用标点符号传递情绪：感叹号表达惊讶兴奋、问号表达疑问好奇、省略号表达犹豫思考。\n"
    "2. 灵活使用语气词增强真实感：如惊讶(诶咦哇呀啊)、思考(嗯唔额)、撒娇(嘛呐嘻)。\n"
    "3. 避免不能辅助语气的符号：不要用括号标注动作、特殊符号(♪☆)等无法语音化的内容。"
)

_SEND_MODE_HINT = (
    "发送模式（默认 voice）：\n"
    "  voice — 发送语音条消息，单条最长约 1 分钟，超出时自动分为多条语音条依次发送。"
    "日常对话默认使用此模式。\n"
    "  file — 合并为单个音频文件发送，无时长限制。"
    "适用于内容超长或需要正式发送的场景，仅在用户明确要求时使用。"
)

_VOICE_STYLE_HINT = (
    "语音的风格。请根据对话内容的实际情感选择相应风格，"
    "具体可用风格请参考下方的【当前可用语音风格】列表。如未提供则使用默认风格。"
)

_SPEED_FACTOR_HINT = (
    "语速因子（可选）。控制语音播放速度，不填则使用默认语速。\n"
    "取值范围限制在 0.85 ~ 1.2 之间，根据对话情绪氛围自行判断，不确定时不要填写。"
)

_FILE_NAME_HINT = (
    "file 模式下发送的文件名（可选，仅 send_mode=file 时生效）。\n"
    "不填时默认使用时间戳命名（如 20260504_151230.wav）。\n"
    "填写时只需填文件名，不需要带扩展名，如 '晚安故事'。"
)

_AUDIO_EFFECTS_HINT = (
    "语音后处理效果器链（可选）。模拟真人使用麦克风效果器处理语音。\n"
    "不填则发送原始语音。传入效果器列表，按顺序依次处理音频。\n"
    "每个效果器为一个对象，必须包含 type 字段，其余为该效果器的参数。\n"
    "\n"
    "【基础效果器】\n"
    "  reverb — 混响，模拟不同空间环境的声音反射。\n"
    "    room_size: 0.0-1.0（0.2 小房间，0.5 中厅，0.9 大教堂）。\n"
    "    wet_level: 0.0-1.0，混响占比。damping: 0.0-1.0，高频衰减。\n"
    "\n"
    "  highpass — 高通滤波器，切除低频。cutoff_hz: Hz（300 轻微，800 电话，2000 对讲机）。\n"
    "  lowpass — 低通滤波器，切除高频。cutoff_hz: Hz（8000 轻微，3000 隔墙，1000 闷声）。\n"
    "  pitch_shift — 变调。semitones: 半音数(-12~12)。\n"
    "  distortion — 失真。drive_db: 0-30dB。\n"
    "  bitcrush — 位深压缩，复古数码感。bit_depth: 4-16（16 无损，8 复古，4 极低保真）。\n"
    "  delay — 延迟回声。delay_seconds: 0.05-1.0s。feedback: 0.0-0.8。mix: 0.0-1.0。\n"
    "  chorus — 合唱效果，产生厚度和飘动感。rate_hz: 0.5-5.0。depth: 0.0-1.0。mix: 0.0-1.0。\n"
    "  gain — 音量调节。gain_db: -20~20dB。\n"
    "\n"
    "【高级效果器】\n"
    "  phaser — 相位器，产生飘忽的扫频效果，类似颤动。\n"
    "    rate_hz: 调制频率(0.1-5.0Hz)。depth: 0.0-1.0。feedback: -1.0~1.0。mix: 0.0-1.0。\n"
    "  compressor — 压缩器，控制动态范围，让轻声更响、响声更柔。\n"
    "    threshold_db: 触发阈值(dB)。ratio: 压缩比(1.0-20.0)。\n"
    "  clipping — 硬削波，产生暴力失真。threshold_db: 削波阈值(-20~0dB)。\n"
    "  noise_gate — 噪声门，低于阈值的声音被静音，产生断断续续效果。\n"
    "    threshold_db: 门限阈值(dB)。ratio: 衰减比。\n"
    "  ladder_filter — 共振滤波器，可产生 wah-wah、酸性电子音色。\n"
    "    cutoff_hz: 截止频率。resonance: 共振(0.0-1.0，越高越尖锐)。drive: 驱动(≥1.0)。\n"
    "  resample — 重采样降质，模拟低采样率设备的粗糙感。\n"
    "    target_sample_rate: 目标采样率(Hz)。8000=电话，4000=对讲机，2000=极低保真。\n"
    "  gsm — GSM 电话编码压缩，直接模拟真实 2G 手机通话音质，无额外参数。\n"
    "  mp3 — MP3 有损压缩。vbr_quality: 0.0-9.0（0 最高质量，9 最低质量/最大失真）。\n"
    "\n"
    "【场景组合参考】\n"
    "  电话: [{\"type\":\"highpass\",\"cutoff_hz\":800},{\"type\":\"lowpass\",\"cutoff_hz\":3000}]\n"
    "  2G手机: [{\"type\":\"gsm\"}]\n"
    "  远处喊话: [{\"type\":\"lowpass\",\"cutoff_hz\":2000},{\"type\":\"reverb\",\"room_size\":0.7,\"wet_level\":0.5}]\n"
    "  广播: [{\"type\":\"highpass\",\"cutoff_hz\":500},{\"type\":\"lowpass\",\"cutoff_hz\":4000},{\"type\":\"distortion\",\"drive_db\":5}]\n"
    "  耳边悄悄话: [{\"type\":\"lowpass\",\"cutoff_hz\":6000},{\"type\":\"reverb\",\"room_size\":0.1,\"wet_level\":0.15},{\"type\":\"gain\",\"gain_db\":-3}]\n"
    "  过山车颤音: [{\"type\":\"phaser\",\"rate_hz\":3.0,\"depth\":0.8,\"feedback\":0.6,\"mix\":0.7},{\"type\":\"chorus\",\"rate_hz\":4.0,\"depth\":0.8,\"mix\":0.5}]\n"
    "  磁带卡带: [{\"type\":\"bitcrush\",\"bit_depth\":10},{\"type\":\"resample\",\"target_sample_rate\":11025},{\"type\":\"chorus\",\"rate_hz\":0.5,\"depth\":0.3,\"mix\":0.2}]\n"
    "  水下: [{\"type\":\"lowpass\",\"cutoff_hz\":800},{\"type\":\"chorus\",\"rate_hz\":0.8,\"depth\":0.6,\"mix\":0.4}]\n"
    "  机器人: [{\"type\":\"bitcrush\",\"bit_depth\":6},{\"type\":\"distortion\",\"drive_db\":8},{\"type\":\"resample\",\"target_sample_rate\":8000}]\n"
    "  大教堂: [{\"type\":\"reverb\",\"room_size\":0.95,\"wet_level\":0.7,\"damping\":0.3}]\n"
    "  对讲机: [{\"type\":\"highpass\",\"cutoff_hz\":1500},{\"type\":\"lowpass\",\"cutoff_hz\":3500},{\"type\":\"clipping\",\"threshold_db\":-8},{\"type\":\"noise_gate\",\"threshold_db\":-35}]"
)


class TTSVoiceSpeedAction(TTSVoiceAction):
    """带语速控制的 TTS 语音合成 Action。

    继承 TTSVoiceAction，在 execute 签名中额外暴露 speed_factor 参数，
    使 LLM 可以根据对话情绪自主调节语速。仅在 llm_speed_control 开关开启时注册。
    """

    async def execute(
        self,
        tts_voice_texts: Annotated[list[str], _TTS_VOICE_TEXTS_HINT],
        send_mode: Annotated[str, _SEND_MODE_HINT] = "voice",
        voice_style: Annotated[str, _VOICE_STYLE_HINT] = "default",
        speed_factor: Annotated[float | None, _SPEED_FACTOR_HINT] = None,
        text_language: Annotated[str | None, LANGUAGE_HELP_TEXT] = None,
        file_name: Annotated[str | None, _FILE_NAME_HINT] = None,
    ) -> tuple[bool, str]:
        """执行 TTS 语音合成并发送（带语速控制）。

        Args:
            tts_voice_texts: 要合成的文本列表
            send_mode: 发送模式，"voice" 或 "file"
            voice_style: 语音风格名称
            speed_factor: 语速因子，None 时使用配置默认值
            text_language: 语言模式
            file_name: file 模式下的自定义文件名

        Returns:
            (是否成功, 结果描述)
        """
        return await self._do_execute(
            tts_voice_texts, send_mode, voice_style, text_language, file_name, speed_factor,
        )


class TTSVoiceEffectsAction(TTSVoiceAction):
    """带音频效果器控制的 TTS 语音合成 Action。

    继承 TTSVoiceAction，在 execute 签名中额外暴露 audio_effects 参数，
    使 LLM 可以自主添加音频后处理效果。仅在 llm_audio_effects 开关开启时注册。
    """

    async def execute(
        self,
        tts_voice_texts: Annotated[list[str], _TTS_VOICE_TEXTS_HINT],
        send_mode: Annotated[str, _SEND_MODE_HINT] = "voice",
        voice_style: Annotated[str, _VOICE_STYLE_HINT] = "default",
        text_language: Annotated[str | None, LANGUAGE_HELP_TEXT] = None,
        file_name: Annotated[str | None, _FILE_NAME_HINT] = None,
        audio_effects: Annotated[list[dict[str, Any]] | None, _AUDIO_EFFECTS_HINT] = None,
    ) -> tuple[bool, str]:
        """执行 TTS 语音合成并发送（带音频效果器）。

        Args:
            tts_voice_texts: 要合成的文本列表
            send_mode: 发送模式，"voice" 或 "file"
            voice_style: 语音风格名称
            text_language: 语言模式
            file_name: file 模式下的自定义文件名
            audio_effects: 音频效果器链

        Returns:
            (是否成功, 结果描述)
        """
        return await self._do_execute(
            tts_voice_texts, send_mode, voice_style, text_language, file_name,
            audio_effects=audio_effects,
        )


class TTSVoiceFullAction(TTSVoiceAction):
    """同时暴露语速和音频效果器控制的 TTS 语音合成 Action。

    继承 TTSVoiceAction，在 execute 签名中同时暴露 speed_factor 和 audio_effects 参数。
    仅在 llm_speed_control 和 llm_audio_effects 同时开启时注册。
    """

    async def execute(
        self,
        tts_voice_texts: Annotated[list[str], _TTS_VOICE_TEXTS_HINT],
        send_mode: Annotated[str, _SEND_MODE_HINT] = "voice",
        voice_style: Annotated[str, _VOICE_STYLE_HINT] = "default",
        speed_factor: Annotated[float | None, _SPEED_FACTOR_HINT] = None,
        text_language: Annotated[str | None, LANGUAGE_HELP_TEXT] = None,
        file_name: Annotated[str | None, _FILE_NAME_HINT] = None,
        audio_effects: Annotated[list[dict[str, Any]] | None, _AUDIO_EFFECTS_HINT] = None,
    ) -> tuple[bool, str]:
        """执行 TTS 语音合成并发送（带语速控制和音频效果器）。

        Args:
            tts_voice_texts: 要合成的文本列表
            send_mode: 发送模式，"voice" 或 "file"
            voice_style: 语音风格名称
            speed_factor: 语速因子，None 时使用配置默认值
            text_language: 语言模式
            file_name: file 模式下的自定义文件名
            audio_effects: 音频效果器链

        Returns:
            (是否成功, 结果描述)
        """
        return await self._do_execute(
            tts_voice_texts, send_mode, voice_style, text_language, file_name,
            speed_factor, audio_effects,
        )
