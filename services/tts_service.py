"""TTS 核心服务。

封装 GPT-SoVITS 语音合成的核心逻辑，包括风格管理、文本清洗、API 调用和空间音效处理。
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import re
from typing import TYPE_CHECKING, Any

import aiohttp
import soundfile as sf
from pedalboard import Convolution, Pedalboard, Reverb
from pedalboard.io import AudioFile

from src.app.plugin_system.api.log_api import get_logger
from src.core.components.base.service import BaseService

if TYPE_CHECKING:
    from src.core.components.base.plugin import BasePlugin

    from ..config import TTSVoiceConfig

logger = get_logger("tts_voice_plugin.service")


class TTSService(BaseService):
    """GPT-SoVITS TTS 核心服务。"""

    service_name: str = "tts"
    service_description: str = "GPT-SoVITS 语音合成服务"
    version: str = "3.1.2"

    def __init__(self, plugin: "BasePlugin") -> None:
        """初始化 TTS 服务。

        Args:
            plugin: 所属插件实例
        """
        super().__init__(plugin)
        self.tts_styles: dict[str, dict[str, Any]] = {}
        self.timeout: int = 60
        self.max_text_length: int = 500
        self._load_config()

    # ------------------------------------------------------------------
    # 配置加载
    # ------------------------------------------------------------------

    @property
    def _config(self) -> "TTSVoiceConfig":
        """获取当前插件配置的快捷属性。"""
        return self.plugin.config  # type: ignore[return-value]

    def _load_config(self) -> None:
        """从插件配置加载 TTS 参数。"""
        try:
            cfg = self._config
            self.timeout = cfg.tts.timeout
            self.max_text_length = cfg.tts.max_text_length
            self.tts_styles = self._load_tts_styles()

            if self.tts_styles:
                logger.info(f"TTS服务已成功加载风格: {list(self.tts_styles.keys())}")
            else:
                logger.warning("TTS风格配置为空，请检查配置文件")
        except Exception as e:
            logger.error(f"TTS服务配置加载失败: {e}")

    def _load_tts_styles(self) -> dict[str, dict[str, Any]]:
        """加载 TTS 风格配置。"""
        styles: dict[str, dict[str, Any]] = {}
        cfg = self._config
        global_server = cfg.tts.server
        tts_styles_list = cfg.tts_styles

        if not tts_styles_list:
            logger.error("tts_styles 配置为空列表")
            return styles

        default_cfg = next((s for s in tts_styles_list if s.style_name == "default"), None)
        if not default_cfg:
            logger.error("在 tts_styles 配置中未找到 'default' 风格，这是必需的。")
            return styles

        default_refer_wav = default_cfg.refer_wav_path
        default_prompt_text = default_cfg.prompt_text
        default_gpt_weights = default_cfg.gpt_weights
        default_sovits_weights = default_cfg.sovits_weights

        if not default_refer_wav:
            logger.warning("TTS 'default' style is missing 'refer_wav_path'.")

        for style_cfg in tts_styles_list:
            style_name = style_cfg.style_name
            if not style_name:
                continue

            styles[style_name] = {
                "url": global_server,
                "name": style_cfg.name or style_name,
                "refer_wav_path": style_cfg.refer_wav_path or default_refer_wav,
                "prompt_text": style_cfg.prompt_text or default_prompt_text,
                "prompt_language": style_cfg.prompt_language or "zh",
                "gpt_weights": style_cfg.gpt_weights or default_gpt_weights,
                "sovits_weights": style_cfg.sovits_weights or default_sovits_weights,
                "speed_factor": style_cfg.speed_factor,
                "text_language": style_cfg.text_language or "auto",
            }
        return styles

    def get_available_styles(self) -> list[str]:
        """获取可用语音风格名称列表。

        Returns:
            可用风格名称列表
        """
        return list(self.tts_styles.keys())

    # ------------------------------------------------------------------
    # 语言检测与规范化
    # ------------------------------------------------------------------

    def _normalize_language_code(self, language_str: str) -> str:
        """从语言配置字符串中提取标准语言代码（去掉描述部分）。

        处理形如 "zh(中英混合)"、"en(English)" 这样的配置格式。

        Args:
            language_str: 原始语言配置字符串

        Returns:
            标准语言代码 (zh/en/ja/yue 等)
        """
        if not language_str:
            return "zh"

        # 提取括号前的部分
        base_code = language_str.split("(")[0].strip().lower()

        # 有效的语言代码白名单
        valid_codes = {"zh", "en", "ja", "yue", "auto", "auto_yue"}
        if base_code in valid_codes:
            return base_code

        # 如果提取后仍非法，默认中文
        logger.warning(f"无效的语言代码 '{language_str}'，已规范化为: zh")
        return "zh"

    def _analyze_text_language(self, text: str) -> tuple[str, str]:
        """分析文本内容，自动检测语言类型。

        返回主要语言和是否为混合语言的标识。

        Args:
            text: 要分析的文本

        Returns:
            (主要语言代码, 混合类型描述)
            例如: ("zh", "已检测中文")、("en", "已检测英文")、("zh", "中英混合")
        """
        # 字符统计
        zh_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        en_count = len(re.findall(r"[a-zA-Z]", text))
        ja_count = len(re.findall(r"[\u3040-\u309f\u30a0-\u30ff]", text))

        total_chars = zh_count + en_count + ja_count

        if total_chars == 0:
            return "zh", "缺省（未检测中英文）"

        # 计算比例
        zh_ratio = zh_count / total_chars
        en_ratio = en_count / total_chars
        ja_ratio = ja_count / total_chars

        # 粤语检测
        cantonese_keywords = ["嘅", "喺", "咗", "唔", "係", "啲", "咩", "乜", "喂"]
        has_cantonese = any(keyword in text for keyword in cantonese_keywords)

        # 返回主要语言和混合信息
        if ja_ratio > 0.3:
            return "ja", f"已检测日语(占比{ja_ratio*100:.0f}%)"
        elif has_cantonese:
            return "yue", "已检测粤语关键词"
        elif en_ratio > 0.3:
            if zh_ratio > 0.1:
                return "en", f"中英混合(中{zh_ratio*100:.0f}%,英{en_ratio*100:.0f}%)"
            else:
                return "en", f"已检测英文(占比{en_ratio*100:.0f}%)"
        else:
            if en_ratio > 0.05:
                return "zh", f"中英混合(中{zh_ratio*100:.0f}%,英{en_ratio*100:.0f}%)"
            else:
                return "zh", "已检测纯中文"

    def _determine_final_language(self, text: str, mode: str) -> str:
        """根据配置的语言策略和文本内容，决定最终发送给 API 的语言代码。

        使用规范化的语言代码，智能检测文本语言特征。

        参数说明:
        - mode: 语言配置模式
          * 标准语言代码 (zh/en/ja/yue): 直接使用
          * 带描述格式 (zh(中英混合)): 自动提取代码
          * auto: 根据文本自动检测
          * auto_yue: 自动检测，优先检查粤语

        Args:
            text: 要合成的文本
            mode: 语言模式配置

        Returns:
            最终语言代码字符串
        """
        # 第一步：规范化配置中的语言代码
        normalized_mode = self._normalize_language_code(mode)

        # 第二步：如果已是确定的语言代码（不是auto模式），直接返回
        if normalized_mode not in ["auto", "auto_yue"]:
            logger.info(f"使用配置的语言代码: {normalized_mode}")
            return normalized_mode

        # 第三步：自动分析文本语言
        detected_lang, detection_info = self._analyze_text_language(text)

        # 特殊处理 auto_yue 模式
        if normalized_mode == "auto_yue":
            logger.info(f"auto_yue 模式 - {detection_info}，最终语言: {detected_lang}")
            return detected_lang

        # 通用 auto 模式
        if detected_lang == "zh" and "中英混合" in detection_info:
            # 中英混合时优先用中文（大多数API支持更好）
            logger.info(f"auto 模式 - {detection_info}，以中文处理，最终语言: zh")
            return "zh"
        else:
            logger.info(f"auto 模式 - {detection_info}，最终语言: {detected_lang}")
            return detected_lang

    # ------------------------------------------------------------------
    # 文本清洗
    # ------------------------------------------------------------------

    def _clean_text_for_tts(self, text: str) -> str:
        """清洗文本以适合 TTS 合成。

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        # 1. 基本清理
        text = re.sub(r"[\(（\[【].*?[\)）\]】]", "", text)
        text = re.sub(r"([，。！？、；：,.!?;:~\-`])\1+", r"\1", text)
        text = re.sub(r"~{2,}|～{2,}", "，", text)
        text = re.sub(r"\.{3,}|…{1,}", "。", text)

        # 2. 词语替换
        replacements = {"www": "哈哈哈", "hhh": "哈哈", "233": "哈哈", "666": "厉害", "88": "拜拜"}
        for old, new in replacements.items():
            text = text.replace(old, new)

        # 3. 移除不必要的字符
        text = re.sub(
            r"[^\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffa-zA-Z0-9\s，。！？、；：,.!?;:~～]",
            "",
            text,
        )

        # 4. 确保结尾有标点
        if text and not text.endswith(tuple("，。！？、；：,.!?;:")):
            text += "。"

        # 5. 智能截断
        if len(text) > self.max_text_length:
            cut_text = text[: self.max_text_length]
            punctuation = "。！？.…"
            last_punc_pos = max(cut_text.rfind(p) for p in punctuation)

            if last_punc_pos != -1:
                text = cut_text[: last_punc_pos + 1]
            else:
                last_comma_pos = max(cut_text.rfind(p) for p in "，、；,;")
                if last_comma_pos != -1:
                    text = cut_text[: last_comma_pos + 1]
                else:
                    text = cut_text

        return text.strip()

    # ------------------------------------------------------------------
    # TTS API 调用
    # ------------------------------------------------------------------

    async def _call_tts_api(
        self,
        server_config: dict[str, Any],
        text: str,
        text_language: str,
        **kwargs: Any,
    ) -> bytes | None:
        """调用 GPT-SoVITS API 进行语音合成。

        先切换模型权重，再发送合成请求。

        Args:
            server_config: 风格服务配置
            text: 合成文本
            text_language: 文本语言
            **kwargs: 额外参数 (refer_wav_path, prompt_text, gpt_weights 等)

        Returns:
            音频字节数据，失败返回 None
        """
        ref_wav_path = kwargs.get("refer_wav_path")
        if not ref_wav_path:
            logger.error(f"API 调用失败：缺少 refer_wav_path。当前风格配置: {server_config}")
            return None

        try:
            base_url = server_config["url"].rstrip("/")

            # 步骤一：切换模型权重
            async def switch_model_weights(weights_path: str | None, weight_type: str) -> None:
                if not weights_path:
                    return
                api_endpoint = f"/set_{weight_type}_weights"
                switch_url = f"{base_url}{api_endpoint}"
                try:
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as session:
                        async with session.get(switch_url, params={"weights_path": weights_path}) as resp:
                            if resp.status != 200:
                                error_text = await resp.text()
                                logger.error(f"切换 {weight_type} 模型失败: {resp.status} - {error_text}")
                            else:
                                logger.info(f"成功切换 {weight_type} 模型为: {weights_path}")
                except Exception as e:
                    logger.error(f"请求切换 {weight_type} 模型时发生网络异常: {e}")

            await switch_model_weights(kwargs.get("gpt_weights"), "gpt")
            await switch_model_weights(kwargs.get("sovits_weights"), "sovits")

            # 步骤二：构建合成请求数据
            data: dict[str, Any] = {
                "text": text,
                "text_lang": text_language,
                "ref_audio_path": ref_wav_path,
                "prompt_text": kwargs.get("prompt_text", ""),
                "prompt_lang": kwargs.get("prompt_language", "zh"),
            }

            # 合并高级配置
            cfg = self._config
            advanced_dict = cfg.tts_advanced.model_dump()
            data.update({k: v for k, v in advanced_dict.items() if v is not None})

            # 优先使用风格特定的语速
            if server_config.get("speed_factor") is not None:
                data["speed_factor"] = server_config["speed_factor"]

            # 步骤三：发送合成请求
            tts_url = base_url if base_url.endswith("/tts") else f"{base_url}/tts"
            logger.info(f"发送到 TTS API 的数据: {data}")

            connector = aiohttp.TCPConnector(limit=100)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.post(tts_url, json=data) as response:
                    if response.status == 200:
                        audio_data = bytearray()
                        async for chunk in response.content.iter_chunked(1024 * 1024):
                            audio_data.extend(chunk)
                        logger.info(f"成功接收音频数据，大小: {len(audio_data)} 字节")
                        return bytes(audio_data)
                    else:
                        error_info = await response.text()
                        logger.error(f"TTS API调用失败: {response.status} - {error_info}")
                        return None

        except asyncio.TimeoutError:
            logger.error("TTS服务请求超时")
            return None
        except Exception as e:
            logger.error(f"TTS API调用异常: {e}")
            return None

    # ------------------------------------------------------------------
    # 空间音效处理
    # ------------------------------------------------------------------

    async def _apply_spatial_audio_effect(self, audio_data: bytes) -> bytes | None:
        """根据配置应用空间效果（混响和卷积）。

        Args:
            audio_data: 原始音频字节

        Returns:
            处理后的音频字节，失败返回原始音频
        """
        try:
            effects_cfg = self._config.spatial_effects
            if not effects_cfg.enabled:
                return audio_data

            # 基于 __file__ 构建 IR 文件路径
            plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ir_path = os.path.join(plugin_dir, "assets", "small_room_ir.wav")

            effects: list[Any] = []

            if effects_cfg.reverb_enabled:
                effects.append(
                    Reverb(
                        room_size=effects_cfg.room_size,
                        damping=effects_cfg.damping,
                        wet_level=effects_cfg.wet_level,
                        dry_level=effects_cfg.dry_level,
                        width=effects_cfg.width,
                    )
                )

            if effects_cfg.convolution_enabled and os.path.exists(ir_path):
                effects.append(
                    Convolution(
                        impulse_response_filename=ir_path,
                        mix=effects_cfg.convolution_mix,
                    )
                )
            elif effects_cfg.convolution_enabled:
                logger.warning(f"卷积混响已启用，但IR文件不存在 ({ir_path})，跳过该效果。")

            if not effects:
                return audio_data

            with io.BytesIO(audio_data) as audio_stream:
                with AudioFile(audio_stream, "r") as f:
                    board = Pedalboard(effects)
                    effected = board(f.read(f.frames), f.samplerate)

            with io.BytesIO() as output_stream:
                sf.write(output_stream, effected.T, f.samplerate, format="WAV")
                processed_audio_data = output_stream.getvalue()

            logger.info("成功应用空间效果。")
            return processed_audio_data

        except Exception as e:
            logger.error(f"应用空间效果时出错: {e}")
            return audio_data

    # ------------------------------------------------------------------
    # 语音生成（主入口）
    # ------------------------------------------------------------------

    async def generate_voice(
        self,
        text: str,
        style_hint: str = "default",
        language_hint: str | None = None,
    ) -> str | None:
        """生成语音并返回 Base64 编码。

        Args:
            text: 要合成的文本
            style_hint: 风格名称提示
            language_hint: 语言提示 (优先级最高)

        Returns:
            Base64 编码的音频数据，失败返回 None
        """
        self._load_config()

        if not self.tts_styles:
            logger.error("TTS风格配置为空，无法生成语音。")
            return None

        # 风格选择回退逻辑
        style = style_hint if style_hint in self.tts_styles else "default"
        if style not in self.tts_styles:
            if "default" in self.tts_styles:
                style = "default"
                logger.warning(f"指定风格 '{style_hint}' 不存在，自动回退到: 'default'")
            elif self.tts_styles:
                style = next(iter(self.tts_styles))
                logger.warning(
                    f"指定风格 '{style_hint}' 和 'default' 均不存在，自动回退到第一个可用风格: {style}"
                )
            else:
                logger.error("没有任何可用的TTS风格配置")
                return None

        server_config = self.tts_styles[style]
        clean_text = self._clean_text_for_tts(text)
        if not clean_text:
            return None

        # 语言决策：优先 language_hint → 风格配置策略 → 自动检测
        if language_hint:
            final_language = language_hint
            logger.info(f"使用决策模型指定的语言: {final_language}")
        else:
            language_policy = server_config.get("text_language", "auto")
            final_language = self._determine_final_language(clean_text, language_policy)
            logger.info(f"决策模型未指定语言，使用策略 '{language_policy}' -> 最终语言: {final_language}")

        logger.info(
            f"开始TTS语音合成，文本：{clean_text[:50]}..., 风格：{style}, 最终语言: {final_language}"
        )

        audio_data = await self._call_tts_api(
            server_config=server_config,
            text=clean_text,
            text_language=final_language,
            refer_wav_path=server_config.get("refer_wav_path"),
            prompt_text=server_config.get("prompt_text"),
            prompt_language=server_config.get("prompt_language"),
            gpt_weights=server_config.get("gpt_weights"),
            sovits_weights=server_config.get("sovits_weights"),
        )

        if audio_data:
            # 空间音效处理
            spatial_cfg = self._config.spatial_effects
            if spatial_cfg.enabled:
                logger.info("检测到已启用空间音频效果，开始处理...")
                processed_audio = await self._apply_spatial_audio_effect(audio_data)
                if processed_audio:
                    logger.info("空间音频效果应用成功！")
                    audio_data = processed_audio
                else:
                    logger.warning("空间音频效果应用失败，将使用原始音频。")

            return base64.b64encode(audio_data).decode("utf-8")
        return None
