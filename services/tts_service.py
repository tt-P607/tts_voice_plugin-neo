"""TTS 核心服务。

封装 GPT-SoVITS 语音合成的核心逻辑，包括风格管理、文本清洗、API 调用和空间音效处理。
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import wave
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import aiohttp
import soundfile as sf
from pedalboard import Convolution, Pedalboard, Reverb  # type: ignore[attr-defined]
from pedalboard.io import AudioFile

from src.app.plugin_system.api.log_api import get_logger
from src.core.components.base.service import BaseService

from ..language import normalize_language_code

if TYPE_CHECKING:
    from src.core.components.base.plugin import BasePlugin

    from ..config import TTSVoiceConfig

logger = get_logger("tts_voice_plugin-neo.service")


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
        # 缓存当前已加载的模型权重路径，避免相同路径重复切换
        self._loaded_gpt_weights: str | None = None
        self._loaded_sovits_weights: str | None = None
        self._load_config()

    # ------------------------------------------------------------------
    # 配置加载
    # ------------------------------------------------------------------

    @property
    def _config(self) -> "TTSVoiceConfig":
        """获取当前插件配置的快捷属性。"""
        return self.plugin.config  # type: ignore[return-value]

    def _load_config(self) -> None:
        """从插件配置加载 TTS 参数。

        本方法在 ``__init__`` 与 ``_generate_raw_audio`` 头部都会被调用——前者
        是一次性初始化，后者是每次合成的"配置热更"。两条路径对加载日志的
        要求不同：

        - 初始化（首次加载）：用 INFO 让用户看到风格列表，确认插件就绪。
        - 每次合成：风格列表通常没变化，再打 INFO 会刷屏 → 改为 DEBUG。

        通过 ``self.tts_styles`` 是否已存在来区分两种路径：``__init__`` 里
        ``self.tts_styles == {}``（首次加载），其它路径已经加载过。
        """
        is_first_load = not self.tts_styles
        try:
            cfg = self._config
            self.timeout = cfg.tts.timeout
            self.max_text_length = cfg.tts.max_text_length
            self.tts_styles = self._load_tts_styles()

            if self.tts_styles:
                if is_first_load:
                    logger.info(f"TTS服务已成功加载风格: {list(self.tts_styles.keys())}")
                else:
                    logger.debug(f"TTS服务热更加载风格: {list(self.tts_styles.keys())}")
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

        在 GSV 合法代码集合上使用 :func:`normalize_language_code` 做最佳匹配，
        覆盖以下场景：

        - 配置写法 ``"zh(中英混合)"``、``"en(English)"`` → 提取括号前部分
        - LLM 幻觉写成 ``"chinese"`` / ``"jp"`` / ``"zh-CN"`` → 别名表映射
        - 拼写错误 ``"yuee"`` / ``"japanease"`` → 模糊匹配兜底
        - 完全无法识别 → 回退到 ``"zh"``

        Args:
            language_str: 原始语言配置字符串

        Returns:
            标准语言代码 (zh/en/ja/yue 等)
        """
        code, kind = normalize_language_code(language_str, default="zh")
        if kind == "fallback":
            logger.warning(f"无效的语言代码 '{language_str}'，无法匹配任何合法代码，已回退为: zh")
        elif kind in ("alias", "fuzzy", "normalized") and language_str:
            logger.info(
                f"语言代码 '{language_str}' 通过 {kind} 匹配规整为: {code}"
            )
        return code

    def _determine_final_language(self, text: str, mode: str) -> str:
        """根据配置决定发送给 API 的语言代码，直接使用配置值无需自动检测。

        Args:
            text: 要合成的文本（保留参数以兼容调用方）
            mode: 语言配置模式

        Returns:
            最终语言代码字符串
        """
        return self._normalize_language_code(mode)

    # ------------------------------------------------------------------
    # 文本清洗
    # ------------------------------------------------------------------

    def _clean_text_for_tts(self, text: str) -> str:
        """清洗文本以适合 TTS 合成。

        移除括号内容，按最大长度截断。

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        # 移除括号/方括号内的内容
        text = re.sub(r"[\(（\[【].*?[\)）\]】]", "", text)
        # 按最大允许长度截断
        if len(text) > self.max_text_length:
            text = text[: self.max_text_length]
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

            # 步骤一：切换模型权重（相同路径跳过，避免重复加载）
            async def switch_model_weights(weights_path: str | None, weight_type: str) -> None:
                if not weights_path:
                    return
                cache_attr = f"_loaded_{weight_type}_weights"
                if getattr(self, cache_attr, None) == weights_path:
                    logger.debug(f"{weight_type} 模型权重未变化，跳过切换: {weights_path}")
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
                                setattr(self, cache_attr, weights_path)
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
            # 完整请求体只走 DEBUG——参数列表 30+ 字段，每次合成都打到 INFO
            # 会让终端被推理日志淹没。INFO 级别的"开始合成"信息已经在
            # _generate_raw_audio 里打过了（含风格 / 语言 / 文本预览），这里
            # 只在排查 GSV 异常时才需要看完整 payload。
            logger.debug(f"发送到 TTS API 的数据: {data}")

            connector = aiohttp.TCPConnector(limit=100)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.post(tts_url, json=data) as response:
                    if response.status == 200:
                        audio_data = bytearray()
                        async for chunk in response.content.iter_chunked(1024 * 1024):
                            audio_data.extend(chunk)
                        # 字节大小对用户没有诊断价值，刷屏；改为 DEBUG。
                        logger.debug(f"成功接收音频数据，大小: {len(audio_data)} 字节")
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

    async def _generate_raw_audio(
        self,
        text: str,
        style_hint: str = "default",
        language_hint: str | None = None,
    ) -> bytes | None:
        """生成语音并返回原始音频字节数据。

        Args:
            text: 要合成的文本
            style_hint: 风格名称提示
            language_hint: 语言提示 (优先级最高)

        Returns:
            音频字节数据，失败返回 None
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
        # 决策路径在最终的"开始合成"日志里已经体现（最终语言会打出来），
        # 这里两条预备日志都改为 DEBUG，避免每次合成都刷三行。
        # 注意：language_hint 可能是 LLM 幻觉出的非法代码（如 "chinese" / "jp" /
        # "zh-CN"），这里强制走一遍归一化保证发给 GSV 的一定是合法代码。
        if language_hint:
            final_language = self._normalize_language_code(language_hint)
            logger.debug(f"使用决策模型指定的语言: {language_hint!r} -> {final_language}")
        else:
            language_policy = server_config.get("text_language", "auto")
            final_language = self._determine_final_language(clean_text, language_policy)
            logger.debug(f"决策模型未指定语言，使用策略 '{language_policy}' -> 最终语言: {final_language}")

        # 这条 INFO 是 TTS 合成的"用户可见入口"——风格 + 语言 + 文本预览
        # 三条信息一行交代，足够诊断绝大多数日常问题。
        logger.info(
            f"开始TTS语音合成，风格：{style}，语言：{final_language}，文本：{clean_text[:50]}..."
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

        return audio_data

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
        audio_data = await self._generate_raw_audio(text, style_hint, language_hint)
        if audio_data:
            return base64.b64encode(audio_data).decode("utf-8")
        return None

    async def generate_voice_bytes(
        self,
        text: str,
        style_hint: str = "default",
        language_hint: str | None = None,
    ) -> bytes | None:
        """生成语音并返回原始字节数据（供多段合并模式使用）。

        Args:
            text: 要合成的文本
            style_hint: 风格名称提示
            language_hint: 语言提示 (优先级最高)

        Returns:
            音频字节数据，失败返回 None
        """
        return await self._generate_raw_audio(text, style_hint, language_hint)

    async def generate_voice_stream(
        self,
        text: str,
        style_hint: str = "default",
        language_hint: str | None = None,
        chunk_size: int = 4096,
    ) -> AsyncGenerator[bytes, None]:
        """流式生成语音，边接收 GSV 响应边 yield 音频块。

        GSV /tts 接口设置 streaming_mode=True 后返回 chunked transfer 响应，
        第一个 chunk 是 WAV header，后续是 raw PCM 数据块。
        此方法使用 httpx 的 stream 模式接收，httpx 对 chunked EOF 的处理比 aiohttp 更可靠。

        Args:
            text: 要合成的文本
            style_hint: 风格名称提示
            language_hint: 语言提示 (优先级最高)
            chunk_size: 每次 yield 的字节块大小

        Yields:
            音频字节块（首块为 WAV header，后续为 raw PCM）
        """
        self._load_config()

        if not self.tts_styles:
            logger.error("TTS风格配置为空，无法生成语音。")
            return

        style = style_hint if style_hint in self.tts_styles else "default"
        if style not in self.tts_styles:
            if self.tts_styles:
                style = next(iter(self.tts_styles))
                logger.warning(f"指定风格 '{style_hint}' 不存在，回退到: {style}")
            else:
                logger.error("没有任何可用的TTS风格配置")
                return

        server_config = self.tts_styles[style]
        clean_text = self._clean_text_for_tts(text)
        if not clean_text:
            return

        # language_hint 可能是 LLM 幻觉出的非法代码，强制归一化以避免合成失败。
        if language_hint:
            final_language = self._normalize_language_code(language_hint)
        else:
            language_policy = server_config.get("text_language", "auto")
            final_language = self._determine_final_language(clean_text, language_policy)

        base_url = server_config["url"].rstrip("/")

        # 切换模型权重（带缓存，相同路径跳过）
        async def switch_model_weights_stream(weights_path: str | None, weight_type: str) -> None:
            if not weights_path:
                return
            cache_attr = f"_loaded_{weight_type}_weights"
            if getattr(self, cache_attr, None) == weights_path:
                logger.debug(f"[stream] {weight_type} 模型权重未变化，跳过切换: {weights_path}")
                return
            switch_url = f"{base_url}/set_{weight_type}_weights"
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as session:
                    async with session.get(switch_url, params={"weights_path": weights_path}) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            logger.error(f"[stream] 切换 {weight_type} 模型失败: {resp.status} - {error_text}")
                        else:
                            setattr(self, cache_attr, weights_path)
                            logger.info(f"[stream] 成功切换 {weight_type} 模型为: {weights_path}")
            except Exception as e:
                logger.error(f"[stream] 请求切换 {weight_type} 模型时发生网络异常: {e}")

        await switch_model_weights_stream(server_config.get("gpt_weights"), "gpt")
        await switch_model_weights_stream(server_config.get("sovits_weights"), "sovits")

        # 构建合成请求（streaming_mode 从配置读取，默认等级 2）
        cfg = self._config
        advanced_dict = cfg.tts_advanced.model_dump()
        streaming_mode = int(getattr(cfg.tts_streaming, "streaming_mode", 2))
        
        data: dict[str, Any] = {
            "text": clean_text,
            "text_lang": final_language,
            "ref_audio_path": server_config.get("refer_wav_path", ""),
            "prompt_text": server_config.get("prompt_text", ""),
            "prompt_lang": server_config.get("prompt_language", "zh"),
            "streaming_mode": streaming_mode,
        }
        data.update({k: v for k, v in advanced_dict.items() if v is not None})
        # streaming_mode 不能被 advanced_dict 覆盖，强制使用配置值
        data["streaming_mode"] = streaming_mode
        if server_config.get("speed_factor") is not None:
            data["speed_factor"] = server_config["speed_factor"]

        tts_url = base_url if base_url.endswith("/tts") else f"{base_url}/tts"
        logger.info(f"[stream] 开始流式 TTS 合成: {clean_text[:50]}...")

        try:
            import httpx
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=10.0)) as client:
                async with client.stream("POST", tts_url, json=data) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        logger.error(f"[stream] TTS API 调用失败: {response.status_code} - {error_body.decode(errors='replace')}")
                        return
                    async for chunk in response.aiter_bytes(chunk_size):
                        if chunk:
                            yield chunk
            logger.info("[stream] 流式 TTS 合成完成")
        except asyncio.TimeoutError:
            logger.error("[stream] TTS 流式请求超时")
        except Exception as e:
            logger.error(f"[stream] TTS 流式 API 调用异常: {e}")

    def merge_audio_bytes(self, audio_list: list[bytes]) -> bytes | None:
        """将多段 WAV 音频字节合并为单段。

        Args:
            audio_list: 多段 WAV 音频字节列表

        Returns:
            合并后的 WAV 字节数据，失败返回 None
        """
        if not audio_list:
            return None
        if len(audio_list) == 1:
            return audio_list[0]

        try:
            output = io.BytesIO()
            with wave.open(output, "wb") as out_wav:
                params_set = False
                for audio_bytes in audio_list:
                    with wave.open(io.BytesIO(audio_bytes)) as in_wav:  # type: ignore[arg-type]
                        if not params_set:
                            out_wav.setparams(in_wav.getparams())
                            params_set = True
                        out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
            return output.getvalue()
        except Exception as e:
            logger.error(f"音频合并失败: {e}")
            return None
