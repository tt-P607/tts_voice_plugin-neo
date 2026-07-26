"""TTS Voice 插件配置。

定义 GPT-SoVITS 服务、语音风格、流式响应、高级推理参数和音频效果器约束。
"""

from __future__ import annotations

from typing import ClassVar, Literal

from src.app.plugin_system.base import BaseConfig, Field, SectionBase, config_section


LanguageCode = Literal[
    "zh",
    "en",
    "ja",
    "yue",
    "ko",
    "auto",
    "auto_yue",
    "all_zh",
    "all_ja",
    "all_yue",
    "all_ko",
    "zh_en",
]
MediaType = Literal["wav", "ogg", "aac", "raw"]
SplitMethod = Literal["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"]
EffectType = Literal[
    "reverb",
    "highpass",
    "lowpass",
    "pitch_shift",
    "distortion",
    "bitcrush",
    "delay",
    "chorus",
    "gain",
    "phaser",
    "compressor",
    "clipping",
    "noise_gate",
    "ladder_filter",
    "resample",
    "gsm",
    "mp3",
]


@config_section("plugin")
class PluginSection(SectionBase):
    """插件基本配置。"""

    enable: bool = Field(default=False, description="是否启用插件")
    llm_speed_control: bool = Field(
        default=False,
        description="是否允许 LLM 自主控制语速",
    )
    llm_audio_effects: bool = Field(
        default=False,
        description="是否允许 LLM 自主控制音频效果器",
    )
    keywords: list[str] = Field(
        default_factory=lambda: [
            "发语音",
            "语音",
            "说句话",
            "用语音说",
            "听你",
            "听声音",
            "想听你",
            "想听声音",
            "讲个话",
            "说段话",
            "念一下",
            "读一下",
            "用嘴说",
            "说",
            "能发语音吗",
            "亲口",
        ],
        description="触发语音合成的关键词列表",
        max_items=100,
    )


@config_section("components")
class ComponentsSection(SectionBase):
    """组件启用控制。"""

    action_enabled: bool = Field(default=True, description="是否启用 Action 组件")
    command_enabled: bool = Field(default=True, description="是否启用 Command 组件")


@config_section("prompt")
class PromptSection(SectionBase):
    """自定义提示词配置。"""

    custom_instructions: str = Field(
        default="",
        max_length=4000,
        description="追加到 TTS Action 描述末尾的自定义使用场景说明",
    )


@config_section("tts")
class TTSSection(SectionBase):
    """GPT-SoVITS 服务和本地文件配置。"""

    server: str = Field(
        default="http://127.0.0.1:9880",
        min_length=1,
        max_length=2048,
        pattern=r"^https?://.+",
        description="GPT-SoVITS 服务地址",
    )
    timeout: int = Field(default=180, ge=1, le=1800, description="TTS 请求超时秒数")
    max_text_length: int = Field(
        default=1000,
        ge=1,
        le=20000,
        description="单次合成最大文本长度",
    )
    upload_max_mb: int = Field(
        default=32,
        ge=1,
        le=512,
        description="WebUI 单个 WAV 上传文件大小上限（MB）",
    )
    wsl_mode: bool = Field(
        default=False,
        description="发送文件时把 Windows 绝对路径转换为 WSL 挂载路径",
    )


@config_section("tts_styles")
class TTSStyle(SectionBase):
    """单个 GPT-SoVITS 语音风格配置。"""

    style_name: str = Field(
        default="default",
        min_length=1,
        max_length=80,
        pattern=r"^[^\s/\\]+(?: [^/\\]+)*$",
        description="风格唯一名称",
    )
    refer_wav_path: str = Field(
        default="C:/path/to/your/reference.wav",
        min_length=1,
        max_length=4096,
        description="主参考音频路径",
    )
    prompt_text: str = Field(
        default="这是一个示例文本，请替换为您自己的参考音频文本。",
        max_length=5000,
        description="参考音频文本",
    )
    prompt_language: LanguageCode = Field(default="zh", description="参考音频语言")
    gpt_weights: str = Field(
        default="C:/path/to/your/gpt_weights.ckpt",
        max_length=4096,
        description="GPT 模型路径",
    )
    sovits_weights: str = Field(
        default="C:/path/to/your/sovits_weights.pth",
        max_length=4096,
        description="SoVITS 模型路径",
    )
    speed_factor: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="该风格默认语速因子",
    )
    text_language: LanguageCode = Field(default="auto", description="待合成文本语言模式")
    aux_refer_wav_paths: list[str] = Field(
        default_factory=list,
        max_items=16,
        description="辅助参考音频路径列表",
    )


@config_section("tts_streaming")
class TTSStreamingSection(SectionBase):
    """GPT-SoVITS 流式响应配置。"""

    enabled: bool = Field(
        default=False,
        description="是否允许 Provider 使用流式合成接口；当前无内置消费方",
    )
    streaming_mode: int = Field(
        default=2,
        ge=0,
        le=2,
        description="传递给 GPT-SoVITS 的 streaming_mode",
    )
    chunk_size: int = Field(
        default=4096,
        ge=512,
        le=1024 * 1024,
        description="流式响应每次读取的字节块大小",
    )


@config_section("tts_advanced")
class TTSAdvancedSection(SectionBase):
    """GPT-SoVITS 高级推理参数。"""

    media_type: MediaType = Field(default="wav", description="输出音频格式")
    top_k: int = Field(default=15, ge=1, le=100, description="Top-K 采样参数")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, description="Top-P 核采样参数")
    temperature: float = Field(default=1.0, gt=0.0, le=2.0, description="温度参数")
    batch_size: int = Field(default=1, ge=1, le=64, description="批处理大小")
    batch_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="批处理阈值",
    )
    split_bucket: bool = Field(default=True, description="是否按长度分桶")
    text_split_method: SplitMethod = Field(default="cut5", description="文本分割方法")
    fragment_interval: float = Field(
        default=0.3,
        ge=0.0,
        le=10.0,
        description="多片段拼接静音间隔秒数",
    )
    overlap_length: int = Field(default=2, ge=0, le=64, description="片段重叠长度")
    min_chunk_length: int = Field(default=16, ge=1, le=4096, description="最小切片长度")
    parallel_infer: bool = Field(default=True, description="是否并行推理")
    seed: int = Field(default=-1, ge=-1, description="随机种子，-1 表示随机")
    repetition_penalty: float = Field(
        default=1.35,
        gt=0.0,
        le=10.0,
        description="重复惩罚因子",
    )
    sample_steps: Literal[8, 16, 32, 64, 128] = Field(
        default=32,
        description="v2pro 系列采样步数",
    )
    super_sampling: bool = Field(default=False, description="是否启用超采样")


@config_section("audio_effect_item")
class AudioEffectItem(SectionBase):
    """单个音频效果器配置。

    ``params`` 只需填写该 ``type`` 真正使用的参数，未填写的参数使用效果器默认值。
    合法参数名与取值范围由 :mod:`services.effects` 的效果器规格表定义，
    非法参数名或超范围取值会在构建效果器链时被拒绝。
    """

    type: EffectType = Field(default="reverb", description="效果器类型")
    params: dict[str, float] = Field(
        default_factory=dict,
        description="该效果器的参数键值对，如 { room_size = 0.5, wet_level = 0.3 }",
    )


@config_section("audio_effects")
class AudioEffectsSection(SectionBase):
    """手动音频效果器链配置。"""

    enabled: bool = Field(default=False, description="是否始终应用手动效果器链")
    chain: list[AudioEffectItem] = Field(
        default_factory=list,
        max_items=16,
        description="按顺序执行的效果器链",
    )


class TTSVoiceConfig(BaseConfig):
    """TTS Voice 插件主配置。"""

    name: ClassVar[str] = "config"
    description: ClassVar[str] = "GPT-SoVITS 语音合成插件配置"

    plugin: PluginSection = Field(default_factory=PluginSection)
    components: ComponentsSection = Field(default_factory=ComponentsSection)
    prompt: PromptSection = Field(default_factory=PromptSection)
    tts: TTSSection = Field(default_factory=TTSSection)
    tts_styles: list[TTSStyle] = Field(
        default_factory=lambda: [TTSStyle()],
        min_items=1,
        max_items=64,
        description="TTS 风格列表",
    )
    tts_streaming: TTSStreamingSection = Field(default_factory=TTSStreamingSection)
    tts_advanced: TTSAdvancedSection = Field(default_factory=TTSAdvancedSection)
    audio_effects: AudioEffectsSection = Field(default_factory=AudioEffectsSection)


__all__ = [
    "AudioEffectItem",
    "AudioEffectsSection",
    "ComponentsSection",
    "PluginSection",
    "PromptSection",
    "TTSAdvancedSection",
    "TTSSection",
    "TTSStreamingSection",
    "TTSStyle",
    "TTSVoiceConfig",
]
