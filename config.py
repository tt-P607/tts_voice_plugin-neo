"""TTS Voice 插件配置。

定义 GPT-SoVITS 语音合成插件的配置项，包括基础设置、风格列表、高级参数和音频效果器。
"""

from typing import ClassVar

from src.core.components.base.config import BaseConfig, Field, SectionBase, config_section


@config_section("plugin")
class PluginSection(SectionBase):
    """插件基本配置。"""

    enable: bool = Field(default=False, description="是否启用插件")
    llm_speed_control: bool = Field(
        default=False,
        description="是否允许 LLM 自主控制语速。关闭时忽略 LLM 传入的语速参数，始终使用风格配置中的语速值。",
    )
    llm_audio_effects: bool = Field(
        default=False,
        description="是否允许 LLM 自主控制音频效果器。关闭时 LLM 看不到效果器参数。",
    )
    keywords: list[str] = Field(
        default_factory=lambda: [
            "发语音", "语音", "说句话", "用语音说", "听你", "听声音",
            "想听你", "想听声音", "讲个话", "说段话", "念一下", "读一下",
            "用嘴说", "说", "能发语音吗", "亲口",
        ],
        description="触发语音合成的关键词列表",
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
        description=(
            "追加到 tts_voice_action action 描述末尾的自定义指令。\n"
            "可描述希望 AI 主动使用语音功能的具体场景，"
            "例如：在表达亲密感、讲故事或用户明确要求听声音时主动使用。\n"
            "不会覆盖已有的触发条件，只是扩充场景说明。"
        ),
    )


@config_section("tts")
class TTSSection(SectionBase):
    """TTS 语音合成基础配置。"""

    server: str = Field(default="http://127.0.0.1:9880", description="GPT-SoVITS 服务地址")
    timeout: int = Field(default=180, description="TTS 请求超时秒数")
    max_text_length: int = Field(default=1000, description="最大合成文本长度")
    wsl_mode: bool = Field(
        default=False,
        description=(
            "WSL 路径转换模式。Bot 运行在 Windows、napcat 运行在 WSL 时开启。\n"
            "启用后，file 模式发送文件时会将 Windows 绝对路径（如 C:\\path\\to\\file）\n"
            "自动转换为 WSL 挂载路径（如 /mnt/c/path/to/file），使 WSL 侧的 napcat 能正确读取。"
        ),
    )


@config_section("tts_styles")
class TTSStyle(SectionBase):
    """TTS 风格参数配置，每个实例代表一种独立的语音风格。"""

    style_name: str = Field(default="default", description="风格唯一标识符，必须有一个名为 default")
    name: str = Field(default="默认", description="显示名称")
    refer_wav_path: str = Field(default="C:/path/to/your/reference.wav", description="参考音频路径")
    prompt_text: str = Field(
        default="这是一个示例文本，请替换为您自己的参考音频文本。",
        description="参考音频文本",
    )
    prompt_language: str = Field(default="zh", description="参考音频语言")
    gpt_weights: str = Field(default="C:/path/to/your/gpt_weights.ckpt", description="GPT 模型路径")
    sovits_weights: str = Field(default="C:/path/to/your/sovits_weights.pth", description="SoVITS 模型路径")
    speed_factor: float = Field(default=1.0, description="语速因子")
    text_language: str = Field(default="auto", description="文本语言模式 (zh/ja/en/auto 等)")


@config_section("tts_streaming")
class TTSStreamingSection(SectionBase):
    """GSV 流式合成配置。启用后 anima_chatter 会边接收边播放，降低首字节延迟。"""

    enabled: bool = Field(default=False, description="是否启用 GSV 流式合成（仅对 anima_chatter 生效）")
    chunk_size: int = Field(default=4096, description="每次从 GSV 读取的字节块大小")
    min_play_bytes: int = Field(
        default=8192,
        description="积累到此字节数后才开始播放第一块，避免音频太短导致播放卡顿",
    )


@config_section("tts_advanced")
class TTSAdvancedSection(SectionBase):
    """TTS 高级参数配置（默认值与 GSV v2pro/v2proplus api_v2.py 的 TTS_Request 一一对应）。"""

    # 输出与采样参数
    media_type: str = Field(default="wav", description="输出音频格式（wav/ogg/aac/raw）")
    top_k: int = Field(default=15, description="Top-K 采样参数")
    top_p: float = Field(default=1.0, description="Top-P 核采样参数")
    temperature: float = Field(default=1.0, description="温度参数")

    # 批处理与分桶
    batch_size: int = Field(default=1, description="批处理大小（影响多段并行合成数量）")
    batch_threshold: float = Field(default=0.75, description="批处理阈值")
    split_bucket: bool = Field(default=True, description="是否分桶处理（True 让同 batch 内长度接近，避免 padding 损质量）")

    # 文本切分与拼接
    text_split_method: str = Field(default="cut5", description="文本分割方法（cut0/cut1/cut2/cut3/cut4/cut5）")
    fragment_interval: float = Field(default=0.3, description="多片段拼接处的静音间隔（秒）")
    overlap_length: int = Field(default=2, description="片段重叠长度（影响拼接平滑度）")
    min_chunk_length: int = Field(default=16, description="最小切片长度")

    # 推理控制
    parallel_infer: bool = Field(default=True, description="是否并行推理")
    seed: int = Field(default=-1, description="随机种子（-1=每次随机会导致音色波动；固定整数=可复现稳定音色）")

    # 质量微调
    repetition_penalty: float = Field(default=1.35, description="重复惩罚因子")
    sample_steps: int = Field(default=32, description="采样步数（v2pro 系列档位 8/16/32/64/128）")
    super_sampling: bool = Field(default=False, description="是否启用超采样（v2pro/v2proplus 专属高保真，仅这两版支持）")


@config_section("audio_effect_item")
class AudioEffectItem(SectionBase):
    """单个音频效果器配置。"""

    type: str = Field(
        default="",
        description=(
            "效果器类型（reverb/highpass/lowpass/pitch_shift/distortion/bitcrush/"
            "delay/chorus/gain/phaser/compressor/clipping/noise_gate/"
            "ladder_filter/resample/gsm/mp3）"
        ),
    )
    # reverb 参数
    room_size: float = Field(default=0.3, description="混响房间大小 (0.0-1.0)")
    wet_level: float = Field(default=0.3, description="混响湿声比例 (0.0-1.0)")
    damping: float = Field(default=0.6, description="混响阻尼 (0.0-1.0)")
    dry_level: float = Field(default=0.8, description="混响干声比例 (0.0-1.0)")
    width: float = Field(default=1.0, description="混响立体声宽度 (0.0-1.0)")
    # 滤波器参数（highpass / lowpass / ladder_filter 共用）
    cutoff_hz: float = Field(default=800.0, description="滤波器截止频率 (Hz)")
    resonance: float = Field(default=0.0, description="ladder_filter 共振 (0.0-1.0)")
    drive: float = Field(default=1.0, description="ladder_filter 驱动增益 (≥1.0)")
    # pitch_shift 参数
    semitones: float = Field(default=0.0, description="变调半音数 (-12到12)")
    # distortion 参数
    drive_db: float = Field(default=10.0, description="失真强度 (0-30dB)")
    # bitcrush 参数
    bit_depth: int = Field(default=8, description="位深 (4-16)")
    # delay 参数
    delay_seconds: float = Field(default=0.3, description="延迟时间 (0.05-1.0秒)")
    feedback: float = Field(default=0.3, description="回声衰减 (0.0-0.8)")
    mix: float = Field(default=0.5, description="效果混合比例 (0.0-1.0)")
    # chorus / phaser 参数
    rate_hz: float = Field(default=1.5, description="调制速率 (0.5-5.0Hz)")
    depth: float = Field(default=0.5, description="调制深度 (0.0-1.0)")
    # gain 参数
    gain_db: float = Field(default=0.0, description="增益 (-20到20dB)")
    # compressor / noise_gate / clipping 参数
    threshold_db: float = Field(default=-20.0, description="阈值 (dB)")
    ratio: float = Field(default=4.0, description="压缩比")
    attack_ms: float = Field(default=1.0, description="起音时间 (ms)")
    release_ms: float = Field(default=100.0, description="释放时间 (ms)")
    # resample 参数
    target_sample_rate: float = Field(default=8000.0, description="目标采样率 (Hz)")
    # mp3 参数
    vbr_quality: float = Field(default=5.0, description="MP3 VBR 质量 (0.0-9.0，越大质量越低)")


@config_section("audio_effects")
class AudioEffectsSection(SectionBase):
    """音频效果器配置。"""

    enabled: bool = Field(default=False, description="是否启用手动效果器链（对所有语音始终生效）")
    chain: list[AudioEffectItem] = Field(
        default_factory=list,
        description="手动效果器链，按顺序依次处理音频",
    )


class TTSVoiceConfig(BaseConfig):
    """TTS Voice 插件主配置类。"""

    config_name: ClassVar[str] = "config"
    config_description: ClassVar[str] = "GPT-SoVITS 语音合成插件配置"

    plugin: PluginSection = Field(default_factory=PluginSection)
    components: ComponentsSection = Field(default_factory=ComponentsSection)
    prompt: PromptSection = Field(default_factory=PromptSection)
    tts: TTSSection = Field(default_factory=TTSSection)
    tts_styles: list[TTSStyle] = Field(
        default_factory=lambda: [TTSStyle()],
        description="TTS 风格列表，每项为一种独立的语音风格配置",
    )
    tts_streaming: TTSStreamingSection = Field(default_factory=TTSStreamingSection)
    tts_advanced: TTSAdvancedSection = Field(default_factory=TTSAdvancedSection)
    audio_effects: AudioEffectsSection = Field(default_factory=AudioEffectsSection)
