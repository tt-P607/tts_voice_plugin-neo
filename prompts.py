"""TTS Voice 插件面向 LLM 的参数说明文本。

Action 的 ``Annotated`` 描述与 Provider 的 ``ParameterGuide`` 共用这里的常量，
避免两处文案漂移，也避免跨模块导入私有名称。
"""

from __future__ import annotations


ACTION_DESCRIPTION: str = (
    "将文本转换为语音并发送。\n"
    "【发送模式】默认使用 voice 模式发送语音条，超出单条时长限制时分多条发送。"
    "file 模式仅在内容超长或用户明确要求时使用。\n"
    "注意：这是纯语音合成，只能说话，不能唱歌！"
)

TTS_SEGMENTS_HINT: str = (
    "需要合成的语音段落列表。每个段落是一个对象，包含以下字段：\n"
    "  text (必填): 该段要合成的文本\n"
    "  voice_style (可选): 该段使用的语音风格，覆盖全局 voice_style\n"
    "  text_language (可选): 该段的语言，覆盖全局 text_language\n"
    "  speed_factor (可选): 该段的语速因子，覆盖全局 speed_factor\n"
    "【使用原则】：\n"
    '- 日常对话直接传单段：[{"text": "内容"}]，无需指定段级参数\n'
    "- 仅在同一段话内需要切换风格、语言或语速时才分多段，"
    "并配合 merge_voice=true 合并为一条语音\n"
    "- 分段应落在句子结束或自然停顿点，避免生硬截断\n"
    "【情感表达】：\n"
    "- 善用标点符号（！、？、……）引导合成声音的起伏与停顿\n"
    "- 使用语气助词（如：诶、呀、嘛、唔）增强口语的真实感\n"
    "- 移除无法语音化的内容（如括号内的动作说明）"
)

SEND_MODE_HINT: str = (
    "发送模式（默认 voice）：\n"
    "  voice — 发送语音条消息。日常对话默认使用此模式。\n"
    "  file — 合并为音频文件发送。适用于内容极长或需要正式传输的场景。"
)

VOICE_STYLE_HINT: str = (
    "全局默认语音风格。请根据对话内容的实际情感选择相应风格，"
    "具体可用风格请参考下方的【当前可用语音风格】列表。如未提供则使用默认风格。\n"
    "当段落对象中指定了 voice_style 时，该段会使用段级风格而非此全局值。"
)

SPEED_FACTOR_HINT: str = (
    "全局默认语速因子（可选）。控制语音播放速度，不填则使用默认语速。\n"
    "取值范围限制在 0.85 ~ 1.2 之间，根据对话情绪氛围自行判断，不确定时不要填写。\n"
    "当段落对象中指定了 speed_factor 时，该段会使用段级语速而非此全局值。"
)

FILE_NAME_HINT: str = (
    "file 模式下发送的文件名（可选，仅 send_mode=file 时生效）。\n"
    "不填时默认使用时间戳命名（如 20260504_151230.wav）。\n"
    "填写时只需填文件名，不需要带扩展名，如 '晚安故事'。"
)

MERGE_VOICE_HINT: str = (
    "控制是否将多段文本合成后的音频合并为一条语音条发送（默认 false）。\n"
    "- 设为 true：各段音频合并为一条语音条。当为了在一句话内实现情绪转换、"
    "多语种混合或语速变化而拆分文本时，"
    "开启此项可将各段拼接为一条连续语音，表现为一次完整的表达。\n"
    "- 设为 false：各段落将作为独立的语音条分多条发送。"
    "仅在确实需要多次发送语音条时使用。"
)

PAUSE_DURATION_HINT: str = (
    "拼接模式下片段之间的自然静音时长（秒，默认 0.3）。\n"
    "仅在 merge_voice=true 时生效。应根据语境逻辑灵活调节：\n"
    "- 情绪剧烈波动或转折点，可适当延长以留出心理缓冲感。\n"
    "- 语言切换或紧凑衔接的对话，应缩短以保持流畅度。\n"
    "- 模拟真人讲话时的换气或思索节奏进行细微调整。"
)

AUX_REFER_WAV_PATHS_HINT: str = "辅助参考音频本地绝对路径列表"

AUDIO_EFFECTS_HINT: str = (
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
    '  电话: [{"type":"highpass","cutoff_hz":800},{"type":"lowpass","cutoff_hz":3000}]\n'
    '  2G手机: [{"type":"gsm"}]\n'
    '  远处喊话: [{"type":"lowpass","cutoff_hz":2000},{"type":"reverb","room_size":0.7,"wet_level":0.5}]\n'
    '  广播: [{"type":"highpass","cutoff_hz":500},{"type":"lowpass","cutoff_hz":4000},'
    '{"type":"distortion","drive_db":5}]\n'
    '  耳边悄悄话: [{"type":"lowpass","cutoff_hz":6000},{"type":"reverb","room_size":0.1,'
    '"wet_level":0.15},{"type":"gain","gain_db":-3}]\n'
    '  过山车颤音: [{"type":"phaser","rate_hz":3.0,"depth":0.8,"feedback":0.6,"mix":0.7},'
    '{"type":"chorus","rate_hz":4.0,"depth":0.8,"mix":0.5}]\n'
    '  磁带卡带: [{"type":"bitcrush","bit_depth":10},{"type":"resample","target_sample_rate":11025},'
    '{"type":"chorus","rate_hz":0.5,"depth":0.3,"mix":0.2}]\n'
    '  水下: [{"type":"lowpass","cutoff_hz":800},{"type":"chorus","rate_hz":0.8,"depth":0.6,"mix":0.4}]\n'
    '  机器人: [{"type":"bitcrush","bit_depth":6},{"type":"distortion","drive_db":8},'
    '{"type":"resample","target_sample_rate":8000}]\n'
    '  大教堂: [{"type":"reverb","room_size":0.95,"wet_level":0.7,"damping":0.3}]\n'
    '  对讲机: [{"type":"highpass","cutoff_hz":1500},{"type":"lowpass","cutoff_hz":3500},'
    '{"type":"clipping","threshold_db":-8},{"type":"noise_gate","threshold_db":-35}]'
)


def build_action_description(styles: list[str], custom_instructions: str) -> str:
    """按当前可用风格与自定义说明构建 Action 描述。

    Args:
        styles: 当前可用的语音风格名称列表。
        custom_instructions: 配置中追加的自定义使用场景说明。

    Returns:
        完整的 Action 描述文本。
    """
    description = ACTION_DESCRIPTION
    if styles:
        style_lines = "\n".join(f"  - '{style}'" for style in styles)
        description += (
            "\n\n【voice_style 参数可选风格】（必须从以下列表中选择）：\n" f"{style_lines}"
        )
    custom = custom_instructions.strip()
    if custom:
        description += f"\n\n自定义指令：\n{custom}"
    return description


__all__ = [
    "ACTION_DESCRIPTION",
    "AUDIO_EFFECTS_HINT",
    "AUX_REFER_WAV_PATHS_HINT",
    "FILE_NAME_HINT",
    "MERGE_VOICE_HINT",
    "PAUSE_DURATION_HINT",
    "SEND_MODE_HINT",
    "SPEED_FACTOR_HINT",
    "TTS_SEGMENTS_HINT",
    "VOICE_STYLE_HINT",
    "build_action_description",
]
