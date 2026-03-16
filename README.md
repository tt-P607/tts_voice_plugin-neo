# TTS Voice Plugin (tts_voice_plugin)

基于 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 的文本转语音插件，为 Neo-MoFox 提供高质量、多语言、多风格的语音合成能力。

## 🌟 功能特性

- **多语言支持**：支持中文 (zh)、英文 (en)、日文 (ja)、粤语 (yue) 等，具备智能语言检测功能。
- **多风格切换**：支持配置多个语音风格（模型权重+参考音频），并可根据需求动态切换。
- **智能文本清洗**：自动处理文本中的特殊符号、表情符号缩写（如 `www`, `233`），并进行智能截断以适应 TTS 合成。
- **空间音效处理**：内置基于 `Pedalboard` 的音效处理器，支持标准混响 (Reverb) 和卷积混响 (Convolution)，营造更真实的听感。
- **灵活的触发方式**：
  - **Action 模式**：允许 AI 根据上下文主动决定是否发送语音。
  - **Command 模式**：支持通过指令手动触发语音合成。
- **模型动态切换**：支持在合成前动态切换 GPT 和 SoVITS 模型权重。

## 🛠️ 安装依赖

本插件需要以下 Python 库：

```bash
pip install aiohttp soundfile pedalboard
```

> **注意**：你需要一个运行中的 [GPT-SoVITS API 服务](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/cn/API.md)。

## ⚙️ 配置说明

配置文件位于 `config/plugins/tts_voice_plugin/config.toml`。

### 基础配置 `[plugin]`
- `enable`: 是否启用插件。
- `keywords`: 触发语音合成的关键词列表。

### TTS 服务配置 `[tts]`
- `server`: GPT-SoVITS API 服务地址（默认 `http://127.0.0.1:9880`）。
- `max_text_length`: 单次合成的最大文本长度。

### 语音风格配置 `[[tts_styles]]`
你可以配置多个风格，必须包含一个名为 `default` 的风格。
- `style_name`: 风格唯一标识。
- `name`: 显示名称。
- `refer_wav_path`: 参考音频的绝对路径。
- `prompt_text`: 参考音频对应的文本内容。
- `gpt_weights`: GPT 模型权重路径。
- `sovits_weights`: SoVITS 模型权重路径。
- `text_language`: 文本语言模式 (`zh`/`en`/`ja`/`yue`/`auto`/`auto_yue`)。

### 空间音效 `[spatial_effects]`
- `enabled`: 是否启用音效。
- `reverb_enabled`: 是否启用标准混响。
- `convolution_enabled`: 是否启用卷积混响（需在插件 `assets/` 目录下放置 `small_room_ir.wav`）。

## 🚀 使用方法

1. **主动语音**：在对话中，如果 AI 认为需要使用语音表达（或匹配到关键词），它会自动调用 `tts_voice_action`。
2. **手动指令**：使用配置的指令（如 `/tts`，具体取决于指令组件实现）来合成特定文本。

## 📄 开源协议

本项目采用 [AGPL-v3.0](LICENSE) 协议。
