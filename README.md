# TTS Voice Plugin Neo

`tts_voice_plugin-neo` 为 Neo-MoFox 提供 GPT-SoVITS 文本转语音能力，包含聊天 Action、手动命令、可复用 Service、TTS Provider 和调试 WebUI。

## 组件

| 签名 | 职责 |
|---|---|
| `tts_voice_plugin-neo:service:tts` | 风格解析、语言归一化、合成编排、效果器与音频拼接 |
| `tts_voice_plugin-neo:action:tts_voice_action` | 由模型合成并向当前聊天流发送语音条或 WAV 文件 |
| `tts_voice_plugin-neo:command:tts` | 提供 `/tts` 与 `/tts file` 手动入口 |
| `tts_voice_plugin-neo:router:tts_voice_webui` | 提供调试合成、WAV 上传和配置编辑页面 |
| `tts_voice_plugin-neo:config:config` | 声明插件、风格、流式响应、高级推理和效果器配置 |

插件可独立加载；若存在 `tts_http_server:service:tts_provider_registry`，还会注册名为 `tts_voice_plugin-neo` 的 Provider 接入 HTTP 协议链路。插件卸载时会注销 Provider、取消命令后台任务并关闭 HTTP 会话。

## 模块结构

```text
plugin.py               插件装配与生命周期
prompts.py              面向 LLM 的参数说明文本（Action 与 Provider 共用）
language.py             语言代码归一化与命令参数判定
protocol.py             跨组件结构协议（Registry / 插件实例 / 合成响应）
config.py               配置模型
config_persistence.py   配置 TOML 写回（唯一的框架内部调用边界）
provider.py             TTS Registry Provider 适配
router.py               调试 WebUI 与 HTTP API
actions/tts_action.py   语音合成 Action
commands/tts_command.py /tts 命令
services/
  tts_service.py        对外门面，负责参数解析与流程编排
  styles.py             配置风格 → 强类型 StyleProfile
  gsv_client.py         GPT-SoVITS HTTP 通信与权重切换
  effects.py            pedalboard 效果器链（规格表驱动）
  audio.py              WAV 拼接、时长估算、临时文件
```

## 依赖

- Neo-MoFox Python `>=3.11`
- 运行中的 GPT-SoVITS API v2 服务
- Python 包：`aiohttp`、`numpy`、`soundfile`、`pedalboard`

可选集成：安装 `tts_http_server` 后，本插件会向共享 Registry 注册 Provider，使语音能力经 `mfx-tts-http-v1` 协议对外提供；未安装时插件照常独立运行。

项目统一使用 `uv`：

```bash
uv sync
```

## 配置

配置文件位置：

```text
config/plugins/tts_voice_plugin-neo/config.toml
```

### `[plugin]`

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `enable` | `false` | 总开关；关闭时不注册组件和 Provider |
| `llm_speed_control` | `false` | 是否向模型暴露 `speed_factor` 参数 |
| `llm_audio_effects` | `false` | 是否向模型暴露 `audio_effects` 参数 |
| `keywords` | 内置列表 | Action 关键词激活列表 |

两个 LLM 开关同时打开时，会额外向模型暴露 `aux_refer_wav_paths`、`merge_voice` 和 `pause_duration`。

### `[components]`

- `action_enabled`：是否注册 TTS Action。
- `command_enabled`：是否注册 `/tts` 命令。

### `[tts]`

| 字段 | 默认值 | 范围/说明 |
|---|---:|---|
| `server` | `http://127.0.0.1:9880` | GPT-SoVITS HTTP 地址，仅接受 HTTP/HTTPS URL |
| `timeout` | `180` | 1~1800 秒 |
| `max_text_length` | `1000` | 1~20000 字符，超出部分会截断 |
| `upload_max_mb` | `32` | WebUI 单个 WAV 上传上限，1~512 MB |
| `wsl_mode` | `false` | 文件发送时将 Windows 路径转换为 `/mnt/<drive>/...` |

### `[[tts_styles]]`

至少配置一个风格，风格名不能重复。

- `style_name`：风格唯一名称。
- `refer_wav_path`：主参考 WAV 路径（必填）。
- `prompt_text`：参考音频对应文本。
- `prompt_language`：参考音频语言代码。
- `gpt_weights` / `sovits_weights`：模型权重路径。
- `speed_factor`：风格默认语速，范围 0.5~2.0。
- `text_language`：待合成文本语言模式。
- `aux_refer_wav_paths`：最多 16 个辅助参考音频路径。

第一个风格作为缺省基准：其余风格留空的 `prompt_text`、`gpt_weights`、`sovits_weights` 会回退到它的对应值。

合法语言代码：`zh`、`en`、`ja`、`yue`、`ko`、`auto`、`auto_yue`、`all_zh`、`all_ja`、`all_yue`、`all_ko`、`zh_en`。

### `[tts_streaming]`

- `enabled`：是否允许调用 Provider 的实验性流式接口。
- `streaming_mode`：传递给 GPT-SoVITS 的流式等级，范围 0~2。
- `chunk_size`：每次读取字节数。

当前 Neo-MoFox 内置插件没有消费 `synthesize_stream()`；启用该配置只开放 Provider 接口，不代表现有聊天链自动边合成边播放。流式路径不支持效果器后处理——效果器需要完整音频才能渲染。

### `[tts_advanced]`

对应 GPT-SoVITS API v2 的采样、批处理、切分和推理参数。主要字段均有范围或枚举约束；无效配置会在加载或 WebUI 保存阶段被拒绝。

### `[audio_effects]`

- `enabled`：是否对所有合成结果应用手动效果器链。
- `chain`：效果器列表，最多 16 项，按顺序执行。

每一项由 `type` 与 `params` 组成，`params` 只需填写该类型真正使用的参数：

```toml
[[audio_effects.chain]]
type = "highpass"
params = { cutoff_hz = 800.0 }

[[audio_effects.chain]]
type = "reverb"
params = { room_size = 0.5, wet_level = 0.4 }
```

支持 `reverb`、`highpass`、`lowpass`、`pitch_shift`、`distortion`、`bitcrush`、`delay`、`chorus`、`gain`、`phaser`、`compressor`、`clipping`、`noise_gate`、`ladder_filter`、`resample`、`gsm`、`mp3`。

未知的效果器类型、不属于该类型的参数名以及超范围取值都会被拒绝并记录错误，不会静默忽略。各类型的合法参数与取值范围由 [`services/effects.py`](services/effects.py) 的 `EFFECT_SPECS` 定义。

## 使用

### 聊天命令

```text
/tts <文本> [风格] [语言]
/tts file <文本> [风格] [语言]
```

命令需要 `OPERATOR` 权限。命令会立即返回任务已提交，并由框架 TaskManager 在后台完成合成和发送。

尾部的风格名与语言代码会被自动识别并剥离。语言判定是严格匹配（支持 `ja`、`纯中文` 这类别名，但不做模糊匹配），因此正文最后一个词不会被误吞。

### Action

Action 默认使用 `voice` 模式发送语音条；`file` 模式会合并音频并发送临时 WAV。临时文件始终写入 `data/tts_voice_plugin-neo/`，发送后删除。自定义文件名会安全归一化，不能逃逸数据目录。

Action 只有一个类，向模型暴露的可选参数由配置开关动态裁剪——未开启的参数不会出现在 Tool Schema 中。

### WebUI

挂载路径：

```text
/plugins/tts-voice/
```

页面提供：

- 查看和选择语音风格；
- 指定语言、语速和辅助参考音频进行调试合成；
- 上传 WAV 到 `data/tts_voice_plugin-neo/assets/`；
- 编辑服务地址和风格配置并原子保存。

上传接口只接受安全 `.wav` 文件名，拒绝目录型文件名、空文件和超限文件；文件名会清洗掉 Windows 非法字符后原样保留，同名文件直接覆盖。

## 运行机制

1. 插件加载配置；关闭总开关时不暴露任何组件。
2. 启用时创建一个插件持有的 `TTSService`，Action、Command、Router 和 Provider 共享该实例。
3. 风格配置在 Service 初始化和配置保存时装载为强类型 `StyleProfile`，不在每次合成时重建。
4. `GSVClient` 用串行锁保护"模型权重切换 + 合成请求"，避免并发音色串线；权重切换失败立即终止本次合成。
5. `aiohttp.ClientSession` 在客户端内复用，并在插件卸载时关闭。
6. 效果器渲染、WAV 拼接和临时文件写入都通过 `asyncio.to_thread` 派发，不阻塞事件循环。
7. 热重载会注销旧 Provider；Action 描述从稳定基础文本重建，不会重复追加风格列表。

## 已知的框架兼容边界

[`config_persistence.py`](config_persistence.py) 是本插件唯一触碰框架内部实现的位置。公开配置 API 目前只有 `load_config` / `reload_config`，没有"保存一个已校验配置实例"的能力，而 WebUI 的配置编辑必须写回 TOML，因此该模块直接调用了 `src.kernel.config.core._render_toml_with_signature` 以复用框架的带注释渲染逻辑。

公开 API 提供保存能力后应删除该模块。在此之前不要把框架私有导入扩散到其它模块。

## 自动测试

插件测试位于：

```text
plugins/tts_voice_plugin-neo/test/
```

运行：

```bash
uv run pytest plugins/tts_voice_plugin-neo/test -q
uv run ruff check plugins/tts_voice_plugin-neo
```

## 故障排查

### 插件未暴露组件

确认 `[plugin].enable = true`。若需要经 `mfx-tts-http-v1` 协议对外提供语音，还需确认 `tts_http_server` 已加载；未加载时 Provider 注册会跳过，但不影响本插件直接使用。

### 合成返回空音频或 502

- 确认 GPT-SoVITS 服务地址和端口可访问；
- 检查参考音频和模型权重路径；
- 查看权重切换接口 `/set_gpt_weights`、`/set_sovits_weights` 的返回；
- 确认 `/tts` 返回非空音频。

### 指定风格不存在

Service 会回退到第一个风格并记录警告。建议修正调用参数或配置，避免依赖回退行为。

### 效果器未生效

检查日志中是否有"音频效果器配置无效"。参数名必须属于对应效果器类型，取值必须在允许范围内——写错时合成会成功但跳过效果处理。

### WebUI 配置保存失败

配置会先经过 Pydantic 校验，再写入临时文件并原子替换。检查 HTTP 422/400 返回的具体字段错误，以及配置目录写权限。

### WAV 上传被拒绝

只接受 `.wav`，文件名不能包含目录，文件大小不能超过 `[tts].upload_max_mb`。

## 人工验证清单

1. 启动 GPT-SoVITS，使用两个不同风格分别执行 `/tts`。
2. 执行 `/tts file`，确认文件发送成功且本地临时文件被删除。
3. 执行 `/tts 今天天气真好`，确认正文末尾的词没有被当作语言参数吞掉。
4. 通过 WebUI 上传 WAV、保存配置并立即合成。
5. 上传非 WAV、路径型文件名、空文件和超限文件，确认分别被拒绝。
6. 分别在四种 LLM 开关组合下检查 Action Schema，确认参数按配置裁剪。
7. 配置一条效果器链并写入非法参数名，确认合成成功但日志报告配置无效。
8. 热重载插件，确认 Registry 中只有一个 Provider，Action 描述没有重复风格列表。
9. 并发请求不同风格，确认模型切换和合成串行，音色不串。
10. 配置错误权重或停止 GPT-SoVITS，确认合成明确失败而不会继续请求错误模型。

## 许可证

本插件使用 [AGPL-v3.0](LICENSE) 许可证。
