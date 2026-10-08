"""TTS Service、Provider 与插件生命周期测试。"""

from __future__ import annotations

import base64
import io
import wave
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from tts_voice_plugin_neo.actions.tts_action import TTSVoiceAction  # noqa: E402
from tts_voice_plugin_neo.commands.tts_command import TTSVoiceCommand  # noqa: E402
from tts_voice_plugin_neo.config import TTSAdvancedSection, TTSStyle, TTSVoiceConfig  # noqa: E402
from tts_voice_plugin_neo.plugin import TTSVoicePlugin  # noqa: E402
from tts_voice_plugin_neo.provider import TTSVoiceProvider  # noqa: E402
from tts_voice_plugin_neo.services import audio  # noqa: E402
from tts_voice_plugin_neo.services import voice_delivery  # noqa: E402
from tts_voice_plugin_neo.services.gsv_client import GSVClient, GSVError  # noqa: E402
from tts_voice_plugin_neo.services.styles import load_styles  # noqa: E402
from tts_voice_plugin_neo.services.tts_service import TTSService  # noqa: E402


def _enabled_config() -> TTSVoiceConfig:
    """创建可实例化 Service 的启用配置。"""

    config = TTSVoiceConfig(
        tts_styles=[
            TTSStyle(
                style_name="default",
                refer_wav_path="reference.wav",
                prompt_text="参考文本",
                gpt_weights="gpt.ckpt",
                sovits_weights="sovits.pth",
            )
        ]
    )
    config.plugin.enable = True
    return config


def _wav_bytes(frames: int = 16, rate: int = 8000) -> bytes:
    """生成最小 PCM WAV 数据。"""

    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00\x00" * frames)
    return output.getvalue()


def test_styles_reject_duplicate_names() -> None:
    """重复风格名称应明确失败，而不是静默覆盖。"""

    config = _enabled_config()
    config.tts_styles.append(config.tts_styles[0].model_copy())
    with pytest.raises(ValueError, match="风格名称重复"):
        load_styles(config)


def test_disabled_style_not_loaded() -> None:
    """enabled=False 的风格不进入注册表。"""

    config = _enabled_config()
    config.tts_styles.append(
        TTSStyle(
            style_name="calm",
            refer_wav_path="calm.wav",
            prompt_text="",
            gpt_weights="",
            sovits_weights="",
        )
    )
    config.tts_styles.append(
        TTSStyle(
            style_name="hidden",
            refer_wav_path="hidden.wav",
            prompt_text="不应出现",
            gpt_weights="hidden.ckpt",
            sovits_weights="hidden.pth",
        )
    )
    config.tts_styles[2].enabled = False

    registry = load_styles(config)
    assert registry.names == ["default", "calm"]
    assert "hidden" not in registry


def test_all_styles_disabled_raises() -> None:
    """所有风格都禁用时应明确报错。"""

    config = _enabled_config()
    config.tts_styles[0].enabled = False
    with pytest.raises(ValueError, match="没有启用的 TTS 风格"):
        load_styles(config)


def test_disabled_first_style_fallback_to_next_enabled() -> None:
    """首个风格被禁用时，回退基准取下一个启用风格。"""

    config = _enabled_config()
    # 把 default 禁用，新增一个启用风格作为基准
    config.tts_styles[0].enabled = False
    config.tts_styles.append(
        TTSStyle(
            style_name="calm",
            refer_wav_path="calm.wav",
            prompt_text="基准文本",
            gpt_weights="calm_gpt.ckpt",
            sovits_weights="calm_sovits.pth",
        )
    )
    config.tts_styles.append(
        TTSStyle(
            style_name="extra",
            refer_wav_path="extra.wav",
            prompt_text="",
            gpt_weights="",
            sovits_weights="",
        )
    )
    registry = load_styles(config)
    assert registry.names == ["calm", "extra"]
    # extra 留空的字段应回退到 calm（首个启用风格）的值
    extra = registry.resolve("extra")
    assert extra.prompt_text == "基准文本"
    assert extra.gpt_weights == "calm_gpt.ckpt"
    assert extra.sovits_weights == "calm_sovits.pth"


def test_style_fallback_and_override() -> None:
    """未命中的风格名回退到首个风格，覆盖参数只影响副本。"""

    config = _enabled_config()
    config.tts_styles.append(
        TTSStyle(
            style_name="calm",
            refer_wav_path="calm.wav",
            prompt_text="",
            gpt_weights="",
            sovits_weights="",
        )
    )
    registry = load_styles(config)

    assert registry.names == ["default", "calm"]
    assert registry.resolve("missing").name == "default"
    # 留空的提示文本与权重回退到首个风格的值。
    calm = registry.resolve("calm")
    assert calm.refer_wav_path == "calm.wav"
    assert calm.prompt_text == "参考文本"
    assert calm.gpt_weights == "gpt.ckpt"
    assert calm.sovits_weights == "sovits.pth"

    base = registry.resolve("default")
    overridden = base.override(speed_factor=1.5)
    assert overridden.speed_factor == 1.5
    assert base.speed_factor == 1.0


def test_payload_sends_numeric_inference_defaults() -> None:
    """请求明确发送全局默认数值，不依赖后端选择默认值。"""

    style = load_styles(_enabled_config()).resolve("default")
    payload = GSVClient._build_payload(style, "文本", "zh", TTSAdvancedSection(), None)
    assert payload["sample_steps"] == 32
    assert payload["cfg_rate"] == 1.3
    assert payload["use_cuda_graph"] is True
    assert payload["parallel_infer"] is True
    assert payload["ref_audio_path"] == "reference.wav"
    assert payload["prompt_text"] == "参考文本"
    assert payload["speed_factor"] == 1.0
    assert payload["media_type"] == "wav"


@pytest.mark.parametrize(
    ("advanced_values", "expected"),
    [
        ({}, {"sample_steps": 32, "cfg_rate": 1.3}),
        ({"sample_steps": 4, "cfg_rate": 0}, {"sample_steps": 4, "cfg_rate": 0}),
        ({"sample_steps": 8}, {"sample_steps": 8, "cfg_rate": 1.3}),
        ({"cfg_rate": 0}, {"sample_steps": 32, "cfg_rate": 0}),
    ],
)
def test_payload_global_inference_settings(
    advanced_values: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    """不同风格使用相同全局步数和 CFG，零 CFG 不得被丢弃。"""

    config = _enabled_config()
    config.tts_styles.append(TTSStyle(style_name="alternate", refer_wav_path="alternate.wav"))
    advanced = TTSAdvancedSection.model_validate(advanced_values)
    for _, style in load_styles(config).items():
        payload = GSVClient._build_payload(style, "text", "en", advanced, None)
        assert {key: payload[key] for key in ("sample_steps", "cfg_rate")} == expected
        assert payload["ref_audio_path"] == style.refer_wav_path
    assert advanced.model_dump() | expected == advanced.model_dump()


@pytest.mark.parametrize("media_type", ["wav", "ogg", "aac", "raw"])
def test_payload_complete_audio_requests_wav(media_type: str) -> None:
    """完整音频始终请求带采样率的 WAV，流式格式配置保持原样。"""

    style = load_styles(_enabled_config()).resolve("default")
    advanced = TTSAdvancedSection.model_validate({"media_type": media_type})
    payload = GSVClient._build_payload(style, "text", "en", advanced, None)
    assert payload["media_type"] == "wav"
    assert payload["streaming_mode"] is False
    assert GSVClient._build_payload(style, "text", "en", advanced, 1)["media_type"] == media_type
    assert advanced.media_type == media_type


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [8000, 32000, 48000])
async def test_audio_helpers(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, rate: int,
) -> None:
    """临时文件隔离，拼接与停顿时长按 WAV 原生采样率计算。"""

    monkeypatch.chdir(tmp_path)
    assert audio.sanitize_file_name("../晚安?.WAV") == "晚安.wav"
    assert audio.to_wsl_path("C:\\voice\\a.wav") == "/mnt/c/voice/a.wav"

    temp_path = await audio.write_temp_audio(b"audio", "../../unsafe")
    assert temp_path.parent == (tmp_path / "data/tts_voice_plugin-neo").resolve()
    assert temp_path.read_bytes() == b"audio"

    frames = rate // 10
    merged = await audio.merge_audio(
        [_wav_bytes(frames=frames, rate=rate), _wav_bytes(frames=frames, rate=rate)],
        pause_duration=0.1,
    )
    assert merged is not None
    with wave.open(io.BytesIO(merged)) as wav_file:
        assert wav_file.getframerate() == rate
        assert wav_file.getnframes() == 3 * frames
    assert audio.estimate_duration(merged) == pytest.approx(0.3)
    assert audio.estimate_duration(b"not-a-wav") == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [32000, 48000])
async def test_service_provider_http_audio_round_trip(rate: int) -> None:
    """模型切换先于合成，数值参数贯通 HTTP，Provider 保留原生 WAV。"""

    calls: list[tuple[str, dict[str, Any]]] = []
    raw = _wav_bytes(frames=rate // 4, rate=rate)

    async def switch_weights(request: web.Request) -> web.Response:
        """记录隔离服务收到的权重切换请求。"""
        calls.append((request.path, dict(request.query)))
        return web.json_response({"message": "success"})

    async def synthesize(request: web.Request) -> web.Response:
        """记录合成参数并返回已知采样率的 WAV。"""
        calls.append((request.path, await request.json()))
        return web.Response(body=raw, content_type="audio/wav")

    app = web.Application()
    app.router.add_get("/set_gpt_weights", switch_weights)
    app.router.add_get("/set_sovits_weights", switch_weights)
    app.router.add_post("/tts", synthesize)
    async with TestServer(app) as server:
        config = _enabled_config()
        config.tts.server = str(server.make_url(""))
        config.tts_styles.append(TTSStyle(
            style_name="alternate", refer_wav_path="alternate.wav", prompt_text="hello",
            prompt_language="en", gpt_weights="alternate.ckpt", sovits_weights="alternate.pth",
            aux_refer_wav_paths=["aux.wav"],
        ))
        config.tts_advanced.seed = 0
        config.tts_advanced.batch_size = 2
        service = TTSService(TTSVoicePlugin(config))
        try:
            assert await service.generate_voice_bytes("hello", language_hint="en") == raw
            config.tts_advanced.sample_steps = 4
            config.tts_advanced.cfg_rate = 0
            provider = TTSVoiceProvider(service)
            response = await provider.synthesize(SimpleNamespace(
                text="hello", options={"style": "alternate", "language": "en", "speed": 1.2},
                markers={},
            ))
            assert base64.b64decode(response.audio_base64) == raw
            assert response.mime_type == "audio/wav"
            assert response.format == "wav"
            assert response.duration_ms == 250
            assert await service.generate_voice_bytes("again", style_hint="alternate") == raw
        finally:
            await service._client.close()

    assert [path for path, _ in calls] == [
        "/set_gpt_weights", "/set_sovits_weights", "/tts",
        "/set_gpt_weights", "/set_sovits_weights", "/tts", "/tts",
    ]
    assert calls[0][1] == {"weights_path": "gpt.ckpt"}
    assert calls[1][1] == {"weights_path": "sovits.pth"}
    assert calls[3][1] == {"weights_path": "alternate.ckpt"}
    assert calls[4][1] == {"weights_path": "alternate.pth"}
    assert calls[2][1]["sample_steps"] == 32
    assert calls[2][1]["cfg_rate"] == 1.3
    payload = calls[5][1]
    assert payload["sample_steps"] == 4
    assert payload["cfg_rate"] == 0
    assert calls[6][1]["sample_steps"] == 4
    assert calls[6][1]["cfg_rate"] == 0
    assert payload["ref_audio_path"] == "alternate.wav"
    assert payload["aux_ref_audio_paths"] == ["aux.wav"]
    assert payload["prompt_lang"] == "en"
    assert payload["text_lang"] == "en"
    assert payload["speed_factor"] == 1.2
    assert payload["batch_size"] == 2
    assert payload["seed"] == 0
    assert payload["use_cuda_graph"] is True
    assert payload["streaming_mode"] is False
    assert payload["media_type"] == "wav"


@pytest.mark.asyncio
@pytest.mark.parametrize("sent", [True, False])
async def test_voice_delivery_returns_send_result(
    monkeypatch: pytest.MonkeyPatch, sent: bool,
) -> None:
    """语音发送只使用标准媒体入口，并保留框架发送结果。"""

    calls: list[tuple[str, str, str | None]] = []

    async def _send_voice(voice_data: str, stream_id: str, processed_plain_text: str | None) -> bool:
        calls.append((voice_data, stream_id, processed_plain_text))
        return sent

    monkeypatch.setattr(voice_delivery, "send_voice", _send_voice)

    assert await voice_delivery.send_voice_audio(b"audio", "stream") is sent
    assert calls == [("YXVkaW8=", "stream", None)]


@pytest.mark.asyncio
async def test_voice_delivery_context_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """文本直传和 ASR 描述均走标准语音媒体入口。"""

    calls: list[tuple[str, Any]] = []

    async def _send_voice(**kwargs: Any) -> bool:
        calls.append(("voice", kwargs["processed_plain_text"]))
        return True

    async def _send_media(*args: Any, **kwargs: Any) -> bool:
        calls.append(("media", kwargs["context_mode"]))
        return True

    monkeypatch.setattr(voice_delivery, "send_voice", _send_voice)
    monkeypatch.setattr(voice_delivery, "send_media", _send_media)

    await voice_delivery.send_voice_audio(b"audio", "stream", "原文")
    await voice_delivery.send_voice_audio(b"audio", "stream", "原文", "asr")

    assert calls == [
        ("voice", "原文"),
        ("media", "description"),
    ]


@pytest.mark.asyncio
async def test_action_context_tracks_successful_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """合成失败的段落不会占据后续语音的上下文文本。"""

    config = _enabled_config()
    config.tts.include_voice_context = True
    plugin = TTSVoicePlugin(config)
    plugin.tts_service = TTSService(plugin)
    action = TTSVoiceAction(SimpleNamespace(stream_id="stream"), plugin)
    sent: list[tuple[bytes, str | None]] = []

    async def _generate(**kwargs: Any) -> bytes | None:
        return None if kwargs["text"] == "失败段" else kwargs["text"].encode("utf-8")

    async def _send(**kwargs: Any) -> bool:
        sent.append((kwargs["audio_bytes"], kwargs["context_text"]))
        return True

    monkeypatch.setattr(plugin.tts_service, "generate_voice_bytes", _generate)
    monkeypatch.setattr("tts_voice_plugin_neo.actions.tts_action.send_voice_audio", _send)

    assert await action.execute(tts_segments=[
        {"text": "第一段"}, {"text": "失败段"}, {"text": "第三段"},
    ]) == (True, "成功发送 2 段语音，总文本长度: 6 字符")
    assert sent == [("第一段".encode("utf-8"), "第一段"), ("第三段".encode("utf-8"), "第三段")]


@pytest.mark.asyncio
async def test_action_stops_when_voice_delivery_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """语音未缓存或平台发送失败时不报告成功，也不继续发后续段落。"""
    plugin = TTSVoicePlugin(_enabled_config())
    plugin.tts_service = TTSService(plugin)
    action = TTSVoiceAction(SimpleNamespace(stream_id="stream"), plugin)
    delivered: list[bytes] = []

    async def _generate(**kwargs: Any) -> bytes:
        return kwargs["text"].encode("utf-8")

    async def _send(**kwargs: Any) -> bool:
        delivered.append(kwargs["audio_bytes"])
        return False

    monkeypatch.setattr(plugin.tts_service, "generate_voice_bytes", _generate)
    monkeypatch.setattr("tts_voice_plugin_neo.actions.tts_action.send_voice_audio", _send)

    assert await action.execute(tts_segments=[
        {"text": "第一段"}, {"text": "第二段"},
    ]) == (False, "第 1/2 段语音发送失败")
    assert delivered == ["第一段".encode("utf-8")]


@pytest.mark.asyncio
async def test_merged_voice_context_joins_successful_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """合并发送时仅注入已合成段落的文本。"""

    config = _enabled_config()
    config.tts.include_voice_context = True
    plugin = TTSVoicePlugin(config)
    plugin.tts_service = TTSService(plugin)
    action = TTSVoiceAction(SimpleNamespace(stream_id="stream"), plugin)
    sent: list[str | None] = []

    async def _generate(**kwargs: Any) -> bytes | None:
        return None if kwargs["text"] == "失败段" else b"audio"

    async def _merge(*args: Any) -> bytes:
        return b"merged"

    async def _send(**kwargs: Any) -> bool:
        sent.append(kwargs["context_text"])
        return True

    monkeypatch.setattr(plugin.tts_service, "generate_voice_bytes", _generate)
    monkeypatch.setattr(audio, "merge_audio", _merge)
    monkeypatch.setattr("tts_voice_plugin_neo.actions.tts_action.send_voice_audio", _send)

    await action.execute(tts_segments=[
        {"text": "第一段"}, {"text": "失败段"}, {"text": "第三段"},
    ], merge_voice=True)
    assert sent == ["第一段 第三段"]


@pytest.mark.asyncio
async def test_merged_voice_delivery_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """合并后的语音发送失败时 Action 不回报成功。"""
    plugin = TTSVoicePlugin(_enabled_config())
    action = TTSVoiceAction(SimpleNamespace(stream_id="stream"), plugin)

    async def _merge(*args: Any) -> bytes:
        return b"merged"

    async def _send(**kwargs: Any) -> bool:
        return False

    monkeypatch.setattr(audio, "merge_audio", _merge)
    monkeypatch.setattr("tts_voice_plugin_neo.actions.tts_action.send_voice_audio", _send)

    assert await action._send_merged_voice([b"audio"], ["文本"], 0) == (
        False, "拼接语音发送失败",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery_ok", [True, False])
async def test_command_passes_synthesis_text_to_voice_delivery(
    monkeypatch: pytest.MonkeyPatch, delivery_ok: bool,
) -> None:
    """命令后台任务透传文本，并在发送失败时告知会话。"""

    config = _enabled_config()
    config.tts.include_voice_context = True
    plugin = TTSVoicePlugin(config)
    plugin.tts_service = TTSService(plugin)
    command = TTSVoiceCommand(plugin, "stream")
    pending: list[Any] = []
    sent: list[tuple[str | None, str]] = []
    feedback: list[str] = []

    async def _generate(*args: Any) -> bytes:
        return b"audio"

    async def _send(**kwargs: Any) -> bool:
        sent.append((kwargs["context_text"], kwargs["context_source"]))
        return delivery_ok

    async def _send_text(content: str, stream_id: str) -> bool:
        feedback.append(content)
        return True

    class _TaskManager:
        def create_task(self, coroutine: Any, **kwargs: Any) -> Any:
            pending.append(coroutine)
            return SimpleNamespace(task_id="task")

    monkeypatch.setattr(plugin.tts_service, "generate_voice_bytes", _generate)
    monkeypatch.setattr("tts_voice_plugin_neo.commands.tts_command.send_voice_audio", _send)
    monkeypatch.setattr("tts_voice_plugin_neo.commands.tts_command.send_text", _send_text)
    monkeypatch.setattr("tts_voice_plugin_neo.commands.tts_command.get_task_manager", _TaskManager)

    assert await command._dispatch(("今晚", "晚安"), as_file=False) == (True, "TTS voice 任务已提交")
    await pending.pop()
    assert sent == [("今晚 晚安", "text")]
    assert feedback == ([] if delivery_ok else ["语音发送失败"])


@pytest.mark.asyncio
async def test_service_returns_none_when_client_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GSV 请求失败时服务返回 None，不向上抛异常。"""

    service = TTSService(TTSVoicePlugin(_enabled_config()))

    async def _fail(*_args: Any, **_kwargs: Any) -> bytes:
        raise GSVError("boom")

    monkeypatch.setattr(service._client, "synthesize", _fail)
    assert await service.generate_voice_bytes("你好") is None


@pytest.mark.asyncio
async def test_service_skips_empty_text() -> None:
    """清洗后为空的文本不应触发任何网络请求。"""

    service = TTSService(TTSVoicePlugin(_enabled_config()))
    assert await service.generate_voice_bytes("（只有动作描写）") is None


@pytest.mark.asyncio
async def test_provider_parses_request_and_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider 应解析结构请求并返回本地兼容对象。"""

    service = TTSService(TTSVoicePlugin(_enabled_config()))

    async def _generate(**kwargs: Any) -> bytes:
        assert kwargs["style_hint"] == "default"
        assert kwargs["language_hint"] == "ja"
        assert kwargs["speed_factor"] == 1.2
        assert kwargs["audio_effects"] == [{"type": "gain", "gain_db": 1}]
        return _wav_bytes()

    monkeypatch.setattr(service, "generate_voice_bytes", _generate)
    provider = TTSVoiceProvider(service)
    response = await provider.synthesize(
        SimpleNamespace(
            text="hello",
            options={"language": "ja", "speed": 1.2, "effects": ['{"type":"gain","gain_db":1}']},
            markers={},
        )
    )
    assert response.provider == "tts_voice_plugin-neo"
    assert response.text == "hello"
    assert response.duration_ms is not None
    assert provider.get_capabilities().style_guide is not None

    with pytest.raises(ValueError, match="speed"):
        await provider.synthesize(
            SimpleNamespace(text="hello", options={"speed": 5}, markers={})
        )


@pytest.mark.asyncio
async def test_plugin_registers_and_unregisters_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider 注册与注销必须成对，Action 描述刷新应幂等。"""

    calls: list[tuple[str, Any]] = []

    class _Registry:
        def register_provider(self, provider: Any, *, default: bool = False) -> None:
            calls.append(("register", (provider.provider_name, default)))

        def unregister_provider(self, provider_name: str) -> bool:
            calls.append(("unregister", provider_name))
            return True

    plugin = TTSVoicePlugin(_enabled_config())
    monkeypatch.setattr(plugin, "_get_provider_registry", lambda: _Registry())
    await plugin.on_plugin_loaded()

    first_description = TTSVoiceAction.description
    assert "default" in first_description
    plugin.refresh_action_description()
    assert TTSVoiceAction.description == first_description

    await plugin.on_plugin_unloaded()
    assert calls == [
        ("register", ("tts_voice_plugin-neo", True)),
        ("unregister", "tts_voice_plugin-neo"),
    ]
    assert plugin.tts_service is None


@pytest.mark.asyncio
async def test_plugin_loads_without_provider_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺少 tts_http_server Registry 时插件应独立加载，不抛异常。"""

    plugin = TTSVoicePlugin(_enabled_config())
    monkeypatch.setattr(plugin, "_get_provider_registry", lambda: None)
    await plugin.on_plugin_loaded()

    assert plugin.tts_service is not None
    assert not plugin._provider_registered

    await plugin.on_plugin_unloaded()
    assert plugin.tts_service is None


@pytest.mark.asyncio
async def test_plugin_unload_cancels_registered_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """插件卸载应取消所有仍登记的命令任务。"""

    plugin = TTSVoicePlugin(_enabled_config())
    plugin.register_command_task("task-1")
    cancelled: list[str] = []
    monkeypatch.setattr(
        "tts_voice_plugin_neo.plugin.get_task_manager",
        lambda: SimpleNamespace(cancel_task=lambda task_id: cancelled.append(task_id) or True),
    )
    await plugin.on_plugin_unloaded()
    assert cancelled == ["task-1"]
    assert plugin._command_task_ids == set()


@pytest.mark.asyncio
async def test_service_close_releases_session() -> None:
    """Service close 应关闭可复用 HTTP 会话。"""

    service = TTSService(TTSVoicePlugin(_enabled_config()))
    session = await service._client._get_session()
    assert not session.closed
    await service.close()
    assert session.closed


@pytest.mark.asyncio
async def test_streaming_requires_enable_flag() -> None:
    """Provider 流式接口应受配置开关控制。"""

    provider = TTSVoiceProvider(TTSService(TTSVoicePlugin(_enabled_config())))
    generator = provider.synthesize_stream("hello")
    with pytest.raises(RuntimeError, match="未启用"):
        await anext(generator)
    await generator.aclose()
