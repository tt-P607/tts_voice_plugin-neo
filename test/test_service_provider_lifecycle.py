"""TTS Service、Provider 与插件生命周期测试。"""

from __future__ import annotations

import io
import wave
from types import SimpleNamespace
from typing import Any

import pytest

from tts_voice_plugin_neo.actions.tts_action import TTSVoiceAction  # noqa: E402
from tts_voice_plugin_neo.config import TTSStyle, TTSVoiceConfig  # noqa: E402
from tts_voice_plugin_neo.plugin import TTSVoicePlugin  # noqa: E402
from tts_voice_plugin_neo.provider import TTSVoiceProvider  # noqa: E402
from tts_voice_plugin_neo.services import audio  # noqa: E402
from tts_voice_plugin_neo.services.gsv_client import GSVError  # noqa: E402
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


@pytest.mark.asyncio
async def test_audio_helpers(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """文件名、WSL 路径、临时文件和 WAV 拼接应可预测。"""

    monkeypatch.chdir(tmp_path)
    assert audio.sanitize_file_name("../晚安?.WAV") == "晚安.wav"
    assert audio.to_wsl_path("C:\\voice\\a.wav") == "/mnt/c/voice/a.wav"

    temp_path = await audio.write_temp_audio(b"audio", "../../unsafe")
    assert temp_path.parent == (tmp_path / "data/tts_voice_plugin-neo").resolve()
    assert temp_path.read_bytes() == b"audio"

    merged = await audio.merge_audio([_wav_bytes(), _wav_bytes()], pause_duration=0.1)
    assert merged is not None
    with wave.open(io.BytesIO(merged)) as wav_file:
        assert wav_file.getnframes() > 32
    assert audio.estimate_duration(merged) > 0
    assert audio.estimate_duration(b"not-a-wav") == 0.0


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
