"""TTS WebUI Router 安全与配置测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tts_voice_plugin_neo.config import TTSStyle, TTSVoiceConfig  # noqa: E402
from tts_voice_plugin_neo.plugin import TTSVoicePlugin  # noqa: E402
from tts_voice_plugin_neo.router import (  # noqa: E402
    ConfigSaveRequest,
    TTSVoiceWebUIRouter,
)
from tts_voice_plugin_neo.services.tts_service import TTSService  # noqa: E402


def _router() -> TTSVoiceWebUIRouter:
    """创建带启用配置的 Router。"""

    config = TTSVoiceConfig()
    config.plugin.enable = True
    return TTSVoiceWebUIRouter(TTSVoicePlugin(config))


def test_safe_upload_name_rejects_paths_and_non_wav() -> None:
    """上传名不得包含目录且必须是 WAV。"""

    with pytest.raises(HTTPException) as path_error:
        TTSVoiceWebUIRouter._safe_upload_name("../evil.wav")
    assert path_error.value.status_code == 400
    with pytest.raises(HTTPException) as type_error:
        TTSVoiceWebUIRouter._safe_upload_name("voice.mp3")
    assert type_error.value.status_code == 400

    safe_name = TTSVoiceWebUIRouter._safe_upload_name("晚安 语音.WAV")
    assert safe_name == "晚安 语音.wav"


def test_safe_upload_name_is_deterministic() -> None:
    """同名文件应生成相同文件名，由上传写入负责覆盖。"""

    first = TTSVoiceWebUIRouter._safe_upload_name("refer.wav")
    second = TTSVoiceWebUIRouter._safe_upload_name("refer.wav")
    assert first == second == "refer.wav"


def test_safe_upload_name_preserves_full_width_punctuation() -> None:
    """中文标点与空格应原样保留，只清洗 Windows 保留字符。"""

    original = "【难过】你会遇到比我更好的人，也可能已经遇见了，你终究还有自己的生活，要去拥抱属于你的明天。.wav"
    assert TTSVoiceWebUIRouter._safe_upload_name(original) == original


def test_safe_upload_name_replaces_windows_reserved_chars() -> None:
    """Windows 保留字符应替换为下划线，其余字符原样保留。"""

    safe_name = TTSVoiceWebUIRouter._safe_upload_name('a<b>c:d"e|f?g*h.wav')
    assert safe_name == "a_b_c_d_e_f_g_h.wav"


def test_merge_config_preserves_hidden_sections_and_validates() -> None:
    """WebUI 保存只覆盖展示字段，并保留高级与效果器配置。"""

    current = TTSVoiceConfig()
    current.plugin.enable = True
    current.tts_advanced.top_k = 22
    request = ConfigSaveRequest(
        server="http://127.0.0.1:9881",
        timeout=90,
        max_text_length=600,
        wsl_mode=True,
        styles=[
            TTSStyle(
                style_name="calm",
                refer_wav_path="calm.wav",
                prompt_text="hello",
            )
        ],
    )
    updated = TTSVoiceWebUIRouter._merge_config(current, request)
    assert updated.tts.server == "http://127.0.0.1:9881"
    assert updated.tts_advanced.top_k == 22
    assert updated.tts_styles[0].style_name == "calm"


def test_config_save_request_reuses_style_constraints() -> None:
    """风格字段的约束由配置模型统一提供，Router 不重复声明。"""

    with pytest.raises(ValueError):
        ConfigSaveRequest(
            server="http://127.0.0.1:9880",
            timeout=90,
            max_text_length=600,
            wsl_mode=False,
            styles=[TTSStyle(style_name="bad/name")],
        )


def test_atomic_config_save_writes_parseable_toml(tmp_path: Path) -> None:
    """配置保存应写出可重新加载的 TOML，且不遗留临时文件。"""

    from tts_voice_plugin_neo.config_persistence import save_config_atomically

    target = tmp_path / "config.toml"
    config = TTSVoiceConfig()
    config.plugin.enable = True
    save_config_atomically(target, config)

    loaded = TTSVoiceConfig.load(target)
    assert loaded.plugin.enable is True
    assert list(tmp_path.glob("*.tmp")) == []


def test_webui_is_external_resource() -> None:
    """WebUI 页面应从独立资源文件加载。"""

    html = TTSVoiceWebUIRouter._load_webui()
    assert "TTS Voice Studio" in html


def test_webui_inference_config_round_trip(tmp_path: Path) -> None:
    """全局数值经 API 保存、热加载、TOML 重读保持一致，风格无覆盖。"""

    config = TTSVoiceConfig()
    config.audio_effects.enabled = True
    plugin = TTSVoicePlugin(config)
    plugin.tts_service = TTSService(plugin)
    router = TTSVoiceWebUIRouter(plugin)
    router.config_path = tmp_path / "config.toml"
    with TestClient(router.app) as client:
        response = client.get("/api/config")
        assert response.status_code == 200
        values = response.json()
        assert values["advanced_schema"]["properties"]["sample_steps"]["default"] == 32
        assert values["advanced_schema"]["properties"]["cfg_rate"]["default"] == 1.3
        assert values["advanced_schema"]["properties"]["use_cuda_graph"]["default"] is True
        values["styles"].append(TTSStyle(style_name="alternate").model_dump())
        assert all({"sample_steps", "cfg_rate"}.isdisjoint(style) for style in values["styles"])
        values["advanced"].update(
            sample_steps=4, cfg_rate=0, batch_size=3,
            parallel_infer=False, use_cuda_graph=False, seed=0,
        )
        response = client.post("/api/config/save", json=values)
        assert response.status_code == 200
        live_styles = client.get("/api/styles").json()
        assert set(live_styles) == {"default", "alternate"}
        assert all({"sample_steps", "cfg_rate"}.isdisjoint(style) for style in live_styles.values())
        assert plugin.tts_service.config.tts_advanced.sample_steps == 4
        assert plugin.tts_service.config.tts_advanced.cfg_rate == 0
        assert client.get("/api/config").json()["advanced"]["batch_size"] == 3
        persisted = router.config_path.read_bytes()
        values["advanced"]["batch_size"] = 0
        assert client.post("/api/config/save", json=values).status_code == 422
        assert router.config_path.read_bytes() == persisted

    loaded = TTSVoiceConfig.load(router.config_path)
    assert loaded.tts_advanced.sample_steps == 4
    assert loaded.tts_advanced.cfg_rate == 0
    assert all({"sample_steps", "cfg_rate"}.isdisjoint(style.model_dump()) for style in loaded.tts_styles)
    assert loaded.tts_advanced.use_cuda_graph is False
    assert loaded.tts_advanced.parallel_infer is False
    assert loaded.tts_advanced.seed == 0
    assert loaded.audio_effects.enabled is True
