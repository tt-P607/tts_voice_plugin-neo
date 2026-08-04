"""TTS WebUI Router 安全与配置测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from tts_voice_plugin_neo.config import TTSStyle, TTSVoiceConfig  # noqa: E402
from tts_voice_plugin_neo.plugin import TTSVoicePlugin  # noqa: E402
from tts_voice_plugin_neo.router import (  # noqa: E402
    ConfigSaveRequest,
    TTSVoiceWebUIRouter,
)


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
