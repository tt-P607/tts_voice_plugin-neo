"""TTS Voice 配置、语言与组件声明测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tts_voice_plugin_neo.actions.tts_action import TTSVoiceAction  # noqa: E402
from tts_voice_plugin_neo.config import (  # noqa: E402
    TTSStreamingSection,
    TTSStyle,
    TTSVoiceConfig,
)
from tts_voice_plugin_neo.language import (  # noqa: E402
    normalize_language_code,
    resolve_language_token,
)
from tts_voice_plugin_neo.plugin import TTSVoicePlugin  # noqa: E402


PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _config(**plugin_values: bool) -> TTSVoiceConfig:
    """创建启用的测试配置。"""

    config = TTSVoiceConfig()
    config.plugin.enable = True
    for key, value in plugin_values.items():
        setattr(config.plugin, key, value)
    return config


def test_mutable_defaults_are_isolated() -> None:
    """列表默认值不应跨配置实例共享。"""

    first = TTSVoiceConfig()
    second = TTSVoiceConfig()
    first.plugin.keywords.append("only-first")
    first.tts_styles[0].aux_refer_wav_paths.append("a.wav")
    assert "only-first" not in second.plugin.keywords
    assert second.tts_styles[0].aux_refer_wav_paths == []


def test_config_constraints_reject_invalid_values() -> None:
    """关键数值和枚举约束应在模型层拒绝非法值。"""

    with pytest.raises(ValidationError):
        TTSStreamingSection(chunk_size=1)
    with pytest.raises(ValidationError):
        TTSStyle(style_name="bad/name")
    with pytest.raises(ValidationError):
        TTSStyle(speed_factor=3.0)


def test_voice_delivery_defaults_to_base64() -> None:
    """语音发送默认使用兼容性更好的 Base64。"""

    assert TTSVoiceConfig().tts.use_base64 is True
    assert TTSVoiceConfig().tts.include_voice_context is False
    assert TTSVoiceConfig().tts.voice_context_source == "text"
    with pytest.raises(ValidationError):
        TTSVoiceConfig().tts.model_validate({"voice_context_source": "invalid"})


def test_language_normalization_covers_alias_and_fallback() -> None:
    """语言归一化应覆盖别名、形态和 fallback。"""

    assert normalize_language_code("zh-CN")[0] == "zh"
    assert normalize_language_code("japanese")[0] == "ja"
    assert normalize_language_code("yuee")[0] == "yue"
    assert normalize_language_code("not-a-language") == ("zh", "fallback")


def test_language_token_resolution_is_strict() -> None:
    """命令参数解析不做模糊匹配，正文词不应被误判为语言。"""

    assert resolve_language_token("纯中文") == "all_zh"
    assert resolve_language_token("JA") == "ja"
    assert resolve_language_token("yuee") is None
    assert resolve_language_token("晚安") is None


@pytest.mark.parametrize(
    ("speed", "effects", "expected"),
    [
        (False, False, set()),
        (True, False, {"speed_factor"}),
        (False, True, {"audio_effects"}),
        (
            True,
            True,
            {
                "speed_factor",
                "audio_effects",
                "aux_refer_wav_paths",
                "merge_voice",
                "pause_duration",
            },
        ),
    ],
)
def test_action_schema_exposes_configured_params(
    speed: bool,
    effects: bool,
    expected: set[str],
) -> None:
    """Action 只向模型暴露当前配置开启的可选参数。"""

    plugin = TTSVoicePlugin(_config(llm_speed_control=speed, llm_audio_effects=effects))
    assert TTSVoiceAction in plugin.get_components()

    properties = TTSVoiceAction.to_schema()["function"]["parameters"]["properties"]
    gated = {
        "speed_factor",
        "audio_effects",
        "aux_refer_wav_paths",
        "merge_voice",
        "pause_duration",
    }
    assert gated & set(properties) == expected
    # 基础参数在任何配置下都必须存在。
    assert {"tts_segments", "send_mode", "voice_style", "text_language"} <= set(properties)


def test_disabled_plugin_exposes_no_components() -> None:
    """总开关关闭时不得注册组件。"""

    assert TTSVoicePlugin(TTSVoiceConfig()).get_components() == []


def test_manifest_matches_components_and_version_policy() -> None:
    """manifest 应与默认启用配置的组件及版本真源规则一致。"""

    manifest = json.loads((PLUGIN_DIR / "manifest.json").read_text(encoding="utf-8"))
    plugin = TTSVoicePlugin(_config())
    declared = {(item["component_type"], item["component_name"]) for item in manifest["include"]}
    actual: set[tuple[str, str]] = set()
    for component in plugin.get_components():
        comp_type = getattr(component, "component_type", "")
        comp_name = getattr(component, "name", "")
        if comp_type and comp_name:
            actual.add((comp_type, comp_name))
    actual.add(("config", TTSVoiceConfig.name))
    assert declared == actual
    assert "plugin_version" not in TTSVoicePlugin.__dict__
