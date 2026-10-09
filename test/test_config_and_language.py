"""TTS Voice 配置、语言与组件声明测试。"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from tts_voice_plugin_neo.actions.tts_action import TTSVoiceAction  # noqa: E402
from tts_voice_plugin_neo.config import (  # noqa: E402
    TTSAdvancedSection,
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


def test_inference_defaults_are_numeric() -> None:
    """步数与 CFG 只存在于全局推理配置，默认值是普通数值。"""

    advanced = TTSAdvancedSection()
    style = TTSStyle()
    assert advanced.use_cuda_graph is True
    assert advanced.sample_steps == 32
    assert advanced.cfg_rate == 1.3
    assert {"sample_steps", "cfg_rate"}.isdisjoint(style.model_dump())
    assert TTSAdvancedSection(sample_steps=4, cfg_rate=0).cfg_rate == 0


@pytest.mark.parametrize("steps", [8, 16, 32, 64, 128])
def test_legacy_sample_steps_remain_valid(steps: int) -> None:
    """已有全局采样步数保持有效，不迁入风格。"""

    config = TTSVoiceConfig.model_validate({"tts_advanced": {"sample_steps": steps}})
    assert config.tts_advanced.sample_steps == steps


@pytest.mark.parametrize("values", [{"sample_steps": 32}, {"cfg_rate": 1.3}])
def test_styles_reject_inference_settings(values: dict[str, int | float]) -> None:
    """风格不能再声明推理参数或覆盖全局值。"""

    with pytest.raises(ValidationError):
        TTSStyle.model_validate(values)


@pytest.mark.parametrize("auto_update", [False, True])
def test_global_inference_toml_load(tmp_path: Path, auto_update: bool) -> None:
    """TOML 加载和自动更新均保留全局值及其他配置。"""

    path = tmp_path / "legacy.toml"
    original = (
        "[tts_advanced]\nsample_steps = 16\ncfg_rate = 0.5\nseed = 0\n"
        '[[tts_styles]]\nstyle_name = "first"\n'
        '[[tts_styles]]\nstyle_name = "second"\n'
        '[audio_effects]\nenabled = true\n[[audio_effects.chain]]\n'
        'type = "gain"\nparams = { gain_db = -3.0 }\n'
    )
    path.write_text(original, encoding="utf-8")
    config = TTSVoiceConfig.load(path, auto_update=auto_update)
    assert config.tts_advanced.sample_steps == 16
    assert config.tts_advanced.cfg_rate == 0.5
    assert config.tts_advanced.seed == 0
    assert config.audio_effects.chain[0].params == {"gain_db": -3.0}
    if auto_update:
        persisted = tomllib.loads(path.read_text(encoding="utf-8"))
        assert persisted["tts_advanced"]["sample_steps"] == 16
        assert persisted["tts_advanced"]["cfg_rate"] == 0.5
        assert all({"sample_steps", "cfg_rate"}.isdisjoint(style) for style in persisted["tts_styles"])
        loaded = TTSVoiceConfig.load(path, auto_update=True)
        assert loaded.model_dump() == config.model_dump()
    else:
        assert path.read_text(encoding="utf-8") == original


def test_invalid_global_inference_read_does_not_write(tmp_path: Path) -> None:
    """普通加载拒绝非法全局值，原文件保持不变。"""

    path = tmp_path / "legacy.toml"
    original = b"[tts_advanced]\nsample_steps = 0\n"
    path.write_bytes(original)
    with pytest.raises(ValidationError):
        TTSVoiceConfig.load(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("value", [0, -1, 1.5, "invalid", "auto", "inherit"])
def test_invalid_sample_steps_are_rejected(value: object) -> None:
    """采样步数必须为正整数。"""

    with pytest.raises(ValidationError):
        TTSVoiceConfig.model_validate({"tts_advanced": {"sample_steps": value}})
    with pytest.raises(ValidationError):
        TTSAdvancedSection.model_validate({"sample_steps": value})


@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan"), "invalid", "auto", "inherit"])
def test_invalid_cfg_rates_are_rejected(value: object) -> None:
    """CFG 必须为有限非负数。"""

    with pytest.raises(ValidationError):
        TTSVoiceConfig.model_validate({"tts_advanced": {"cfg_rate": value}})
    with pytest.raises(ValidationError):
        TTSAdvancedSection.model_validate({"cfg_rate": value})


def test_voice_context_defaults() -> None:
    """语音上下文默认关闭，开启后使用合成文本。"""

    assert TTSVoiceConfig().tts.include_voice_context is False
    assert TTSVoiceConfig().tts.voice_context_source == "text"
    with pytest.raises(ValidationError):
        TTSVoiceConfig().tts.model_validate({"voice_context_source": "invalid"})


def test_rule_reminder_defaults_on() -> None:
    """语音规则提醒默认开启，显式关闭配置保持有效。"""
    assert TTSVoiceConfig().prompt.inject_rule_reminder is True
    assert TTSVoiceConfig.model_validate({"prompt": {}}).prompt.inject_rule_reminder is True
    disabled = TTSVoiceConfig.model_validate({"prompt": {"inject_rule_reminder": False}})
    assert disabled.prompt.inject_rule_reminder is False


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
