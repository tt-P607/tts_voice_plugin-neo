"""音频效果器链构建与渲染测试。"""

from __future__ import annotations

import io
import wave

import pytest
from pedalboard import Gain, Reverb  # type: ignore[attr-defined]

from tts_voice_plugin_neo.services.effects import (  # noqa: E402
    EFFECT_SPECS,
    apply_effects,
    build_effect_chain,
)


def _wav_bytes(frames: int = 8000, rate: int = 8000) -> bytes:
    """生成一段静音 WAV 数据。"""

    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00\x00" * frames)
    return output.getvalue()


def test_build_chain_accepts_flat_and_nested_params() -> None:
    """效果器参数可平铺在对象里，也可收在 params 子字典里。"""

    flat = build_effect_chain([{"type": "gain", "gain_db": -3.0}])
    nested = build_effect_chain([{"type": "gain", "params": {"gain_db": -3.0}}])
    assert isinstance(flat[0], Gain)
    assert isinstance(nested[0], Gain)
    assert flat[0].gain_db == nested[0].gain_db == -3.0


def test_build_chain_applies_defaults_and_aliases() -> None:
    """未填写的参数使用默认值，cutoff_hz 别名映射到实际参数名。"""

    reverb = build_effect_chain([{"type": "reverb"}])[0]
    assert isinstance(reverb, Reverb)
    assert reverb.room_size == pytest.approx(0.3)

    lowpass = build_effect_chain([{"type": "lowpass", "cutoff_hz": 1200}])[0]
    assert lowpass.cutoff_frequency_hz == pytest.approx(1200)


def test_build_chain_rejects_invalid_input() -> None:
    """未知效果器、未知参数与越界取值都必须明确失败。"""

    with pytest.raises(ValueError, match="未知的效果器类型"):
        build_effect_chain([{"type": "nonexistent"}])
    with pytest.raises(ValueError, match="不支持参数"):
        build_effect_chain([{"type": "gain", "room_size": 0.5}])
    with pytest.raises(ValueError, match="必须位于"):
        build_effect_chain([{"type": "gain", "gain_db": 999}])
    with pytest.raises(ValueError, match="必须是数字"):
        build_effect_chain([{"type": "gain", "gain_db": "loud"}])


def test_int_params_are_coerced() -> None:
    """声明为整数的参数应转换为 int，避免传给 pedalboard 时报错。"""

    bitcrush = build_effect_chain([{"type": "bitcrush", "bit_depth": 8.4}])[0]
    assert bitcrush.bit_depth == 8


def test_every_spec_builds_with_defaults() -> None:
    """规格表中的每种效果器都能仅用默认值构建。"""

    for effect_type in EFFECT_SPECS:
        assert build_effect_chain([{"type": effect_type}])


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [8000, 32000, 48000])
async def test_apply_effects_returns_wav(rate: int) -> None:
    """效果链保留不同模型输出的采样率与音频时长。"""

    processed = await apply_effects(
        _wav_bytes(frames=rate, rate=rate), [{"type": "gain", "gain_db": -6.0}]
    )
    with wave.open(io.BytesIO(processed)) as wav_file:
        assert wav_file.getframerate() == rate
        assert wav_file.getnframes() == rate


@pytest.mark.asyncio
async def test_apply_effects_without_chain_is_passthrough() -> None:
    """空效果链原样返回输入，不做任何解码。"""

    raw = _wav_bytes()
    assert await apply_effects(raw, []) is raw
