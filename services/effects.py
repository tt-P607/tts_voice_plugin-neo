"""基于 pedalboard 的音频效果器链。

效果器由一张规格表（:data:`EFFECT_SPECS`）驱动：每种效果器声明自己的参数名、
取值范围与默认值，构建时按规格校验，非法参数直接报错而不是静默忽略。
渲染是 CPU 密集的同步运算，通过 :func:`apply_effects` 派发到线程池执行。
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import Any, Callable, Final

import numpy as np
import soundfile as sf
from pedalboard import (  # type: ignore[attr-defined]
    Bitcrush,
    Chorus,
    Clipping,
    Compressor,
    Delay,
    Distortion,
    GSMFullRateCompressor,
    Gain,
    HighpassFilter,
    LadderFilter,
    LowpassFilter,
    MP3Compressor,
    NoiseGate,
    Phaser,
    PitchShift,
    Resample,
    Reverb,
)
from pedalboard._pedalboard import Pedalboard
from pedalboard.io import AudioFile

from src.app.plugin_system.api.log_api import get_logger

logger = get_logger("tts_voice_plugin-neo.effects")

# 会产生尾音的效果器；渲染时需要额外补静音让尾音自然衰减。
_TAIL_EFFECT_TYPES: Final[frozenset[str]] = frozenset(
    {"reverb", "delay", "phaser", "chorus"}
)
# 尾音 padding 时长（秒）与裁剪时的静音判定阈值。
_TAIL_PADDING_SECONDS: Final[float] = 5.0
_TAIL_SILENCE_THRESHOLD: Final[float] = 1e-4
_TAIL_KEEP_SECONDS: Final[float] = 0.1


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """一个效果器参数的取值规格。"""

    default: float
    minimum: float
    maximum: float
    is_int: bool = False

    def coerce(self, effect_type: str, name: str, raw: Any) -> float | int:
        """校验并转换单个参数值。

        Args:
            effect_type: 所属效果器类型，用于错误信息。
            name: 参数名，用于错误信息。
            raw: 原始输入值。

        Returns:
            转换后的数值。

        Raises:
            ValueError: 值不是数字或超出允许范围。
        """
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"效果器 {effect_type} 的参数 {name} 必须是数字") from error
        if not self.minimum <= value <= self.maximum:
            raise ValueError(
                f"效果器 {effect_type} 的参数 {name} 必须位于 "
                f"{self.minimum} 到 {self.maximum} 之间"
            )
        return int(round(value)) if self.is_int else value


@dataclass(frozen=True, slots=True)
class EffectSpec:
    """一种效果器的构建规格。"""

    factory: Callable[..., Any]
    params: dict[str, ParamSpec]
    extra_kwargs: dict[str, Any] | None = None

    def build(self, effect_type: str, raw_params: dict[str, Any]) -> Any:
        """按规格构建 pedalboard 效果器实例。

        Args:
            effect_type: 效果器类型名。
            raw_params: 调用方提供的参数字典。

        Returns:
            pedalboard 效果器实例。

        Raises:
            ValueError: 存在未知参数或参数取值非法。
        """
        unknown = set(raw_params) - set(self.params)
        if unknown:
            raise ValueError(
                f"效果器 {effect_type} 不支持参数: {', '.join(sorted(unknown))}"
            )
        kwargs: dict[str, Any] = {
            name: spec.coerce(effect_type, name, raw_params[name])
            if name in raw_params
            else (int(spec.default) if spec.is_int else spec.default)
            for name, spec in self.params.items()
        }
        if self.extra_kwargs:
            kwargs.update(self.extra_kwargs)
        return self.factory(**kwargs)


EFFECT_SPECS: Final[dict[str, EffectSpec]] = {
    "reverb": EffectSpec(
        Reverb,
        {
            "room_size": ParamSpec(0.3, 0.0, 1.0),
            "wet_level": ParamSpec(0.3, 0.0, 1.0),
            "damping": ParamSpec(0.6, 0.0, 1.0),
            "dry_level": ParamSpec(0.8, 0.0, 1.0),
            "width": ParamSpec(1.0, 0.0, 1.0),
        },
    ),
    "highpass": EffectSpec(
        HighpassFilter,
        {"cutoff_frequency_hz": ParamSpec(800.0, 1.0, 96000.0)},
    ),
    "lowpass": EffectSpec(
        LowpassFilter,
        {"cutoff_frequency_hz": ParamSpec(3000.0, 1.0, 96000.0)},
    ),
    "pitch_shift": EffectSpec(
        PitchShift,
        {"semitones": ParamSpec(0.0, -12.0, 12.0)},
    ),
    "distortion": EffectSpec(
        Distortion,
        {"drive_db": ParamSpec(10.0, 0.0, 30.0)},
    ),
    "bitcrush": EffectSpec(
        Bitcrush,
        {"bit_depth": ParamSpec(8, 4, 16, is_int=True)},
    ),
    "delay": EffectSpec(
        Delay,
        {
            "delay_seconds": ParamSpec(0.3, 0.05, 1.0),
            "feedback": ParamSpec(0.3, 0.0, 1.0),
            "mix": ParamSpec(0.5, 0.0, 1.0),
        },
    ),
    "chorus": EffectSpec(
        Chorus,
        {
            "rate_hz": ParamSpec(1.5, 0.1, 5.0),
            "depth": ParamSpec(0.5, 0.0, 1.0),
            "mix": ParamSpec(0.3, 0.0, 1.0),
        },
    ),
    "gain": EffectSpec(
        Gain,
        {"gain_db": ParamSpec(0.0, -20.0, 20.0)},
    ),
    "phaser": EffectSpec(
        Phaser,
        {
            "rate_hz": ParamSpec(1.0, 0.1, 5.0),
            "depth": ParamSpec(0.5, 0.0, 1.0),
            "feedback": ParamSpec(0.0, -1.0, 1.0),
            "mix": ParamSpec(0.5, 0.0, 1.0),
        },
    ),
    "compressor": EffectSpec(
        Compressor,
        {
            "threshold_db": ParamSpec(-20.0, -120.0, 0.0),
            "ratio": ParamSpec(4.0, 1.0, 100.0),
            "attack_ms": ParamSpec(1.0, 0.01, 1000.0),
            "release_ms": ParamSpec(100.0, 0.01, 10000.0),
        },
    ),
    "clipping": EffectSpec(
        Clipping,
        {"threshold_db": ParamSpec(-6.0, -120.0, 0.0)},
    ),
    "noise_gate": EffectSpec(
        NoiseGate,
        {
            "threshold_db": ParamSpec(-40.0, -120.0, 0.0),
            "ratio": ParamSpec(10.0, 1.0, 100.0),
            "attack_ms": ParamSpec(1.0, 0.01, 1000.0),
            "release_ms": ParamSpec(100.0, 0.01, 10000.0),
        },
    ),
    "ladder_filter": EffectSpec(
        LadderFilter,
        {
            "cutoff_hz": ParamSpec(1000.0, 1.0, 96000.0),
            "resonance": ParamSpec(0.0, 0.0, 1.0),
            "drive": ParamSpec(1.0, 1.0, 100.0),
        },
        extra_kwargs={"mode": LadderFilter.Mode.LPF12},
    ),
    "resample": EffectSpec(
        Resample,
        {"target_sample_rate": ParamSpec(8000.0, 1000.0, 192000.0)},
    ),
    "gsm": EffectSpec(GSMFullRateCompressor, {}),
    "mp3": EffectSpec(
        MP3Compressor,
        {"vbr_quality": ParamSpec(5.0, 0.0, 9.0)},
    ),
}

# LLM 与配置侧使用的简短参数别名 → pedalboard 实际参数名。
_PARAM_ALIASES: Final[dict[str, str]] = {"cutoff_hz": "cutoff_frequency_hz"}


def _normalize_params(effect_type: str, effect: dict[str, Any]) -> dict[str, Any]:
    """把效果器描述中的参数键规整为 pedalboard 参数名。

    同时兼容两种输入形态：参数平铺在效果器对象里（LLM 调用），
    或收在 ``params`` 子字典里（配置文件）。

    Args:
        effect_type: 效果器类型名。
        effect: 单个效果器描述字典。

    Returns:
        规整后的参数字典。
    """
    raw = effect.get("params")
    if not isinstance(raw, dict):
        raw = {key: value for key, value in effect.items() if key != "type"}
    spec = EFFECT_SPECS[effect_type]
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        name = _PARAM_ALIASES.get(key, key)
        # 别名只在目标效果器确实拥有该参数时生效，避免误改同名参数。
        if name not in spec.params and key in spec.params:
            name = key
        normalized[name] = value
    return normalized


def build_effect_chain(effects: list[dict[str, Any]]) -> list[Any]:
    """按顺序构建 pedalboard 效果器实例列表。

    Args:
        effects: 效果器描述列表，每项须含 ``type``。

    Returns:
        pedalboard 效果器实例列表。

    Raises:
        ValueError: 效果器类型未知，或参数名/取值非法。
    """
    chain: list[Any] = []
    for effect in effects:
        effect_type = str(effect.get("type", "")).strip()
        spec = EFFECT_SPECS.get(effect_type)
        if spec is None:
            raise ValueError(f"未知的效果器类型: {effect_type or '(空)'}")
        chain.append(spec.build(effect_type, _normalize_params(effect_type, effect)))
    return chain


def _render(audio_data: bytes, board_effects: list[Any], has_tail: bool) -> bytes:
    """在当前线程同步渲染效果链。

    Args:
        audio_data: 原始 WAV 字节。
        board_effects: 已构建的效果器实例列表。
        has_tail: 效果链中是否存在会产生尾音的效果器。

    Returns:
        渲染后的 WAV 字节。
    """
    with io.BytesIO(audio_data) as audio_stream, AudioFile(audio_stream, "r") as source:
        board = Pedalboard(board_effects)
        raw = source.read(source.frames)
        sample_rate = int(source.samplerate)

    if has_tail:
        # pedalboard 输出长度等于输入长度，尾音会被截断；
        # 先补一段静音让尾音渲染完整，再按音量阈值裁掉纯静音部分。
        padding = np.zeros(
            (raw.shape[0], int(sample_rate * _TAIL_PADDING_SECONDS)), dtype=raw.dtype
        )
        rendered = board(np.concatenate([raw, padding], axis=1), sample_rate)
        envelope = (
            np.abs(rendered).max(axis=0) if rendered.ndim > 1 else np.abs(rendered)
        )
        voiced = np.where(envelope > _TAIL_SILENCE_THRESHOLD)[0]
        if len(voiced) > 0:
            end = min(
                voiced[-1] + int(sample_rate * _TAIL_KEEP_SECONDS), len(envelope)
            )
            rendered = rendered[:end]
    else:
        rendered = board(raw, sample_rate)

    with io.BytesIO() as output:
        sf.write(output, rendered.T, sample_rate, format="WAV")
        return output.getvalue()


async def apply_effects(audio_data: bytes, effects: list[dict[str, Any]]) -> bytes:
    """按效果链顺序处理音频。

    渲染在线程池中执行，避免阻塞事件循环。

    Args:
        audio_data: 原始 WAV 字节。
        effects: 效果器描述列表。

    Returns:
        处理后的 WAV 字节；效果链为空时原样返回。

    Raises:
        ValueError: 效果器类型或参数非法。
    """
    if not effects:
        return audio_data

    board_effects = build_effect_chain(effects)
    has_tail = any(
        str(effect.get("type", "")) in _TAIL_EFFECT_TYPES for effect in effects
    )
    processed = await asyncio.to_thread(_render, audio_data, board_effects, has_tail)
    logger.info(f"已应用音频效果器: {[effect.get('type') for effect in effects]}")
    return processed


__all__ = [
    "EFFECT_SPECS",
    "EffectSpec",
    "ParamSpec",
    "apply_effects",
    "build_effect_chain",
]
