"""TTS Voice 插件的跨组件结构协议与兼容数据对象。

本模块只描述运行时需要的最小形状：对外用于对接 TTS Registry，
对内用于让 Action / Command / Router 以结构类型访问插件实例，
避免与 ``plugin`` 模块相互导入。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .config import TTSVoiceConfig
    from .services.tts_service import TTSService


@runtime_checkable
class SynthesisRequestLike(Protocol):
    """TTS Provider 接收的规范化请求最小协议。"""

    text: str
    markers: dict[str, Any]
    options: dict[str, Any]


@dataclass(slots=True)
class SynthesisResponse:
    """符合 TTS HTTP Registry 结构约定的合成响应。"""

    audio_base64: str
    mime_type: str = "audio/wav"
    format: str = "wav"
    sample_rate: int | None = None
    duration_ms: int | None = None
    provider: str = ""
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PCMStream:
    """GPT-SoVITS V5 返回的原序 PCM 流及其格式元数据。"""

    sample_rate: int
    channels: int
    sample_format: str
    chunks: AsyncIterator[bytes]


@dataclass(slots=True)
class ParameterGuide:
    """向上游描述一个可选 TTS 参数。"""

    description: str
    param_type: str = "string"
    default: Any = None
    required: bool = False
    valid_values: list[str] | None = None
    min_value: float | None = None
    max_value: float | None = None


@dataclass(slots=True)
class ProviderCapabilities:
    """TTS Provider 向消费方暴露的参数能力。"""

    style_guide: ParameterGuide | None = None
    language_guide: ParameterGuide | None = None
    speed_guide: ParameterGuide | None = None
    effects_guide: ParameterGuide | None = None
    aux_refer_wav_paths_guide: ParameterGuide | None = None


@runtime_checkable
class ProviderRegistryLike(Protocol):
    """TTS Provider Registry Service 的最小调用协议。"""

    def register_provider(self, provider: Any, *, default: bool = False) -> None:
        """注册一个 Provider。"""

        ...

    def unregister_provider(self, provider_name: str) -> bool:
        """注销指定 Provider。"""

        ...


class TTSPluginLike(Protocol):
    """组件访问插件实例时依赖的最小接口。"""

    config: "TTSVoiceConfig"
    tts_service: "TTSService | None"

    def register_command_task(self, task_id: str) -> None:
        """登记 Command 创建的后台任务。"""

        ...

    def discard_command_task(self, task_id: str) -> None:
        """移除已完成的 Command 后台任务登记。"""

        ...

    def refresh_action_description(self) -> None:
        """按当前风格与配置重建 Action 描述。"""

        ...


__all__ = [
    "PCMStream",
    "ParameterGuide",
    "ProviderCapabilities",
    "ProviderRegistryLike",
    "SynthesisRequestLike",
    "SynthesisResponse",
    "TTSPluginLike",
]
