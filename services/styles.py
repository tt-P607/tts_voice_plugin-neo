"""语音风格的运行时表示与装载。

配置中的 :class:`~..config.TTSStyle` 是面向 TOML 的声明式模型；
本模块把它转换为合成流程真正使用的强类型 :class:`StyleProfile`，
避免在服务层用裸 ``dict`` + 字符串键访问风格字段。
"""

from __future__ import annotations

from collections.abc import ItemsView
from dataclasses import dataclass, replace

from ..config import TTSVoiceConfig


@dataclass(frozen=True, slots=True)
class StyleProfile:
    """一个可直接用于合成请求的语音风格。"""

    name: str
    server_url: str
    refer_wav_path: str
    prompt_text: str
    prompt_language: str
    gpt_weights: str
    sovits_weights: str
    speed_factor: float
    text_language: str
    aux_refer_wav_paths: tuple[str, ...]

    def override(
        self,
        *,
        speed_factor: float | None = None,
        aux_refer_wav_paths: list[str] | None = None,
    ) -> "StyleProfile":
        """生成一个覆盖了调用级参数的新风格副本。

        Args:
            speed_factor: 覆盖语速；``None`` 表示沿用风格默认值。
            aux_refer_wav_paths: 覆盖辅助参考音频；``None`` 表示沿用风格默认值。

        Returns:
            应用了覆盖值的新 :class:`StyleProfile`。
        """
        changes: dict[str, object] = {}
        if speed_factor is not None:
            changes["speed_factor"] = speed_factor
        if aux_refer_wav_paths is not None:
            changes["aux_refer_wav_paths"] = tuple(aux_refer_wav_paths)
        if not changes:
            return self
        return replace(self, **changes)  # type: ignore[arg-type]


class StyleRegistry:
    """按名称索引的风格集合，保持配置声明顺序。"""

    def __init__(self, profiles: dict[str, StyleProfile]) -> None:
        """初始化风格注册表。

        Args:
            profiles: 风格名到风格档案的有序映射。
        """
        self._profiles = profiles

    @property
    def names(self) -> list[str]:
        """按声明顺序返回全部风格名称。"""
        return list(self._profiles)

    def items(self) -> ItemsView[str, StyleProfile]:
        """按声明顺序返回 ``(风格名, 风格档案)`` 视图。"""
        return self._profiles.items()

    def __contains__(self, name: object) -> bool:
        """判断风格名是否存在。"""
        return name in self._profiles

    def __len__(self) -> int:
        """返回风格数量。"""
        return len(self._profiles)

    def resolve(self, style_hint: str | None) -> StyleProfile:
        """按提示名解析风格，未命中时回退到第一个风格。

        Args:
            style_hint: 期望的风格名，可为空。

        Returns:
            命中的风格；提示名不存在时返回首个风格。

        Raises:
            LookupError: 注册表为空。
        """
        if not self._profiles:
            raise LookupError("没有任何可用的 TTS 风格配置")
        if style_hint and style_hint in self._profiles:
            return self._profiles[style_hint]
        return next(iter(self._profiles.values()))


def load_styles(config: TTSVoiceConfig) -> StyleRegistry:
    """从插件配置构建风格注册表。

    首个风格作为缺省基准：其它风格留空的提示文本与权重路径会回退到它的
    对应值。``refer_wav_path`` 在配置模型层已强制非空，无需回退。

    Args:
        config: 插件配置实例。

    Returns:
        构建好的 :class:`StyleRegistry`。

    Raises:
        ValueError: 风格列表为空或存在重名风格。
    """
    style_configs = config.tts_styles
    if not style_configs:
        raise ValueError("tts_styles 配置不能为空")

    base = style_configs[0]
    profiles: dict[str, StyleProfile] = {}
    for style_config in style_configs:
        name = style_config.style_name.strip()
        if name in profiles:
            raise ValueError(f"TTS 风格名称重复: {name}")
        profiles[name] = StyleProfile(
            name=name,
            server_url=config.tts.server,
            refer_wav_path=style_config.refer_wav_path,
            prompt_text=style_config.prompt_text or base.prompt_text,
            prompt_language=style_config.prompt_language,
            gpt_weights=style_config.gpt_weights or base.gpt_weights,
            sovits_weights=style_config.sovits_weights or base.sovits_weights,
            speed_factor=style_config.speed_factor,
            text_language=style_config.text_language,
            aux_refer_wav_paths=tuple(style_config.aux_refer_wav_paths),
        )
    return StyleRegistry(profiles)


__all__ = ["StyleProfile", "StyleRegistry", "load_styles"]
