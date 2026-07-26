"""TTS Voice 配置 TOML 持久化的框架兼容边界。

.. warning::

    本模块是整个插件唯一允许触碰框架内部实现的位置。

    ``src.app.plugin_system.api.config_api`` 目前只提供 ``load_config`` /
    ``reload_config``，没有"保存一个已在内存中校验通过的配置实例"的能力，
    而 WebUI 的配置编辑功能必须写回 TOML。因此这里直接调用
    ``src.kernel.config.core._render_toml_with_signature`` 复用框架的
    带注释渲染逻辑，以保证写出的文件与框架自动生成的格式完全一致。

    等公开配置 API 提供保存能力后，应删除本模块并改用公开接口。
    在此之前，请不要把框架私有导入扩散到其它模块。
"""

from __future__ import annotations

import os
import tomllib
import uuid
from pathlib import Path
from typing import cast

from src.kernel.config.core import ConfigBase, _render_toml_with_signature

from .config import TTSVoiceConfig


def render_config(config: TTSVoiceConfig) -> str:
    """把已校验配置渲染为带注释的 TOML 文本。

    Args:
        config: 已通过校验的配置实例。

    Returns:
        TOML 文本。
    """
    return _render_toml_with_signature(
        cast(type[ConfigBase], TTSVoiceConfig),
        config.model_dump(mode="python"),
    )


def save_config_atomically(path: Path, config: TTSVoiceConfig) -> None:
    """把配置写入临时文件，解析校验后原子替换目标文件。

    Args:
        path: 目标配置文件路径。
        config: 待写入的配置实例。

    Raises:
        tomllib.TOMLDecodeError: 渲染结果不是合法 TOML。
        OSError: 写入或替换失败。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(render_config(config), encoding="utf-8")
        with temp_path.open("rb") as stream:
            tomllib.load(stream)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


__all__ = ["render_config", "save_config_atomically"]
