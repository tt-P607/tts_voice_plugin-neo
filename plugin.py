"""TTS Voice 插件入口。

基于 GPT-SoVITS 的文本转语音插件，支持多种语言和多风格语音合成。
"""

from __future__ import annotations

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import BasePlugin, register_plugin

from typing import cast

from src.app.plugin_system.api.service_api import get_service

from .actions.tts_action import TTSVoiceAction
from .commands.tts_command import TTSVoiceCommand
from .config import TTSVoiceConfig
from .services.tts_service import TTSService

logger = get_logger("tts_voice_plugin-neo")


@register_plugin
class TTSVoicePlugin(BasePlugin):
    """GPT-SoVITS 语音合成插件。"""

    plugin_name: str = "tts_voice_plugin-neo"
    plugin_description: str = "基于GPT-SoVITS的文本转语音插件，支持多种语言和多风格语音合成"
    plugin_version: str = "3.1.2"

    configs = [TTSVoiceConfig]
    dependent_components = ["tts_http_server:service:tts_provider_registry"]

    def __init__(self, config: TTSVoiceConfig | None = None) -> None:
        """初始化插件。

        Args:
            config: 插件配置实例
        """
        super().__init__(config)
        self.tts_service: TTSService | None = None

    async def on_plugin_loaded(self) -> None:
        """插件加载后回调，初始化 TTS 服务。"""
        logger.info("初始化 TTSVoicePlugin...")
        self.tts_service = TTSService(self)
        logger.info("TTSService 已成功初始化。")

        # 注册为 TTS Provider (适配语音通话插件组)
        try:
            from .provider import TTSVoiceProvider
            registry = get_service("tts_http_server:service:tts_provider_registry")
            if registry:
                provider = TTSVoiceProvider(self.tts_service)
                # 使用 cast 避免类型检查错误，或者直接调用
                getattr(registry, "register_provider")(provider, default=True)
                logger.info("已成功将 tts_voice_plugin-neo 注册为 TTS Provider")
        except Exception as e:
            logger.warning(f"注册 TTS Provider 失败 (可能未安装 calling 插件组): {e}")

        # 将自定义场景说明追加到 action 的描述，使 Chatter 侧感知使用时机
        if isinstance(self.config, TTSVoiceConfig):
            custom = self.config.prompt.custom_instructions.strip()
            if custom:
                TTSVoiceAction.action_description = (
                    TTSVoiceAction.action_description.rstrip() + "\n\n自定义指令：\n" + custom
                )
                logger.debug("已将自定义场景说明追加到 tts_voice_action 描述")

    def get_components(self) -> list[type]:
        """返回插件内所有组件类。

        根据配置判断是否启用 Action 和 Command 组件。

        Returns:
            组件类列表
        """
        components: list[type] = [TTSService]

        cfg: TTSVoiceConfig | None = self.config  # type: ignore[assignment]

        action_enabled = True
        command_enabled = True
        if cfg is not None:
            action_enabled = cfg.components.action_enabled
            command_enabled = cfg.components.command_enabled

        if action_enabled:
            components.append(TTSVoiceAction)
        if command_enabled:
            components.append(TTSVoiceCommand)

        return components
