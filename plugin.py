"""TTS Voice 插件入口与资源生命周期管理。"""

from __future__ import annotations

from typing import Any, ClassVar, cast

from src.app.plugin_system.api import prompt_api
from src.app.plugin_system.api.event_api import EventDecision
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.service_api import get_service
from src.app.plugin_system.base import BaseEventHandler, BasePlugin, register_plugin
from src.app.plugin_system.types import EventType, SystemReminderBucket
from src.kernel.concurrency import get_task_manager

from . import prompts
from .actions.tts_action import TTSVoiceAction
from .commands.tts_command import TTSVoiceCommand
from .config import TTSVoiceConfig
from .protocol import ProviderRegistryLike
from .provider import TTSVoiceProvider
from .router import TTSVoiceWebUIRouter
from .services.tts_service import TTSService

logger = get_logger("tts_voice_plugin-neo")

_PROVIDER_REGISTRY_SERVICE = "tts_http_server:service:tts_provider_registry"
_RULE_REMINDER_NAME = "tts_voice_rules"


class TTSVoiceRuleReminderHandler(BaseEventHandler):
    """在提示词构建时同步当前聊天流的语音规则提醒。"""

    name: str = "tts_voice_rule_reminder"
    description: str = "按配置向对话末尾注入语音表达规则"
    init_subscribe: ClassVar[list[EventType | str]] = [EventType.ON_PROMPT_BUILD]

    async def execute(
        self, event_name: str, params: dict[str, Any]
    ) -> tuple[EventDecision, dict[str, Any]]:
        """写入流隔离提醒并保持事件参数不变。

        Args:
            event_name: 事件名称。
            params: 提示词构建参数，values 中可携带 stream_id。

        Returns:
            成功决策及原始事件参数。
        """
        values = params.get("values")
        stream_id = values.get("stream_id") if isinstance(values, dict) else None
        if isinstance(stream_id, str) and stream_id.strip():
            cast(TTSVoicePlugin, self.plugin).sync_rule_reminder(stream_id.strip())
        return EventDecision.SUCCESS, params


@register_plugin
class TTSVoicePlugin(BasePlugin):
    """装配 GPT-SoVITS Service、Provider、Action、Command 与 WebUI。"""

    plugin_name: str = "tts_voice_plugin-neo"

    configs: list[type] = [TTSVoiceConfig]
    dependent_components: list[str] = [_PROVIDER_REGISTRY_SERVICE]

    def __init__(self, config: TTSVoiceConfig | None = None) -> None:
        """初始化插件持有的运行时资源。

        Args:
            config: 插件配置；为空时由框架注入。
        """
        super().__init__(config)
        self.tts_service: TTSService | None = None
        self._provider_registered = False
        self._command_task_ids: set[str] = set()
        self._rule_reminder_streams: set[str] = set()

    @property
    def tts_config(self) -> TTSVoiceConfig:
        """当前插件配置。"""
        return cast(TTSVoiceConfig, self.config)

    def register_command_task(self, task_id: str) -> None:
        """登记 Command 创建的后台任务。

        Args:
            task_id: TaskManager 分配的任务 ID。
        """
        self._command_task_ids.add(task_id)

    def discard_command_task(self, task_id: str) -> None:
        """移除已完成的 Command 后台任务登记。

        Args:
            task_id: TaskManager 分配的任务 ID。
        """
        self._command_task_ids.discard(task_id)

    @staticmethod
    def _get_provider_registry() -> ProviderRegistryLike | None:
        """从公开 Service API 获取 TTS Provider Registry。

        Returns:
            Registry Service 实例；服务不存在时返回 ``None``。

        Raises:
            TypeError: 服务实例不符合预期协议。
        """
        service = get_service(_PROVIDER_REGISTRY_SERVICE)
        if service is None:
            return None
        if not isinstance(service, ProviderRegistryLike):
            raise TypeError("TTS Provider Registry Service 不符合预期协议")
        return service

    def refresh_action_description(self) -> None:
        """按当前可用风格与自定义说明刷新 Action 描述和语音规则提醒。"""
        styles = self.tts_service.get_available_styles() if self.tts_service else []
        TTSVoiceAction.description = prompts.build_action_description(
            styles, self.tts_config.prompt.custom_instructions
        )
        for stream_id in tuple(self._rule_reminder_streams):
            self.sync_rule_reminder(stream_id)

    def sync_rule_reminder(self, stream_id: str) -> None:
        """按当前开关同步单个聊天流的动态语音规则提醒。

        Args:
            stream_id: 提示词所属的聊天流 ID。
        """
        config = self.tts_config
        if not config.plugin.enable or not config.prompt.inject_rule_reminder:
            prompt_api.delete_stream_reminder(
                stream_id, SystemReminderBucket.ACTOR.value, _RULE_REMINDER_NAME
            )
            self._rule_reminder_streams.discard(stream_id)
            return

        styles = self.tts_service.get_available_styles() if self.tts_service else []
        rules = [
            (
                "【语音表达规则】以下规则仅在准备语音合成时适用，不要求每轮都发送语音；"
                "普通文字回复保持自然写法，不必改写为读音文本。"
            ),
            prompts.build_action_description(styles, config.prompt.custom_instructions),
            prompts.TTS_SEGMENTS_HINT,
        ]
        if config.plugin.llm_speed_control:
            rules.append(prompts.SPEED_FACTOR_HINT)
        prompt_api.add_stream_reminder(
            stream_id=stream_id,
            bucket=SystemReminderBucket.ACTOR.value,
            name=_RULE_REMINDER_NAME,
            content="\n\n".join(rules),
            insert_type=prompt_api.SystemReminderInsertType.DYNAMIC,
            consume=prompt_api.SystemReminderConsumeType.FOREVER,
        )
        self._rule_reminder_streams.add(stream_id)

    def _apply_action_exposure(self) -> None:
        """按配置开关决定 Action 向模型暴露哪些可选参数。"""
        exposed: set[str] = set()
        plugin_config = self.tts_config.plugin
        if plugin_config.llm_speed_control:
            exposed.add("speed_factor")
        if plugin_config.llm_audio_effects:
            exposed.add("audio_effects")
        if plugin_config.llm_speed_control and plugin_config.llm_audio_effects:
            exposed.update({"aux_refer_wav_paths", "merge_voice", "pause_duration"})
        TTSVoiceAction.exposed_optional_params = frozenset(exposed)

    async def on_plugin_loaded(self) -> None:
        """初始化 Service 并注册 TTS Provider。

        插件可独立加载；缺少 ``tts_http_server`` 的 Provider Registry Service 时，
        仅跳过 Provider 注册，其余能力不受影响。
        """
        if not self.tts_config.plugin.enable:
            logger.info("TTS Voice 插件已通过配置关闭")
            return

        self.tts_service = TTSService(self)
        registry = self._get_provider_registry()
        if registry is None:
            logger.warning("未发现 tts_http_server Provider Registry Service，跳过 Provider 注册")
        else:
            registry.register_provider(TTSVoiceProvider(self.tts_service), default=True)
            self._provider_registered = True
        self.refresh_action_description()
        logger.info("TTS Voice Plugin 已初始化并注册 Provider")

    async def on_plugin_unloaded(self) -> None:
        """取消后台任务、注销 Provider 并关闭网络资源。"""
        for stream_id in self._rule_reminder_streams:
            prompt_api.delete_stream_reminder(
                stream_id, SystemReminderBucket.ACTOR.value, _RULE_REMINDER_NAME
            )
        self._rule_reminder_streams.clear()
        task_manager = get_task_manager()
        for task_id in tuple(self._command_task_ids):
            task_manager.cancel_task(task_id)
        self._command_task_ids.clear()

        if self._provider_registered:
            try:
                registry = self._get_provider_registry()
                if registry is not None:
                    registry.unregister_provider(TTSVoiceProvider.provider_name)
            except (TypeError, RuntimeError) as error:
                logger.warning(f"注销 TTS Provider 失败: {error}")
            finally:
                self._provider_registered = False

        if self.tts_service is not None:
            await self.tts_service.close()
            self.tts_service = None
        logger.info("TTS Voice Plugin 运行时资源已清理")

    def get_components(self) -> list[type]:
        """按配置返回插件组件类。

        Returns:
            需要注册的组件类列表；插件关闭时为空。
        """
        if not self.tts_config.plugin.enable:
            return []

        components: list[type] = [
            TTSService, TTSVoiceWebUIRouter, TTSVoiceRuleReminderHandler
        ]
        if self.tts_config.components.action_enabled:
            self._apply_action_exposure()
            components.append(TTSVoiceAction)
        if self.tts_config.components.command_enabled:
            components.append(TTSVoiceCommand)
        return components


__all__ = ["TTSVoicePlugin"]
