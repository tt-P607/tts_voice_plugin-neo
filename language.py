"""tts_voice_plugin-neo 的 language 参数说明常量。

把"模型该怎么填 language 参数"的说明抽到这一处，被两个地方共享：

- :class:`plugins.tts_voice_plugin-neo.actions.tts_action.TTSVoiceAction` 的
  ``text_language`` 参数 description。
- :class:`plugins.tts_voice_plugin-neo.provider.TTSVoiceProvider` 的
  ``language_help``，由 :mod:`plugins.tts_http_server` 通过 ``/status`` 暴露给
  上层 chatter / action（例如 :mod:`plugins.anima_chatter`）。

这样未来调整说明只需要改这一个常量；上层 plugin 在生成 schema 时会自动同步。
"""

from __future__ import annotations


LANGUAGE_HELP_TEXT: str = (
    "语音合成的语言模式，根据文本内容选择。只填代码本身，不填括号内的说明文字。\n"
    "混合模式（文本中包含多种语言或外来词时选此类）：\n"
    "  zh — 中文为主（夹杂英文）  en — 英文为主（夹杂其他语言）\n"
    "  ja — 日文为主（夹杂英文）  yue — 粤语（夹杂英文）\n"
    "  ko — 韩文（夹杂英文）      auto — 自动识别多语种\n"
    "  auto_yue — 自动识别（含粤语优先）\n"
    "纯语言模式（文本仅含单一语言时优先选此类，推理效果更好）：\n"
    "  all_zh — 纯中文  all_ja — 纯日文  all_yue — 纯粤语  all_ko — 纯韩文\n"
    "不填则沿用风格配置中的默认语言。"
)


__all__ = ["LANGUAGE_HELP_TEXT"]
