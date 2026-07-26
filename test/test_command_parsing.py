"""/tts 命令参数解析测试。"""

from __future__ import annotations

import pytest

from tts_voice_plugin_neo.commands.tts_command import TTSVoiceCommand  # noqa: E402


_STYLES = {"default", "calm", "晚安"}


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        # 只有正文
        (("你好", "世界"), ("你好 世界", "default", "zh")),
        # 正文 + 风格
        (("你好", "calm"), ("你好", "calm", "zh")),
        # 正文 + 语言
        (("你好", "ja"), ("你好", "default", "ja")),
        # 正文 + 风格 + 语言
        (("你好", "calm", "纯日文"), ("你好", "calm", "all_ja")),
        # 空串占位不应影响解析
        (("你好", "", "", "calm", ""), ("你好", "calm", "zh")),
        # 完全没有参数
        ((), ("", "default", "zh")),
    ],
)
def test_parse_words(words: tuple[str, ...], expected: tuple[str, str, str]) -> None:
    """尾部的风格与语言参数应被正确剥离。"""

    assert TTSVoiceCommand._parse_words(words, _STYLES) == expected


def test_parse_words_keeps_text_that_looks_like_style() -> None:
    """风格名同时是正文最后一个词时，仍按风格剥离（用户可用引号规避）。"""

    text, style, language = TTSVoiceCommand._parse_words(("说点", "晚安"), _STYLES)
    assert (text, style, language) == ("说点", "晚安", "zh")


def test_parse_words_does_not_swallow_normal_text() -> None:
    """普通词不应被模糊匹配成语言代码而丢失。"""

    text, style, language = TTSVoiceCommand._parse_words(("今天", "天气", "真好"), _STYLES)
    assert text == "今天 天气 真好"
    assert style == "default"
    assert language == "zh"
