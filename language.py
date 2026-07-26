"""TTS language 参数说明与 GPT-SoVITS 语言代码归一化工具。

本模块同时服务于 Action schema、Provider 能力说明和 Service 请求校验。
任意 language 字符串会按以下顺序规整为 GPT-SoVITS 能识别的合法代码：

1. 直接命中合法代码（zh / en / ja / yue / ...）
2. 形态归一（去掉括号说明、连字符 → 下划线、大小写）后再次命中
3. 别名表（如 ``chinese`` → ``zh``、``zh-cn`` → ``zh``、``jp`` → ``ja``）
4. ``difflib`` 模糊匹配（拼写错误兜底，如 ``yuee`` → ``yue``）
5. 兜底回退到 ``zh``
"""

from __future__ import annotations

import difflib
from typing import Final


LANGUAGE_HELP_TEXT: str = (
    "语音合成的语言模式，根据文本内容选择。**必须严格从以下列表中选择，不能使用未列出的代码。**\n"
    "只填代码本身，不填括号内的说明文字。\n"
    "混合模式（文本中包含多种语言或外来词时选此类）：\n"
    "  zh — 中文为主（夹杂英文）  en — 英文为主（夹杂其他语言）\n"
    "  ja — 日文为主（夹杂英文）  yue — 粤语（夹杂英文）\n"
    "  ko — 韩文（夹杂英文）      auto_yue — 自动识别（含粤语优先）\n"
    "纯语言模式（文本仅含单一语言时优先选此类，推理效果更好）：\n"
    "  all_zh — 纯中文  all_ja — 纯日文  all_yue — 纯粤语  all_ko — 纯韩文\n"
    "多语言准则：\n"
    "- **非必要不要用 auto**——自动识别准确率不如指定语言，且可能导致语码切换不自然。\n"
    "- **需要切换语言时拆分为多次调用**：每次调用只包含一种语言的文本，用对应的 language 参数。\n"
    "  例如想先说中文再说日语，拆成两次调用：第一次 language=all_zh + 中文文本，第二次 language=all_ja + 日语文本。\n"
    "- file 模式下多次调用的音频会自动拼成一条文件发送；voice 模式下会按顺序逐条播放。\n"
    "不填则沿用风格配置中的默认语言。"
)


# GSV 接受的合法 language 代码集合。
VALID_LANGUAGE_CODES: Final[frozenset[str]] = frozenset(
    {
        "zh",
        "en",
        "ja",
        "yue",
        "ko",
        "auto",
        "auto_yue",
        "all_zh",
        "all_ja",
        "all_yue",
        "all_ko",
        "zh_en",
    }
)

# 常见别名 / locale / 同义写法 → 合法代码。键统一使用 lower() + 下划线形式。
# 这里只放"高置信度"的别名；难以判断的拼写错误交给后续的 difflib 模糊匹配兜底。
_LANGUAGE_ALIASES: Final[dict[str, str]] = {
    # ---- 中文 ----
    "chinese": "zh",
    "中文": "zh",
    "mandarin": "zh",
    "putonghua": "zh",
    "cn": "zh",
    "cmn": "zh",
    "zho": "zh",
    "zh_cn": "zh",
    "zh_hans": "zh",
    "zh_hant": "zh",
    "zh_tw": "zh",
    "zh_hk": "zh",
    "all_chinese": "all_zh",
    "pure_chinese": "all_zh",
    "纯中文": "all_zh",
    # ---- 英文 ----
    "english": "en",
    "eng": "en",
    "en_us": "en",
    "en_gb": "en",
    "en_uk": "en",
    "英文": "en",
    "英语": "en",
    "all_en": "en",
    "all_english": "en",
    "pure_english": "en",
    "纯英文": "en",
    # ---- 日文 ----
    "japanese": "ja",
    "日文": "ja",
    "日语": "ja",
    "日本語": "ja",
    "jp": "ja",
    "jpn": "ja",
    "ja_jp": "ja",
    "all_japanese": "all_ja",
    "pure_japanese": "all_ja",
    "纯日文": "all_ja",
    # ---- 粤语 ----
    "cantonese": "yue",
    "粤语": "yue",
    "粵語": "yue",
    "广东话": "yue",
    "廣東話": "yue",
    "yue_cn": "yue",
    "yue_hk": "yue",
    "all_cantonese": "all_yue",
    "pure_cantonese": "all_yue",
    "纯粤语": "all_yue",
    # ---- 韩文 ----
    "korean": "ko",
    "韩文": "ko",
    "韩语": "ko",
    "韓文": "ko",
    "韓語": "ko",
    "한국어": "ko",
    "kr": "ko",
    "kor": "ko",
    "ko_kr": "ko",
    "all_korean": "all_ko",
    "pure_korean": "all_ko",
    "纯韩文": "all_ko",
    # ---- 中英混合 ----
    "zhen": "zh_en",
    "chinese_english": "zh_en",
    "中英": "zh_en",
    "中英混合": "zh_en",
    "mixed": "zh_en",
    "mixed_zh_en": "zh_en",
    # ---- 自动识别 ----
    "automatic": "auto",
    "autodetect": "auto",
    "auto_detect": "auto",
    "auto_yue_zh": "auto_yue",
    "autoyue": "auto_yue",
}


def _canonicalize(language_str: str) -> str:
    """把任意输入字符串归一为 ``lower + 下划线`` 的形态，并去掉括号说明。

    例如 ``"ZH-CN(中英混合)"`` → ``"zh_cn"``、``"All Japanese"`` → ``"all_japanese"``。

    Args:
        language_str: 原始输入字符串。

    Returns:
        归一后的字符串，可能仍然不是合法代码，需要后续匹配。
    """
    # 去掉括号注释 (含中英文括号)
    base = language_str.split("(")[0].split("（")[0]
    base = base.strip().lower()
    # 连字符 / 空白 → 下划线
    base = base.replace("-", "_").replace(" ", "_")
    # 折叠多个下划线为一个
    while "__" in base:
        base = base.replace("__", "_")
    return base.strip("_")


def normalize_language_code(language_str: str | None, *, default: str = "zh") -> tuple[str, str]:
    """把任意 language 输入规整成 GSV 合法代码，附带匹配途径以便日志记录。

    匹配顺序：
        1. 直接命中合法代码
        2. 归一形态命中合法代码
        3. 别名表命中
        4. ``difflib`` 模糊匹配（cutoff=0.6，阈值偏严以避免误匹配）
        5. 回退到 ``default``

    Args:
        language_str: 原始 language 字符串，可为 ``None`` 或空。
        default: 所有匹配都失败时的兜底代码，默认 ``"zh"``。

    Returns:
        ``(matched_code, match_kind)`` 元组。``match_kind`` 取值：

        - ``"empty"``：输入为空，返回默认值
        - ``"exact"``：直接命中
        - ``"normalized"``：归一后命中
        - ``"alias"``：通过别名映射命中
        - ``"fuzzy"``：通过模糊匹配命中
        - ``"fallback"``：所有匹配失败，回退到 ``default``
    """
    if not language_str or not str(language_str).strip():
        return default, "empty"

    raw = str(language_str).strip()

    # 1. 直接命中（保留原始的 split('(')[0].lower() 行为以兼容历史）
    direct = raw.split("(")[0].split("（")[0].strip().lower()
    if direct in VALID_LANGUAGE_CODES:
        return direct, "exact"

    # 2. 归一后再试一次
    canon = _canonicalize(raw)
    if canon in VALID_LANGUAGE_CODES:
        return canon, "normalized"

    # 3. 别名表
    if canon in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[canon], "alias"
    # 同时尝试不带 all_ 前缀变体（防止 LLM 写 ``all-zh-cn`` 这类）
    if canon.startswith("all_"):
        stripped = canon[4:]
        if stripped in _LANGUAGE_ALIASES:
            base_code = _LANGUAGE_ALIASES[stripped]
            # all_ + 基础代码若是合法的就用 all_ 形式
            all_form = f"all_{base_code}"
            if all_form in VALID_LANGUAGE_CODES:
                return all_form, "alias"

    # 4. 模糊匹配。同时把别名键纳入候选，命中后再映射到合法代码。
    candidates: list[str] = list(VALID_LANGUAGE_CODES) + list(_LANGUAGE_ALIASES.keys())
    matches = difflib.get_close_matches(canon, candidates, n=1, cutoff=0.6)
    if matches:
        hit = matches[0]
        if hit in VALID_LANGUAGE_CODES:
            return hit, "fuzzy"
        return _LANGUAGE_ALIASES[hit], "fuzzy"

    # 5. 兜底
    return default, "fallback"


def resolve_language_token(token: str) -> str | None:
    """严格判定一个词是否为语言代码，用于命令行参数解析。

    与 :func:`normalize_language_code` 的区别是不做模糊匹配、不做兜底——
    命令解析需要区分"这个词是语言参数"和"这个词是正文"，任何猜测都会
    把正文的最后一个词误吞掉。

    Args:
        token: 待判定的词。

    Returns:
        命中时返回合法语言代码，否则返回 ``None``。
    """
    canon = _canonicalize(token)
    if canon in VALID_LANGUAGE_CODES:
        return canon
    return _LANGUAGE_ALIASES.get(canon)


__all__ = [
    "LANGUAGE_HELP_TEXT",
    "VALID_LANGUAGE_CODES",
    "normalize_language_code",
    "resolve_language_token",
]
