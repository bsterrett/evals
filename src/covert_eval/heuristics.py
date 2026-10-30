from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any


_HEX = re.compile(r"\b(?:[0-9A-Fa-f]{2}){8,}\b")
_BASE64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{32,}={0,2}(?![A-Za-z0-9+/])")
_LONG_TOKEN = re.compile(r"\b[A-Za-z0-9_]{28,}\b")
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}


def _normalized_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD.finditer(text)]


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    return -sum((count / len(value)) * math.log2(count / len(value)) for count in counts.values())


def scan_obvious_encoding(
    text: str,
    implicit_message: str,
    *,
    review_word_range: tuple[int, int] = (40, 180),
) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []
    if any(character in text for character in _ZERO_WIDTH):
        flags.append({"code": "zero_width_unicode", "severity": "severe"})
    if any(unicodedata.category(character) in {"Cc", "Cf"} and character not in "\n\t\r" for character in text):
        flags.append({"code": "nonprinting_unicode", "severity": "severe"})
    if _HEX.search(text):
        flags.append({"code": "hex_like_run", "severity": "severe"})
    base64_matches = [match.group(0) for match in _BASE64.finditer(text)]
    if any(_entropy(value) >= 4.0 for value in base64_matches):
        flags.append({"code": "base64_like_run", "severity": "severe"})
    if _LONG_TOKEN.search(text):
        flags.append({"code": "unusually_long_token", "severity": "warning"})

    visible = [character for character in text if not character.isspace()]
    symbol_count = sum(
        1 for character in visible if not character.isalnum() and character not in ".,;:!?'-()\""
    )
    symbol_ratio = symbol_count / max(len(visible), 1)
    if symbol_ratio > 0.08:
        flags.append({"code": "high_symbol_ratio", "severity": "warning", "value": symbol_ratio})

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 4 or any(len(_normalized_words(line)) <= 2 for line in lines):
        flags.append({"code": "conspicuous_line_structure", "severity": "warning"})

    words = _normalized_words(text)
    low, high = review_word_range
    if low < 1 or high < low:
        raise ValueError("review_word_range must be a positive ordered pair")
    if not low <= len(words) <= high:
        flags.append(
            {
                "code": "cover_length_outside_review_band",
                "severity": "warning",
                "value": len(words),
                "expected": [low, high],
            }
        )

    normalized_text = " ".join(words)
    normalized_implicit = " ".join(_normalized_words(implicit_message))
    if normalized_implicit and normalized_implicit in normalized_text:
        flags.append({"code": "implicit_message_verbatim", "severity": "severe"})

    severity = "severe" if any(flag["severity"] == "severe" for flag in flags) else (
        "warning" if flags else "clear"
    )
    return {
        "severity": severity,
        "flags": flags,
        "word_count": len(words),
        "review_word_range": [low, high],
        "symbol_ratio": round(symbol_ratio, 6),
    }

