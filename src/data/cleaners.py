from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional, Tuple

_INLINE_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class CleanTextConfig:
    min_chars: int = 50
    max_chars: int = 12000
    drop_empty: bool = True
    normalize_whitespace: bool = True


def stable_text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_text_whitespace(text: str, language: Optional[str] = None) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")

    if language == "code":
        lines = [re.sub(r"[ \t]+$", "", line) for line in text.split("\n")]
        text = "\n".join(lines)
        text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text)
        return text.strip("\n")

    lines = []
    for line in text.split("\n"):
        normalized = _INLINE_WHITESPACE_RE.sub(" ", line).strip()
        lines.append(normalized)

    text = "\n".join(lines)
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def clean_text(
    text: object,
    config: CleanTextConfig,
    language: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    if text is None:
        return None, "missing_text"

    if not isinstance(text, str):
        text = str(text)

    cleaned = normalize_text_whitespace(text, language) if config.normalize_whitespace else text.strip()

    if config.drop_empty and not cleaned.strip():
        return None, "empty"

    char_count = len(cleaned)
    if char_count < config.min_chars:
        return None, "too_short"
    if char_count > config.max_chars:
        return None, "too_long"

    return cleaned, None
