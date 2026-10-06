"""Text and identifier normalisation shared by ingestion, dedup and search."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional

_VERSION_RE = re.compile(r"v\d+$")
_INLINE_MATH_RE = re.compile(r"\$\$?([^$]*)\$\$?")
_LATEX_CMD_ARG_RE = re.compile(r"\\(?:emph|textbf|textit|texttt|mathrm|mathbf|mathcal|text|textsc|url|cite|ref)\s*\{([^{}]*)\}")
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+\*?")
_BRACES_RE = re.compile(r"[{}]")
_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s]")

# Frequent English function words. A real English abstract is ~25-45% these;
# other languages score near zero, so a low threshold is a cheap, dependency-
# free language filter.
_EN_STOPWORDS = frozenset("""
a an the of and or in on for to with by from as at is are was were be been this that
these those we our it its which can using based than into over under between such via
not also show results method methods propose proposed approach paper model models
""".split())


def base_arxiv_id(raw_id: str) -> str:
    """``2501.01234v3`` / ``http://arxiv.org/abs/cs/0101001v1`` -> bare id."""
    s = str(raw_id or "").strip()
    if "arxiv.org/abs/" in s:
        s = s.split("arxiv.org/abs/", 1)[1]
    return _VERSION_RE.sub("", s)


def arxiv_version(raw_id: str) -> int:
    m = re.search(r"v(\d+)$", str(raw_id or "").strip())
    return int(m.group(1)) if m else 1


def strip_latex(text: str) -> str:
    text = _INLINE_MATH_RE.sub(lambda m: " " + m.group(1) + " ", text)
    for _ in range(3):  # unwrap nested \cmd{...}
        new = _LATEX_CMD_ARG_RE.sub(r"\1", text)
        if new == text:
            break
        text = new
    text = _LATEX_CMD_RE.sub(" ", text)
    text = _BRACES_RE.sub("", text)
    return text.replace("~", " ").replace("\\", " ")


def normalize_text(text: Optional[str]) -> str:
    """Unicode NFKC, LaTeX markup stripped, whitespace collapsed."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = strip_latex(s)
    return _WS_RE.sub(" ", s).strip()


def title_key(title: str) -> str:
    """Normalised title for de-duplication: lowercase, punctuation stripped."""
    s = unicodedata.normalize("NFKD", normalize_text(title)).encode("ascii", "ignore").decode()
    s = _NON_WORD_RE.sub(" ", s.lower())
    return _WS_RE.sub(" ", s).strip()


def parse_categories(categories: str | Iterable[str]) -> List[str]:
    if isinstance(categories, str):
        return [c for c in categories.split() if c]
    return [str(c) for c in categories if c]


def has_category(categories: str | Iterable[str], prefix: str = "cs.",
                 whitelist: Optional[Iterable[str]] = None) -> bool:
    """True when any listed category matches. Checks each category token, so
    ``physics.comp-ph`` never matches the ``cs.`` prefix by substring."""
    cats = parse_categories(categories)
    if whitelist is not None:
        allowed = set(whitelist)
        return any(c in allowed for c in cats)
    return any(c.startswith(prefix) for c in cats)


def english_score(text: str) -> float:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in _EN_STOPWORDS) / len(words)


def is_english(text: str, threshold: float = 0.08) -> bool:
    if not text:
        return False
    ascii_share = sum(1 for ch in text if ord(ch) < 128) / len(text)
    if ascii_share < 0.8:
        return False
    return english_score(text) >= threshold
