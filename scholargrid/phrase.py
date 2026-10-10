"""Phrase-aware matching over titles and abstracts.

Text is normalised the same way for papers and queries: ASCII-folded,
lowercased, punctuation and hyphens become spaces, and simple plurals are
folded ("debts" -> "debt", "studies" -> "study"). A paper then matches a query
phrase at one of three levels (lower is better):

1. the phrase appears in the title,
2. the phrase appears in the abstract,
3. every content word of the phrase appears within ``window`` words.

Normalised text is computed once per corpus (stored in the bundle as
``phrase_text.parquet``) so matching is a substring scan that takes
milliseconds even for 50k papers.
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

PHRASE_PARQUET = "phrase_text.parquet"
LEVEL_TITLE, LEVEL_ABSTRACT, LEVEL_NEAR = 1, 2, 3
LEVEL_NAMES = {LEVEL_TITLE: "phrase in title", LEVEL_ABSTRACT: "phrase in abstract",
               LEVEL_NEAR: "all words nearby"}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(["a", "an", "the", "of", "and", "or", "in", "on", "for", "to", "with", "by", "from", "as", "at", "is", "are", "was", "were", "be", "been", "this", "that", "these", "those", "we", "our", "it", "its", "which", "via", "into", "using", "based", "towards", "toward", "vs", "versus"])


def singular(tok: str) -> str:
    if len(tok) <= 3 or tok.isdigit():
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith(("ss", "us", "is")):
        return tok
    if tok.endswith("s"):
        return tok[:-1]
    return tok


def tokens(text: Optional[str]) -> List[str]:
    if not text:
        return []
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return [singular(t) for t in _TOKEN_RE.findall(s)]


def normalize(text: Optional[str]) -> str:
    return " ".join(tokens(text))


def content_tokens(phrase: str) -> List[str]:
    return [t for t in phrase.split() if t not in STOPWORDS]


def _min_span(text_tokens: Sequence[str], wanted: Sequence[str]) -> int:
    """Smallest number of consecutive tokens containing every wanted token."""
    need = set(wanted)
    hits = [(i, t) for i, t in enumerate(text_tokens) if t in need]
    best = len(text_tokens) + 1
    counts: Dict[str, int] = {}
    left = 0
    for right in range(len(hits)):
        counts[hits[right][1]] = counts.get(hits[right][1], 0) + 1
        while len(counts) == len(need):
            best = min(best, hits[right][0] - hits[left][0] + 1)
            t = hits[left][1]
            counts[t] -= 1
            if counts[t] == 0:
                del counts[t]
            left += 1
    return best


class PhraseIndex:
    def __init__(self, titles: Iterable[str], abstracts: Iterable[str], normalized: bool = False):
        fn = (lambda s: str(s or "")) if normalized else normalize
        self.titles: List[str] = [f" {fn(t)} " for t in titles]
        self.abstracts: List[str] = [f" {fn(a)} " for a in abstracts]

    def __len__(self) -> int:
        return len(self.titles)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> PhraseIndex:
        return cls(df["title"].astype(str), df["abstract"].fillna("").astype(str))

    def append(self, titles: Iterable[str], abstracts: Iterable[str]) -> None:
        self.titles.extend(f" {normalize(t)} " for t in titles)
        self.abstracts.extend(f" {normalize(a)} " for a in abstracts)

    def match(self, phrases: Sequence[str], mask: Optional[np.ndarray] = None,
              window: int = 8) -> Dict[int, int]:
        """``{row: level}`` for every row matching any of the (normalised) phrases."""
        levels: Dict[int, int] = {}
        allowed = (lambda i: True) if mask is None else (lambda i: bool(mask[i]))
        for phrase in dict.fromkeys(p for p in phrases if p):
            pp = f" {phrase} "
            for i, t in enumerate(self.titles):
                if pp in t and allowed(i):
                    levels[i] = LEVEL_TITLE
            for i, a in enumerate(self.abstracts):
                if pp in a and levels.get(i, 9) > LEVEL_ABSTRACT and allowed(i):
                    levels[i] = LEVEL_ABSTRACT
            words = list(dict.fromkeys(content_tokens(phrase)))
            if len(words) < 2:
                continue
            padded = [f" {w} " for w in words]
            for i, (t, a) in enumerate(zip(self.titles, self.abstracts)):
                if i in levels:
                    continue
                if (all(w in t or w in a for w in padded) and allowed(i)
                        and _min_span((t + a).split(), words) <= max(window, len(words))):
                    levels[i] = LEVEL_NEAR
        return levels

    # -- persistence -----------------------------------------------------------
    def save(self, directory: str) -> None:
        pd.DataFrame({"t": [t.strip() for t in self.titles],
                      "a": [a.strip() for a in self.abstracts]}).to_parquet(
            os.path.join(directory, PHRASE_PARQUET), index=False)

    @classmethod
    def load(cls, directory: str, n_expected: int) -> Optional[PhraseIndex]:
        path = os.path.join(directory, PHRASE_PARQUET)
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        if len(df) != n_expected:
            return None
        return cls(df["t"].fillna(""), df["a"].fillna(""), normalized=True)


def matched_words(phrases: Sequence[str], title: str, abstract: str) -> List[str]:
    """Content words of the phrases that occur in the paper (normalised)."""
    text = f" {normalize(title)} {normalize(abstract)} "
    words = dict.fromkeys(w for p in phrases for w in content_tokens(p))
    return [w for w in words if f" {w} " in text]
