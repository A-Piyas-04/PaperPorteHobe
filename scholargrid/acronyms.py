"""Acronym mining and query expansion.

Abstracts define their acronyms inline: "self-admitted technical debt (SATD)".
:func:`mine` collects those pairs (checking that the initials really spell the
acronym), and :func:`expand` turns a query into the phrases to search for, so
"SATD" also finds "self-admitted technical debt" and vice versa. Manual
``search.synonyms`` from the config are applied the same way, in both
directions.

Two forms of every phrase are kept: ``phrases`` (normalised with plural
folding, for local matching) and ``api_phrases`` (lowercased as written, for
exact-phrase queries to arXiv/OpenAlex, which do their own stemming).
"""
from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .phrase import STOPWORDS, normalize
from .utils import load_json, save_json

ACRONYMS_JSON = "acronyms.json"
_DEF_RE = re.compile(r"((?:[A-Za-z][\w'-]*[\s]+){1,8}?)\(\s*([A-Z][A-Za-z0-9-]{1,11})\s*\)")
_WORD_RE = re.compile(r"[A-Za-z][\w'-]*")
_SIMPLE_RE = re.compile(r"[a-z0-9]+")
_MAX_FORMS = 3
_MIN_ADDED_ACRONYM = 3
_MIN_ADDED_COUNT = 2

AcronymMap = Dict[str, Dict[str, int]]


def simple(text: Optional[str]) -> str:
    """Lowercase ASCII words, punctuation and hyphens as spaces, plurals kept."""
    s = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()
    return " ".join(_SIMPLE_RE.findall(s))


def _letters(acronym: str) -> str:
    a = acronym[:-1] if acronym.endswith("s") and acronym[:-1].isupper() else acronym
    return "".join(ch for ch in a.lower() if ch.isalpha())


def _initials(words: Sequence[str], split_hyphens: bool, skip_stop: bool) -> str:
    parts: List[str] = []
    for w in words:
        parts.extend(p for p in (w.split("-") if split_hyphens else [w]) if p)
    if skip_stop:
        parts = [p for p in parts if p.lower() not in STOPWORDS]
    return "".join(p[0].lower() for p in parts)


def _long_form(before: str, letters: str) -> Optional[str]:
    words = _WORD_RE.findall(before)
    for k in range(1, min(len(words), len(letters) + 3) + 1):
        cand = words[-k:]
        if cand[0].lower() in STOPWORDS:
            continue
        for split in (True, False):
            for skip in (False, True):
                if _initials(cand, split, skip) == letters:
                    return " ".join(cand)
    return None


def mine(texts: Iterable[str]) -> AcronymMap:
    """``{acronym (normalised): {long form (as written, lowercased): count}}``."""
    found: Dict[str, Counter] = defaultdict(Counter)
    for text in texts:
        if not text or "(" not in text:
            continue
        for m in _DEF_RE.finditer(str(text)):
            acronym = m.group(2)
            letters = _letters(acronym)
            if len(letters) < 2 or sum(ch.isupper() for ch in acronym) < 2:
                continue
            long = _long_form(m.group(1), letters)
            if long:
                found[normalize(acronym)][simple(long)] += 1
    return {a: dict(c) for a, c in found.items()}


def merge(*maps: Optional[Mapping[str, Mapping[str, int]]]) -> AcronymMap:
    out: Dict[str, Counter] = defaultdict(Counter)
    for m in maps:
        for a, forms in (m or {}).items():
            out[a].update(forms)
    return {a: dict(c) for a, c in out.items()}


def save(acronyms: AcronymMap, directory: str) -> None:
    save_json(acronyms, os.path.join(directory, ACRONYMS_JSON))


def load(directory: str) -> AcronymMap:
    path = os.path.join(directory, ACRONYMS_JSON)
    return load_json(path) if os.path.exists(path) else {}


@dataclass
class Expansion:
    query: str                                          # normalised query
    phrases: List[str] = field(default_factory=list)    # normalised: query, then expansions
    api_phrases: List[str] = field(default_factory=list)  # as written, for remote sources
    added: List[str] = field(default_factory=list)      # expansions, for display

    @property
    def embed_text(self) -> str:
        return "; ".join([self.api_phrases[0] if self.api_phrases else self.query, *self.added])


def _top_forms(forms: Mapping[str, int]) -> List[tuple]:
    """Most frequent ``(long form, count)``, merging singular/plural spellings."""
    grouped: Dict[str, Counter] = defaultdict(Counter)
    for f, n in forms.items():
        grouped[normalize(f)][f] += n
    ranked = sorted(((sum(c.values()), c.most_common(1)[0][0]) for c in grouped.values()),
                    key=lambda t: (-t[0], t[1]))
    if not ranked:
        return []
    best = ranked[0][0]
    return [(f, n) for n, f in ranked[:_MAX_FORMS] if n >= max(1, best * 0.3)]


def _swap(text: str, old: str, new: str) -> Optional[str]:
    padded = f" {text} "
    if f" {old} " not in padded:
        return None
    return padded.replace(f" {old} ", f" {new} ").strip()


def expand(query: str, acronyms: Optional[Mapping[str, Mapping[str, int]]] = None,
           synonyms: Optional[Mapping[str, Sequence[str]]] = None) -> Expansion:
    q, q_simple = normalize(query), simple(query)
    exp = Expansion(q, [q] if q else [], [q_simple] if q_simple else [])
    if not q:
        return exp
    # (term, alternative, also replace alternative -> term?)
    pairs: List[tuple] = []
    for a, forms in (acronyms or {}).items():
        for long, count in _top_forms(forms):
            # Short or rarely defined acronyms ("TD", "ML") are too ambiguous to
            # search for on their own; they are only expanded, never added.
            pairs.append((a, long, len(_letters(a)) >= _MIN_ADDED_ACRONYM and count >= _MIN_ADDED_COUNT))
    for term, alts in (synonyms or {}).items():
        pairs.extend((simple(term), simple(alt), True) for alt in alts or [])

    for a, b, both_ways in pairs:
        if not a or not b:
            continue
        for old, new in ((a, b), (b, a)) if both_ways else ((a, b),):
            norm_variant = _swap(q, normalize(old), normalize(new))
            if not norm_variant or norm_variant in exp.phrases:
                continue
            simple_variant = None
            for form in (old, old + "s"):
                simple_variant = _swap(q_simple, form, new)
                if simple_variant:
                    break
            exp.phrases.append(norm_variant)
            if simple_variant and simple_variant not in exp.api_phrases:
                exp.api_phrases.append(simple_variant)
            exp.added.append(simple_variant or norm_variant)
    return exp
