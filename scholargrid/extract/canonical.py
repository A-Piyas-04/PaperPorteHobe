"""Merge facet surface forms ("LLMs", "large language models") into canonical values.

Precision over recall: two surface forms of one facet type merge when
1. their normalized forms are equal (lowercase, unicode dashes, articles, simple plural folding), or
2. one is an acronym whose mined long form equals the other, or
3. embedding cosine >= ``canonical_min_cosine`` AND ``token_set_ratio`` >= ``canonical_min_string_ratio``
   (both guards, because embeddings put "BERT" next to "RoBERTa" and "BLEU" next to "ROUGE").
A wrong merge creates fake gaps downstream; a missed merge only adds noise.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Mapping, Optional

import numpy as np

from ..index import exact_topk
from ..labeling import _stem

_ARTICLE = re.compile(r"^(the|a|an)\s+")


def normalize_value(v: str) -> str:
    s = unicodedata.normalize("NFKC", str(v)).lower().replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"[\"'`()\[\]]", " ", s)
    s = _ARTICLE.sub("", " ".join(s.split())).strip(" .,;:")
    words = s.split()
    if words:
        words[-1] = _stem(words[-1])
    return " ".join(words)


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def canonicalize_values(raw_counts: Mapping[str, int], vectors: Optional[np.ndarray],
                        acronyms: Mapping[str, Mapping[str, int]], min_cos: float, min_ratio: int
                        ) -> Dict[str, str]:
    """{raw value -> canonical display value}. ``vectors`` rows align with ``list(raw_counts)``."""
    from rapidfuzz import fuzz

    raws = list(raw_counts)
    norms = [normalize_value(r) for r in raws]
    uf = _UF(len(raws))
    first: Dict[str, int] = {}
    for i, n in enumerate(norms):                       # rule 1
        if n in first:
            uf.union(i, first[n])
        else:
            first[n] = i
    long_of = {normalize_value(a): max(forms, key=lambda f: forms[f]) for a, forms in acronyms.items() if forms}
    for i, n in enumerate(norms):                       # rule 2
        lf = long_of.get(n)
        if lf:
            j = first.get(normalize_value(lf))
            if j is not None:
                uf.union(i, j)
    if vectors is not None and len(raws) > 1:           # rule 3
        idx, sims = exact_topk(vectors, vectors, min(10, len(raws) - 1), exclude_self=True)
        for i in range(len(raws)):
            for j, s in zip(idx[i], sims[i]):
                if s >= min_cos and fuzz.token_set_ratio(norms[i], norms[int(j)]) >= min_ratio:
                    uf.union(i, int(j))
    groups: Dict[int, List[int]] = {}
    for i in range(len(raws)):
        groups.setdefault(uf.find(i), []).append(i)
    out: Dict[str, str] = {}
    for members in groups.values():
        canon = max(members, key=lambda k: (raw_counts[raws[k]], -len(raws[k])))
        for k in members:
            out[raws[k]] = raws[canon]
    return out
