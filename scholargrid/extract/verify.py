"""Verify that every stated-problem quote exists in the source text.

The quote shown to a user must be literally in the paper. The model's quote is located in the cited
section (falling back to the whole document) and replaced by the exact source span, so tiny
transcription differences (whitespace, curly quotes) are repaired. Unlocatable problems are dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .document import PaperDocument
from .schemas import PaperExtraction, StatedProblem

_TRANS = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'", "\u2013": "-",
                        "\u2014": "-", "\u00a0": " "})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.translate(_TRANS)).strip()


@dataclass
class VerifiedProblem:
    problem: StatedProblem
    quote: str               # exact span from the source
    section_label: str
    ratio: float             # 100 = exact


def locate(quote: str, source: str, min_ratio: int) -> Optional[Tuple[str, float]]:
    q, s = _norm(quote).rstrip("…").strip(), _norm(source)
    if not q or not s:
        return None
    i = s.find(q)
    if i >= 0:
        return s[i:i + len(q)], 100.0
    from rapidfuzz import fuzz

    al = fuzz.partial_ratio_alignment(q, s)
    if al is None or al.score < min_ratio:
        return None
    return s[al.dest_start:al.dest_end].strip(), float(al.score)


def verify_quotes(ext: PaperExtraction, doc: PaperDocument, min_ratio: int, max_problems: int
                  ) -> Tuple[List[VerifiedProblem], int]:
    """Returns (verified problems, number dropped)."""
    kept: List[VerifiedProblem] = []
    dropped = 0
    whole = " ".join(doc.sources.values()) or doc.text
    seen: set = set()
    for p in ext.open_problems:
        label = p.section if p.section in doc.sources else ""
        hit = locate(p.quote, doc.sources[label], min_ratio) if label else None
        if hit is None:
            hit = locate(p.quote, whole, min_ratio)
            if hit is not None:
                found = _norm(hit[0])
                label = next((k for k, v in doc.sources.items() if found in _norm(v)), label)
        if hit is None or len(hit[0]) < 20 or hit[0] in seen:
            dropped += 1
            continue
        seen.add(hit[0])
        kept.append(VerifiedProblem(p, hit[0][:300], label, hit[1]))
    kept.sort(key=lambda v: (-v.problem.specificity, -v.ratio))
    return kept[:max_problems], dropped
