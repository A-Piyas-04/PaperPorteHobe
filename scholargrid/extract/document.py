"""Builds the text the LLM reads for one paper: budgeted sections with [S#] markers + cue sentences."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

_ORDER = ["abstract", "introduction", "experiments", "results", "discussion", "limitations",
          "future_work", "threats_to_validity", "conclusion", "broader_impact"]


@dataclass
class PaperDocument:
    arxiv_id: str
    text: str                     # what the model sees
    sources: Dict[str, str]       # "S5" -> full stored section text (for quote verification)
    anchors: Dict[str, str]       # "S5" -> HTML anchor (e.g. "S5.SS1")
    kinds: Dict[str, str]         # "S5" -> section kind
    has_fulltext: bool

    @property
    def input_hash(self) -> str:
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()[:16]


def build_document(arxiv_id: str, title: str, abstract: str, sections: pd.DataFrame, cues: pd.DataFrame,
                   budget: Dict[str, int], max_chars: int, max_cues: int) -> PaperDocument:
    parts: List[str] = [f"TITLE: {title}"]
    sources: Dict[str, str] = {}
    anchors: Dict[str, str] = {}
    kinds: Dict[str, str] = {}
    secs = sections.sort_values("idx") if len(sections) else sections
    have_abstract = bool(len(secs)) and bool((secs["kind"] == "abstract").any())
    if not have_abstract:
        taken = {int(i) for i in secs["idx"]} if len(secs) else set()
        label = "S0" if 0 not in taken else f"S{max(taken) + 1}"
        parts.append(f"[{label}] ABSTRACT\n{abstract}")
        sources[label], anchors[label], kinds[label] = abstract, "abstract", "abstract"
    for kind in _ORDER:
        if not len(secs):
            break
        for r in secs[secs["kind"] == kind].itertuples():
            if not r.text:
                continue
            label = f"S{int(r.idx)}"
            body = str(r.text)[: int(budget.get(kind, 2000))]
            parts.append(f"[{label}] {kind.upper()}: {r.heading}\n{body}")
            sources[label], anchors[label], kinds[label] = str(r.text), str(r.anchor or ""), kind
    if max_cues and len(cues):
        lines = [f"- [S{int(c.section_idx)}] {c.sentence}" for c in cues.head(max_cues).itertuples()]
        parts.append("GAP-CUE SENTENCES DETECTED AUTOMATICALLY (may be incomplete):\n" + "\n".join(lines))
    text = "\n\n".join(parts)[:max_chars]
    return PaperDocument(arxiv_id, text, sources, anchors, kinds, has_fulltext=bool(len(secs)))
