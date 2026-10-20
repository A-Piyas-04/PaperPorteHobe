"""Data shapes shared by the full-text parsers and the store."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

SectionKind = Literal[
    "abstract", "introduction", "background", "related_work", "method", "experiments", "results",
    "discussion", "limitations", "future_work", "threats_to_validity", "broader_impact", "ethics",
    "conclusion", "appendix", "acknowledgements", "references", "other"]

GAP_KINDS = {"limitations", "future_work", "discussion", "conclusion", "threats_to_validity",
             "broader_impact"}


@dataclass
class Section:
    idx: int
    parent_idx: Optional[int]
    heading: str
    kind: str
    text: str
    anchor: str = ""          # HTML id (e.g. "S5") so the app can deep-link https://arxiv.org/html/<id>#S5

    def to_row(self, arxiv_id: str, source: str) -> dict:
        return {"arxiv_id": arxiv_id, "source": source, **asdict(self), "n_chars": len(self.text)}


@dataclass
class Cue:
    section_idx: int
    kind: str
    sentence: str
    cue: str                  # name of the pattern that matched
    char_start: int
