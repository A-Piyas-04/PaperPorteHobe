"""Full-text acquisition: fetch arXiv HTML -> ar5iv -> PDF, split into classified sections, find
gap-cue sentences. Authors state open problems in Limitations / Future Work, not in abstracts."""
from __future__ import annotations

from .model import GAP_KINDS, Cue, Section, SectionKind
from .run import run_fulltext
from .sections import classify_heading, find_cues
from .store import FulltextStore

__all__ = ["Section", "Cue", "SectionKind", "GAP_KINDS", "FulltextStore", "run_fulltext",
           "classify_heading", "find_cues"]
