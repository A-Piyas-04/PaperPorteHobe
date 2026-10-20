"""Per-paper structured extraction: facets plus the authors' own open problems with verified quotes.
Every gap detector (plan 06) works on these facts; quote verification keeps invented quotes out."""
from __future__ import annotations

from .canonical import canonicalize_values
from .run import run_extract
from .schemas import Finding, MethodMention, PaperExtraction, StatedProblem
from .store import ExtractionStore

canonicalize_facets = canonicalize_values

__all__ = ["PaperExtraction", "StatedProblem", "MethodMention", "Finding", "ExtractionStore", "run_extract",
           "canonicalize_facets", "canonicalize_values"]
