"""Gap detectors: pure functions of a GapContext (no I/O), so they are unit-testable with hand-built tables."""
from __future__ import annotations

from .bridge import detect_bridge
from .combinatorial import detect_combinatorial
from .evaluation import detect_evaluation
from .stated import detect_stated
from .unsettled import detect_unsettled

DETECTORS = {"stated": detect_stated, "combinatorial": detect_combinatorial, "evaluation": detect_evaluation,
             "unsettled": detect_unsettled, "bridge": detect_bridge}

__all__ = ["DETECTORS", "detect_stated", "detect_combinatorial", "detect_evaluation", "detect_unsettled",
           "detect_bridge"]
