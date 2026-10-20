"""The Gap Engine: evidence-backed research-gap candidates from five detectors, judged and written up by
the LLM (grounded only in evidence), text-verified, novelty-checked against newer work, and scored.
Every card shows its evidence, why it might not be a gap, and what the novelty check covered."""
from __future__ import annotations

from typing import Any

from .model import TYPE_HELP, TYPE_LABELS, Evidence, GapCandidate, GapCard, NoveltyCheck, NoveltyHit, Scores

__all__ = ["GapCard", "GapCandidate", "Evidence", "NoveltyCheck", "NoveltyHit", "Scores", "TYPE_LABELS",
           "TYPE_HELP", "run_offline_gaps", "gaps_for_workspace"]


def run_offline_gaps(*args: Any, **kwargs: Any) -> Any:
    from .engine import run_offline_gaps as _run

    return _run(*args, **kwargs)


def gaps_for_workspace(*args: Any, **kwargs: Any) -> Any:
    from .workspace import gaps_for_workspace as _run

    return _run(*args, **kwargs)
