"""Evaluation: frozen, blind, overlapping rating rounds; quality metrics against the launch bar; and the
free-model comparison for bulk extraction. The product promise ("real, open, feasible gaps") is an
empirical claim, so it is measured."""
from __future__ import annotations

from . import metrics
from .rounds import list_rounds, load_round, rater_queue, save_round
from .sampling import (
                       make_extraction_round,
                       make_gap_round,
                       make_plan_round,
                       make_reading_round,
                       make_search_round,
                       stratified,
                       wilson,
)

__all__ = ["metrics", "list_rounds", "load_round", "save_round", "rater_queue", "make_gap_round",
           "make_extraction_round", "make_search_round", "make_reading_round", "make_plan_round", "stratified",
           "wilson"]
