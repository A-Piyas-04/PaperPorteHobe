"""Unsettled questions: findings on the same (task, dataset, metric) whose directions disagree.

Findings are LLM paraphrases, so they are never shown as quotes ("Reported (paraphrased)"), and every
candidate goes through the LLM conflict judge because direction labels are noisy.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List

import pandas as pd

from ...extract.canonical import normalize_value
from ..evidence import evidence_for, number_evidence
from ..model import GapCandidate

_POSITIVE = {"improves"}
_NEGATIVE = {"no_effect", "worsens"}


def _canon(ctx: Any, facet: str, value: str) -> str:
    v = str(value or "").strip()
    if not v:
        return ""
    return ctx.facet_aliases.get(facet, {}).get(v) or normalize_value(v)


def _disagree(a: str, b: str) -> bool:
    return (a in _POSITIVE and b in _NEGATIVE) or (b in _POSITIVE and a in _NEGATIVE) or "mixed" in (a, b)


def _similar(a: str, b: str) -> bool:
    from rapidfuzz import fuzz

    a, b = str(a or "").lower(), str(b or "").lower()
    if not a or not b:
        return False
    return fuzz.token_set_ratio(a, b) >= 70 or a in b or b in a


def detect_unsettled(ctx: Any) -> List[GapCandidate]:
    g = ctx.cfg["gaps"]
    F = ctx.findings
    if F is None or not len(F):
        return []
    F = F.copy()
    F["k_task"] = [_canon(ctx, "task", v) for v in F["task"]]
    F["k_dataset"] = [_canon(ctx, "dataset", v) for v in F["dataset"]]
    F["k_metric"] = [_canon(ctx, "metric", v) for v in F["metric"]]
    F = F[(F["k_task"] != "") & (F["k_dataset"] != "") & (F["k_metric"] != "")]
    out: List[GapCandidate] = []
    for (task, dataset, metric), grp in F.groupby(["k_task", "k_dataset", "k_metric"]):
        if grp["arxiv_id"].nunique() < int(g["unsettled_min_papers"]):
            continue
        conflicting: Dict[str, pd.Series] = {}
        for (_, x), (_, y) in combinations(grp.iterrows(), 2):
            if x["arxiv_id"] == y["arxiv_id"]:
                continue
            if _disagree(str(x["direction"]), str(y["direction"])) and _similar(x["compared_to"], y["compared_to"]):
                conflicting[str(x["finding_id"])] = x
                conflicting[str(y["finding_id"])] = y
        if len({r["arxiv_id"] for r in conflicting.values()}) < 2:
            continue
        ev = [evidence_for(ctx, str(r["arxiv_id"]), "conflicting_result",
                           note=f"Reported (paraphrased): {r['claim']} [{r['direction']}]")
              for r in conflicting.values()]
        rows = list(conflicting.values())
        out.append(GapCandidate(
            type="unsettled_question", scope=ctx.scope,
            key_terms={"task": str(rows[0]["task"]), "dataset": str(rows[0]["dataset"]),
                       "metric": str(rows[0]["metric"])},
            seed_statement=(f"Papers report conflicting results for {rows[0]['task']} on {rows[0]['dataset']} "
                            f"({rows[0]['metric']}) when compared with {rows[0]['compared_to'] or 'the same baseline'}."),
            evidence=number_evidence(ev),
            stats={"n_conflicting": len(conflicting), "n_papers": int(grp["arxiv_id"].nunique()),
                   "directions": {str(k): int(v) for k, v in pd.Series([r["direction"] for r in rows]).value_counts().items()},
                   "task": task, "dataset": dataset, "metric": metric},
            specificity=0.75, strength=float(min(1.0, len(conflicting) / 4))))
    return out
