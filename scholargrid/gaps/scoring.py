"""Transparent scoring (five weighted components, each explained) and a conservative confidence level."""
from __future__ import annotations

import math
from typing import Any, List

from ..config import Config
from .model import GapCandidate, GapCard, Scores


def _has_code(ctx: Any, ids: List[str]) -> bool:
    p = ctx.profiles
    if p is None or p.empty or "code_url" not in p.columns:
        return False
    return bool((p[p["arxiv_id"].astype(str).isin(ids)]["code_url"].fillna("").astype(str) != "").any())


def score(card: GapCard, cand: GapCandidate, ctx: Any, cfg: Config) -> Scores:
    w = cfg["gaps"]["weights"]
    s = cand.stats
    if card.type in ("stated_open_problem", "long_standing_problem"):
        evidence = 1 - math.exp(-s["support"] / 3) * (1.0 if s.get("author_groups", 1) > 1 else 1.3)
        ev_txt = f"{s['support']} papers ({s.get('author_groups', '?')} independent author groups) state it"
    elif card.type == "unexplored_combination":
        evidence = (1 - s["p_value"]) * min(1.0, s["expected"] / 8)
        ev_txt = f"expected ~{s['expected']} papers combining them, found {s['observed']}"
    elif card.type == "evaluation_gap":
        evidence = min(1.0, s.get("n_papers", 0) / 10)
        ev_txt = f"{s.get('n_papers')} papers on the method"
    elif card.type == "unsettled_question":
        evidence = min(1.0, s.get("n_conflicting", 0) / 4)
        ev_txt = f"{s.get('n_conflicting')} conflicting results"
    else:
        evidence = 1 - min(1.0, s.get("ratio", 1.0))
        ev_txt = f"cross-citation {s.get('ratio', 0):.0%} of expected"
    evidence = max(0.0, min(1.0, evidence))
    novelty = {"novel": 1.0, "partially_explored": 0.5, "not_checked": 0.6,
               "already_addressed": 0.0}[card.novelty.status]
    feas = {"high": 0.9, "medium": 0.6, "low": 0.3}[card.feasibility]
    if _has_code(ctx, [e.paper_id for e in card.evidence]):
        feas = min(1.0, feas + 0.1)
    years = [e.year for e in card.evidence if e.year]
    recency = sum(1 for y in years if y >= ctx.ref_date.year - 1) / max(1, len(years))
    spec = max(0.0, min(1.0, cand.specificity))
    total = (w["evidence"] * evidence + w["novelty"] * novelty + w["feasibility"] * feas
             + w["specificity"] * spec + w["recency"] * recency)
    return Scores(evidence=round(evidence, 3), novelty=novelty, feasibility=round(feas, 3), specificity=round(spec, 3),
                  recency=round(recency, 3), total=round(total, 3),
                  explain={"evidence": ev_txt, "novelty": card.novelty.status.replace("_", " "),
                           "feasibility": f"{card.feasibility} (from evidence: data/code/compute)",
                           "specificity": f"{spec:.0%} concrete",
                           "recency": f"{recency:.0%} of evidence from the last 2 years"})


def confidence(card: GapCard) -> str:
    s = card.scores
    live = card.novelty.scope == "collection+live"
    if s.evidence >= 0.7 and card.novelty.status == "novel" and live:
        return "high"
    if s.evidence >= 0.4 and card.novelty.status in ("novel", "partially_explored", "not_checked"):
        return "medium"
    return "low"
