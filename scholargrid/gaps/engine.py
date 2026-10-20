"""Gap engine: detectors -> judge -> synthesize -> verify -> novelty -> score -> filter -> sort.

All LLM work happens in ``finalize`` so the order and the request count are controlled in one place.
Precision over recall: twelve good cards beat a hundred mediocre ones.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import Config
from ..utils import get_logger, l2_normalize, save_json
from . import prompts as _prompts  # noqa: F401  (registers the gap prompts)
from .context import GapContext, build_context
from .detectors import DETECTORS, detect_bridge
from .model import GapCandidate, GapCard
from .novelty import novelty_check
from .scoring import confidence, score
from .synthesize import evidence_lines, synthesize, template_card

log = get_logger("gaps")
GAPS_JSON = "gaps.json"
MIN_AREA_PAPERS = 30
Pair = Tuple[GapCandidate, GapContext]


def candidates_for(ctx: GapContext) -> List[GapCandidate]:
    out: List[GapCandidate] = []
    for name in ctx.cfg["gaps"]["detectors"]:
        try:
            out += DETECTORS[name](ctx)
        except Exception as exc:          # one detector must never break the others
            log.exception("Detector %s failed on %s: %s", name, ctx.scope, exc)
    seen, uniq = set(), []
    for c in sorted(out, key=lambda c: -c.strength):
        if c.id not in seen:
            seen.add(c.id)
            uniq.append(c)
    return uniq[: int(ctx.cfg["gaps"]["max_candidates_per_scope"])]


def _check_evidence(c: GapCandidate, ctx: GapContext) -> GapCandidate:
    """Every quote must come from problems.parquet for that paper; anything else is a bug signal."""
    P = ctx.problems
    if not any(e.quote for e in c.evidence):
        return c
    known: Dict[str, set] = {}
    if P is not None and len(P):
        for aid, q in zip(P["arxiv_id"].astype(str), P["quote"].astype(str)):
            known.setdefault(aid, set()).add(q)
    kept = [e for e in c.evidence if not e.quote or e.quote in known.get(e.paper_id, set())]
    if len(kept) != len(c.evidence):
        log.error("Dropped %d evidence items with unverified quotes from %s.", len(c.evidence) - len(kept), c.id)
        c.evidence = kept
    return c


def _judge(pairs: List[Pair], llm: Any) -> Tuple[List[Pair], int, bool]:
    """Combination and conflict judges (packed). Returns (kept pairs, judged out, quota hit)."""
    from ..llm import BudgetExceeded, LLMError, QuotaExhausted
    from .prompts import ComboJudgement, ConflictJudgement

    if llm is None or not llm.available():
        return pairs, 0, False
    combo = [(c.id, c) for c, _ in pairs if c.type == "unexplored_combination"]
    conflict = [(c.id, c) for c, _ in pairs if c.type == "unsettled_question"]
    drop: set = set()
    quota = False

    def run(prompt: str, schema: type, items: List[Tuple[str, Dict[str, str]]]) -> Dict[str, Any]:
        nonlocal quota
        if not items:
            return {}
        try:
            res = llm.structured_many("judge", prompt, schema, items)
        except LLMError:
            return {}
        quota = quota or any(isinstance(v, (QuotaExhausted, BudgetExceeded)) for v in res.values())
        return res

    combo_items = []
    for cid, c in combo:
        (fa, a), (fb, b) = list(c.key_terms.items())[:2]
        combo_items.append((cid, {"a": a, "fa": fa, "b": b, "fb": fb, "bridge": str(c.stats.get("bridge", "")),
                                  "evidence": evidence_lines(c)}))
    for cid, res in run("judge_combination", ComboJudgement, combo_items).items():
        if not isinstance(res, LLMError) and res is not None and (not res.value.meaningful or res.value.trivial):
            drop.add(cid)
    conflict_items = [(cid, {"question": " / ".join(c.key_terms.values()), "findings": evidence_lines(c)})
                      for cid, c in conflict]
    by_id = {c.id: c for c, _ in pairs}
    for cid, res in run("judge_conflict", ConflictJudgement, conflict_items).items():
        if isinstance(res, LLMError) or res is None:
            continue
        if not res.value.is_conflict:
            drop.add(cid)
        else:
            by_id[cid].stats["possible_reasons"] = res.value.possible_reasons
    return [(c, x) for c, x in pairs if c.id not in drop], len(drop), quota


def _card(c: GapCandidate, draft: Any, generated_by: Dict) -> GapCard:
    fallback = template_card(c)
    why_not = [w for w in draft.why_it_might_not_be_a_gap if w] or fallback.why_it_might_not_be_a_gap
    return GapCard(id=c.id, type=c.type, title=draft.title or fallback.title, statement=draft.statement or c.seed_statement,
                   why_it_matters=draft.why_it_matters, why_it_might_not_be_a_gap=why_not,
                   first_steps=draft.first_steps, skills=draft.skills, data_needs=draft.data_needs,
                   compute=draft.compute, feasibility=draft.feasibility, beginner_friendly=draft.beginner_friendly,
                   evidence=c.evidence, key_terms=dict(c.key_terms),
                   stats={**c.stats, "specificity": round(c.specificity, 3), "strength": round(c.strength, 3)},
                   scope=dict(c.scope), generated_by=generated_by, novelty_queries=draft.novelty_queries)


def finalize(pairs: List[Pair], embedder: Any, llm: Any, cfg: Config, *, offline: bool, searcher: Any = None,
             live: bool = False, max_cards: Optional[int] = None) -> Tuple[List[GapCard], List[GapCard], Dict]:
    """Returns (cards, already_explored, stats); cards are capped per scope and sorted by score."""
    g = cfg["gaps"]
    ledger = getattr(llm, "ledger", None)
    req_before = getattr(ledger, "requests", 0)
    pairs = [(_check_evidence(c, ctx), ctx) for c, ctx in pairs]
    pairs = [(c, ctx) for c, ctx in pairs if c.evidence]
    stats: Dict[str, Any] = {"candidates": len(pairs)}
    judge_on = bool(g["judge_with_llm"])
    if judge_on:
        pairs, judged_out, q1 = _judge(pairs, llm)
    else:
        judged_out, q1 = 0, False
    stats["judged_out"] = judged_out
    if llm is not None and llm.available() and g["synthesize_with_llm"]:
        n_judge = sum(1 for c, _ in pairs if c.type in ("unexplored_combination", "unsettled_question"))
        log.info("Gap LLM work: %d candidates -> ~%d synthesis + ~%d judge requests (see "
                 "`python -m scholargrid.llm quota`).", len(pairs), len(pairs),
                 math.ceil(n_judge / 5) + (len(pairs) if g["novelty_check"] else 0))
    ctx_of = {c.id: ctx for c, ctx in pairs}
    drafted, sstats = synthesize([c for c, _ in pairs], llm, cfg, offline)
    stats.update(sstats)
    stats["quota_stopped"] = bool(sstats["quota_stopped"] or q1)
    cards: List[GapCard] = []
    already: List[GapCard] = []
    for c, draft, gen in drafted:
        ctx = ctx_of[c.id]
        card = _card(c, draft, gen)
        if g["novelty_check"]:
            card.novelty = novelty_check(card, ctx, embedder, llm, cfg, searcher=searcher, live=live)
        card.scores = score(card, c, ctx, cfg)
        card.confidence = confidence(card)  # type: ignore[assignment]
        (already if card.novelty.status == "already_addressed" else cards).append(card)
    stats["already_addressed"] = len(already)
    cap = int(max_cards if max_cards is not None else g["max_cards_per_scope"])
    by_scope: Dict[str, List[GapCard]] = {}
    for card in sorted(cards, key=lambda c: -c.scores.total):
        by_scope.setdefault(str(card.scope.get("id")), []).append(card)
    cards = sorted((c for lst in by_scope.values() for c in lst[:cap]), key=lambda c: -c.scores.total)
    if max_cards is not None:
        cards = cards[:max_cards]
    stats["cards"] = len(cards)
    stats["by_type"] = dict(Counter(c.type for c in cards))
    stats["requests_used"] = int(getattr(ledger, "requests", 0) - req_before)
    return cards, (already if g["keep_already_addressed"] else []), stats


def area_names(bundle: Any) -> Dict[int, str]:
    from ..labeling import pretty_label

    return {int(cid): (pretty_label(info["keywords"]) if info.get("keywords") else info.get("label", f"Area {cid}"))
            for cid, info in bundle.clusters_meta.items()}


def write_empty(bundle_dir: str) -> None:
    save_json({"generated_at": time.time(), "version": 1, "areas": {}, "global": [], "stats": {"enabled": False}},
              os.path.join(bundle_dir, GAPS_JSON))


def run_offline_gaps(cfg: Config, bundle_dir: str, llm: Any = None) -> Dict:
    from ..artifacts import load_bundle
    from ..llm import get_llm

    if not cfg["gaps"]["enabled"]:
        write_empty(bundle_dir)
        return {"enabled": False}
    b = load_bundle(cfg, directory=bundle_dir)
    llm = llm if llm is not None else get_llm(cfg)
    all_vecs = l2_normalize(np.asarray(b.embeddings, dtype=np.float32))
    names = area_names(b)
    contexts: List[GapContext] = []
    for cid in sorted(b.clusters_meta):
        idx = np.where(b.labels == int(cid))[0]
        if len(idx) < MIN_AREA_PAPERS:
            continue
        contexts.append(build_context(b, idx, {"kind": "area", "id": int(cid), "label": names.get(int(cid), "")},
                                      cfg, b.embedder, all_vectors=all_vecs))
    contexts.sort(key=lambda c: -len(c.problems))           # areas with the most stated problems first
    pairs: List[Pair] = [(c, ctx) for ctx in contexts for c in candidates_for(ctx)]
    if "bridge" in cfg["gaps"]["detectors"] and len(names) >= 2:
        gctx = build_context(b, np.arange(len(b.df)), {"kind": "global", "id": "global", "label": "All areas"}, cfg,
                             None, sub_labels=b.labels, sub_names=names, all_vectors=all_vecs)
        try:
            pairs += [(c, gctx) for c in detect_bridge(gctx)[: int(cfg["gaps"]["max_candidates_per_scope"])]]
        except Exception as exc:
            log.exception("Global bridge pass failed: %s", exc)
    cards, already, stats = finalize(pairs, b.embedder, llm, cfg, offline=True, live=False)
    areas: Dict[str, List[Dict]] = {}
    global_cards: List[Dict] = []
    for card in cards:
        if card.scope.get("kind") == "global":
            global_cards.append(card.model_dump(mode="json"))
        else:
            areas.setdefault(str(card.scope.get("id")), []).append(card.model_dump(mode="json"))
    stats.update(areas_scanned=len(contexts), areas_with_cards=len(areas))
    out = {"generated_at": time.time(), "version": 1, "areas": areas, "global": global_cards,
           "already_explored": [c.model_dump(mode="json") for c in already], "stats": stats}
    save_json(json.loads(json.dumps(out, default=str)), os.path.join(bundle_dir, GAPS_JSON))
    log.info("Gap cards: %s", {k: v for k, v in stats.items() if k != "by_type"})
    return stats
