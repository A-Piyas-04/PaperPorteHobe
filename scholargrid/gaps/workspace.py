"""Online gap cards for a topic workspace, plus relevant offline area cards ("From the wider area").

One topic costs about ``online_max_new_cards`` synthesis requests plus one packed judge request per card.
Cached in ``data/workspaces/<ws.key>.gaps.json``; not cached when the free quota stopped generation, so a
later visit finishes the job (finished LLM answers are cached and cost nothing again).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..config import Config
from ..utils import ensure_dir, get_logger, l2_normalize
from .context import build_context
from .engine import Pair, candidates_for, finalize
from .model import GapCandidate, GapCard

log = get_logger("gaps.workspace")
Progress = Optional[Callable[[str, float], None]]
AREA_SHARE = 0.2
AREA_MIN_SIM = 0.35


def cache_path(ws_key: str, cfg: Config) -> str:
    return os.path.join(cfg.workspace_dir, f"{ws_key}.gaps.json")


def load_cached_gaps(ws_key: str, cfg: Config) -> Optional[Dict]:
    path = cache_path(ws_key, cfg)
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > float(cfg["workspace"]["cache_ttl_hours"]) * 3600:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def save_cached_gaps(ws_key: str, cfg: Config, data: Dict) -> None:
    ensure_dir(cfg.workspace_dir)
    path = cache_path(ws_key, cfg)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, default=str)
    os.replace(tmp, path)


def area_cards_for(ws: Any, bundle: Any, cfg: Config) -> List[Dict]:
    """Offline area cards relevant to the topic: areas covering >= 20% of its collection papers, cards whose
    statement is close to the query."""
    areas = (getattr(bundle, "gaps", None) or {}).get("areas") or {}
    idx = np.array(ws.bundle_indices(), dtype=int)
    if not areas or not len(idx):
        return []
    labels = bundle.labels[idx]
    cids, counts = np.unique(labels[labels >= 0], return_counts=True)
    covering = {str(int(c)) for c, n in zip(cids, counts) if n >= AREA_SHARE * len(idx)}
    cards = [c for cid in covering for c in areas.get(cid, [])]
    if not cards:
        return []
    emb = bundle.embedder
    if emb is None:
        return cards[:6]
    q = np.asarray(emb.encode_query(ws.query), dtype=np.float32)
    vecs = l2_normalize(np.asarray(emb.transform([c["statement"] for c in cards]), dtype=np.float32))
    sims = vecs @ (q / (np.linalg.norm(q) + 1e-9))
    keep = [dict(c, relevance=round(float(s), 3)) for c, s in zip(cards, sims) if s >= AREA_MIN_SIM]
    return sorted(keep, key=lambda c: -c["relevance"])[:8]


def gaps_for_workspace(ws: Any, bundle: Any, searcher: Any, cfg: Config, llm: Any, progress: Progress = None,
                       use_cache: bool = True) -> Dict:
    if use_cache:
        cached = load_cached_gaps(ws.key, cfg)
        if cached is not None:
            return cached
    g = cfg["gaps"]

    def say(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)

    say("Collecting what authors say is still open…", 0.1)
    order = list(dict.fromkeys(p.bundle_index for p in ws.papers if p.bundle_index is not None))
    idx = np.array(order, dtype=int)
    sub_of = {p.bundle_index: p.subtopic for p in ws.papers if p.bundle_index is not None}
    sub_labels = np.array([sub_of.get(int(i), -1) for i in idx], dtype=int)
    names = {s.id: s.name for s in ws.top_level()}
    pairs: List[Pair] = []
    if len(idx):
        ctx = build_context(bundle, idx, {"kind": "workspace", "id": ws.key, "label": ws.query}, cfg,
                            bundle.embedder, sub_labels=sub_labels, sub_names=names)
        pairs += [(c, ctx) for c in candidates_for(ctx)]
        say("Looking inside each sub-topic…", 0.3)
        for s in ws.top_level():
            sidx = np.array([i for i, lab in zip(idx, sub_labels) if lab == s.id], dtype=int)
            if len(sidx) < max(10, int(g["min_support"]) * 3):
                continue
            sctx = build_context(bundle, sidx, {"kind": "subtopic", "id": f"{ws.key}:{s.id}", "label": s.name},
                                 cfg, bundle.embedder, all_vectors=ctx.all_vectors)
            pairs += [(c, sctx) for c in candidates_for(sctx)]
    seen: set = set()
    uniq: List[Pair] = []
    for c, x in sorted(pairs, key=lambda p: -p[0].strength):
        key = (c.type, tuple(sorted(c.key_terms.items())))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((c, x))
    budget = int(g["online_max_new_cards"])
    uniq = uniq[: budget * 2]
    say("Writing and checking the cards (this uses the AI)…", 0.5)
    cards, already, stats = finalize(uniq, bundle.embedder, llm, cfg, offline=False, searcher=searcher,
                                     live=bool(cfg["search"]["live"]) and searcher is not None,
                                     max_cards=budget)
    say("Adding cards from the wider area…", 0.9)
    data = {"created_at": time.time(), "ws_key": ws.key, "query": ws.query,
            "cards": [c.model_dump(mode="json") for c in cards],
            "already_explored": [c.model_dump(mode="json") for c in already],
            "area_cards": area_cards_for(ws, bundle, cfg), "stats": stats}
    data = json.loads(json.dumps(data, default=str))
    if not stats.get("quota_stopped"):
        save_cached_gaps(ws.key, cfg, data)
    say("Done", 1.0)
    return data


def recheck_novelty(card: Dict, ws: Any, bundle: Any, searcher: Any, cfg: Config, llm: Any) -> Dict:
    """'Check novelty now': a live novelty check for one card, rescored; updates the workspace cache."""
    from .novelty import novelty_check
    from .scoring import confidence, score

    gc = GapCard.model_validate(card)
    idx = np.array(sorted({e for e in ws.bundle_indices()}), dtype=int)
    ctx = build_context(bundle, idx if len(idx) else np.arange(min(1, len(bundle.df))),
                        {"kind": "workspace", "id": ws.key, "label": ws.query}, cfg, None)
    gc.novelty = novelty_check(gc, ctx, bundle.embedder, llm, cfg, searcher=searcher, live=searcher is not None)
    cand = GapCandidate(type=gc.type, scope=gc.scope, key_terms=gc.key_terms, seed_statement=gc.statement,
                        evidence=gc.evidence, stats=gc.stats, specificity=float(gc.stats.get("specificity", 0.5)))
    gc.scores = score(gc, cand, ctx, cfg)
    gc.confidence = confidence(gc)  # type: ignore[assignment]
    new = json.loads(json.dumps(gc.model_dump(mode="json"), default=str))
    cached = load_cached_gaps(ws.key, cfg)
    if cached is not None:
        for key in ("cards", "area_cards"):
            cached[key] = [new if c.get("id") == gc.id else c for c in cached.get(key, [])]
        save_cached_gaps(ws.key, cfg, cached)
    return new
