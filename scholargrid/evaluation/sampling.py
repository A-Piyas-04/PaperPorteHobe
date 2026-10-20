"""Stratified, overlapping samples for evaluation rounds (gap cards, extractions, search, reading, plans)."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from ..config import Config
from .rounds import new_round

STRATUM_FLOOR = 2


def _mark_overlap(items: List[Dict], overlap: float, rng: random.Random) -> List[Dict]:
    k = int(round(len(items) * float(overlap)))
    chosen = set(rng.sample(range(len(items)), min(k, len(items)))) if items else set()
    return [{**it, "overlap": i in chosen} for i, it in enumerate(items)]


def stratified(cards: List[Dict], n: int, seed: int) -> List[Dict]:
    """Proportional allocation over (type, confidence) strata, with a floor of 2 per stratum when available."""
    rng = random.Random(seed)
    strata: Dict[tuple, List[Dict]] = defaultdict(list)
    for c in cards:
        strata[(c.get("type", ""), c.get("confidence", ""))].append(c)
    for v in strata.values():
        rng.shuffle(v)
    n = min(n, len(cards))
    total = len(cards)
    alloc = {k: min(len(v), max(STRATUM_FLOOR, int(round(n * len(v) / total)))) for k, v in strata.items()}
    while sum(alloc.values()) > n:                     # trim the largest strata first, never below the floor
        k = max((k for k in alloc if alloc[k] > min(STRATUM_FLOOR, len(strata[k]))), key=lambda k: alloc[k],
                default=None)
        if k is None:
            break
        alloc[k] -= 1
    picked = [c for k, v in sorted(strata.items()) for c in v[: alloc[k]]]
    left = [c for k, v in sorted(strata.items()) for c in v[alloc[k]:]]
    rng.shuffle(left)
    picked += left[: max(0, n - len(picked))]
    return picked[:n] if len(picked) > n else picked


def make_gap_round(name: str, cards: List[Dict], n: int, overlap: float, seed: int,
                   context: Dict | None = None) -> Dict:
    rng = random.Random(seed)
    seen, uniq = set(), []
    for c in cards:
        if c.get("id") not in seen:
            seen.add(c.get("id"))
            uniq.append(c)
    items = [{"item_id": str(c["id"]), "item_type": "gap_card", "payload": c,
              "context": {**(context or {}), "scope": c.get("scope", {}),
                          "model": (c.get("generated_by") or {}).get("model", "")}}
             for c in stratified(uniq, n, seed)]
    return new_round(name, "gaps", _mark_overlap(items, overlap, rng), {"n_pool": len(uniq), "seed": seed})


def make_extraction_round(name: str, bundle: Any, n_papers: int, seed: int, overlap: float = 0.2,
                          context: Dict | None = None) -> Dict:
    rng = random.Random(seed)
    P = bundle.problems
    if P is None or not len(P):
        return new_round(name, "extraction", [], {"n_pool": 0})
    ids = sorted(set(P["arxiv_id"].astype(str)))
    rng.shuffle(ids)
    ids = ids[:n_papers]
    df = bundle.df.drop_duplicates("arxiv_id").set_index(bundle.df.drop_duplicates("arxiv_id")["arxiv_id"].astype(str))
    F = bundle.facets
    col = "value" if len(F) and "value" in F.columns else "value_raw"
    items: List[Dict] = []
    for aid in ids:
        r = df.loc[aid] if aid in df.index else None
        title = str(r["title"]) if r is not None else aid
        url = f"https://arxiv.org/abs/{aid}"
        for q in P[P["arxiv_id"].astype(str) == aid].itertuples():
            items.append({"item_id": f"problem:{q.problem_id}", "item_type": "problem",
                          "payload": {"paper_id": aid, "title": title, "url": url, "quote": q.quote,
                                      "statement": q.statement, "section": q.section_kind,
                                      "anchor_url": f"https://arxiv.org/html/{aid}#{q.anchor}" if q.anchor else ""},
                          "context": {**(context or {}), "mode": getattr(q, "source_mode", "")}})
        fac = F[F["arxiv_id"].astype(str) == aid] if len(F) else F
        facets = {k: sorted(set(map(str, fac.loc[fac["facet"] == f, col]))) if len(fac) else []
                  for k, f in (("tasks", "task"), ("methods", "method"), ("datasets", "dataset"), ("metrics", "metric"))}
        items.append({"item_id": f"facets:{aid}", "item_type": "facets",
                      "payload": {"paper_id": aid, "title": title, "url": url,
                                  "abstract": str(r["abstract"])[:1500] if r is not None else "", **facets},
                      "context": dict(context or {})})
    return new_round(name, "extraction", _mark_overlap(items, overlap, rng), {"n_papers": len(ids), "seed": seed})


def make_search_round(name: str, cfg: Config, bundle: Any, n_queries: int, top: int = 10, overlap: float = 0.2,
                      seed: int = 7) -> Dict:
    from ..judgments import sample_queries
    from ..search import search
    from ..utils import load_json

    rng = random.Random(seed)
    spec = load_json(cfg.abspath(cfg["validation"]["queries_file"]))
    items: List[Dict] = []
    for q in sample_queries(spec["queries"], n_queries):
        res = search(q["query"], bundle.embedder, bundle.embeddings, bundle.df, bundle.labels, bundle.clusters_meta,
                     top_k=top, bm25=bundle.bm25, index=bundle.index, cfg=cfg, phrase_index=bundle.phrase,
                     acronyms=bundle.acronyms)
        for r in res["results"]:
            items.append({"item_id": f"search:{q['query']}|{r['arxiv_id']}", "item_type": "search",
                          "payload": {"query": q["query"], "arxiv_id": r["arxiv_id"], "rank": r["rank"],
                                      "title": r["title"], "abstract": str(r.get("abstract", ""))[:800],
                                      "url": r["url"]},
                          "context": {"mode": res.get("mode")}})
    return new_round(name, "search", _mark_overlap(items, overlap, rng), {"n_queries": n_queries, "top": top})


def make_reading_round(name: str, bundle: Any, n_areas: int, seed: int, overlap: float = 0.2) -> Dict:
    rng = random.Random(seed)
    paths = {k: v for k, v in (getattr(bundle, "reading_paths", None) or {}).items() if v}
    keys = sorted(paths)
    rng.shuffle(keys)
    items = [{"item_id": f"reading:{k}", "item_type": "reading",
              "payload": {"area_id": k, "label": (bundle.clusters_meta.get(int(k)) or {}).get("label", f"Area {k}"),
                          "path": paths[k]}, "context": {}} for k in keys[:n_areas]]
    return new_round(name, "reading", _mark_overlap(items, overlap, rng), {"seed": seed})


def make_plan_round(name: str, plans: List[Dict], seed: int, overlap: float = 0.2) -> Dict:
    rng = random.Random(seed)
    items = [{"item_id": f"plan:{p['id']}", "item_type": "plan", "payload": p,
              "context": {"mode": (p.get("generated_by") or {}).get("mode", ""),
                          "model": (p.get("generated_by") or {}).get("model", "")}} for p in plans]
    return new_round(name, "plans", _mark_overlap(items, overlap, rng), {"seed": seed})


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    """Wilson score interval for a proportion (small samples)."""
    if n <= 0:
        return [float("nan"), float("nan")]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def round_digest(rnd: Dict) -> str:
    return json.dumps({"name": rnd["name"], "n": len(rnd["items"])})
