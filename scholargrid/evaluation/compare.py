"""Free-model comparison for bulk extraction: quality vs requests per paper, per route.

Each route extracts the same fixed sample into its own store (the LLM cache key includes the route, so
runs don't collide and a rerun after a quota stop continues). The first route is the reference.
Decision rule (written into the report): use the higher-quota route for bulk extraction only if, versus
the reference, human-rated problem precision is within 5 points, the verified-quote rate is >= 95% of the
reference's, and the task/dataset Jaccard is >= 0.7.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from datetime import date
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import ensure_dir, get_logger
from .rounds import new_round, save_round

log = get_logger("evaluation.compare")
DECISION_RULE = ("Use the higher-quota route for bulk extraction only if, versus the reference, the human-rated "
                 "problem precision is within 5 percentage points, the verified-quote rate is >= 95% of the "
                 "reference's, and the task/dataset Jaccard is >= 0.7. Otherwise keep the stronger model first in "
                 "the route and accept slower progress (more days). Record the decision in Docs/runbook.md.")


def _slug(route: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in route)


def _route_cfg(cfg: Config, route: str, base_dir: str) -> Config:
    raw = copy.deepcopy(cfg.raw)
    raw["llm"]["tasks"]["extract"]["route"] = [route]
    raw["extract"]["store_dir"] = os.path.join(base_dir, _slug(route))
    raw["extract"]["mode"] = "llm"
    raw["extract"]["on_quota"] = "skip"
    raw["scope"]["max_papers"] = None
    return Config(raw=raw, path=cfg.path)


def _jaccard(a: set, b: set) -> Optional[float]:
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def sample_papers(cfg: Config, n: int, seed: int) -> pd.DataFrame:
    from ..fulltext import FulltextStore
    from ..scope import select_scope

    path = os.path.join(cfg.stages_dir, "enriched.parquet")
    df = pd.read_parquet(path)
    scope = select_scope(df, cfg)
    st = FulltextStore(cfg).status
    ok = set(st.loc[st["status"] == "ok", "arxiv_id"].astype(str)) if len(st) else set()
    pool = scope[scope["arxiv_id"].astype(str).isin(ok)] if ok else scope
    return pool.sample(min(n, len(pool)), random_state=seed) if len(pool) else pool


def compare_extract(cfg: Config, routes: List[str], n: int, *, llm_for: Optional[Callable[[Config], Any]] = None,
                    embedder: Any = None, df: Optional[pd.DataFrame] = None, round_name: Optional[str] = None
                    ) -> Dict:
    from ..extract import ExtractionStore, run_extract
    from ..extract.canonical import normalize_value
    from ..llm import get_llm

    seed = int(cfg["evaluation"]["sample_seed"])
    sample = df if df is not None else sample_papers(cfg, n, seed)
    base = ensure_dir(os.path.join(cfg.abspath(cfg["evaluation"]["report_dir"]), "compare_extract"))
    llm_for = llm_for or get_llm
    per_route: Dict[str, Dict] = {}
    stores: Dict[str, pd.DataFrame] = {}
    for route in routes:
        rc = _route_cfg(cfg, route, base)
        t0 = time.time()
        stats = run_extract(sample, rc, llm=llm_for(rc))
        secs = time.time() - t0
        rows = ExtractionStore(rc).latest(sample["arxiv_id"].astype(str))
        stores[route] = rows
        n_done = max(1, len(rows))
        raw = int(rows["n_problems_raw"].sum()) if len(rows) else 0
        ver = int(rows["n_problems_verified"].sum()) if len(rows) else 0
        fallback = int(rows["mode"].astype(str).str.startswith("heuristic").sum()) if len(rows) else 0
        per_route[route] = {"route": route, "papers": int(len(rows)), "llm_papers": int(stats.get("llm", 0)),
                            "verified_quote_rate": round(ver / raw, 3) if raw else None,
                            "problems_per_paper": round(ver / n_done, 2),
                            "requests_per_paper": round(int(stats.get("requests_used", 0)) / n_done, 3),
                            "truncation_or_fallback_rate": round(fallback / n_done, 3),
                            "seconds_per_paper": round(secs / n_done, 2), "quota_stopped": bool(stats.get("quota_stopped"))}
    ref = routes[0]

    def facet_sets(rows: pd.DataFrame) -> Dict[str, Dict[str, set]]:
        out = {}
        for r in rows.itertuples():
            p = json.loads(r.payload)
            out[str(r.arxiv_id)] = {"tasks": {normalize_value(x) for x in p.get("tasks", [])},
                                    "datasets": {normalize_value(x) for x in p.get("datasets", [])}}
        return out

    def statements(rows: pd.DataFrame) -> Dict[str, List[str]]:
        return {str(r.arxiv_id): [q["statement"] for q in json.loads(r.problems)] for r in rows.itertuples()}

    ref_f, ref_s = facet_sets(stores[ref]), statements(stores[ref])
    for route in routes:
        f, s = facet_sets(stores[route]), statements(stores[route])
        jac = [j for aid in ref_f if aid in f
               for j in (_jaccard(ref_f[aid]["tasks"], f[aid]["tasks"]), _jaccard(ref_f[aid]["datasets"], f[aid]["datasets"]))
               if j is not None]
        per_route[route]["task_dataset_jaccard"] = round(float(np.mean(jac)), 3) if jac else None
        matched, total = 0, 0
        for aid, ref_list in ref_s.items():
            mine = s.get(aid, [])
            for st in ref_list:
                total += 1
                matched += int(_matches(st, mine, embedder))
        per_route[route]["ref_problems_matched"] = round(matched / total, 3) if total else None
    items: List[Dict] = []
    for route, rows in stores.items():
        for r in rows.itertuples():
            for q in json.loads(r.problems):
                qid = hashlib.sha1(q["quote"].encode("utf-8")).hexdigest()[:10]
                items.append({"item_id": f"problem:{_slug(route)}:{r.arxiv_id}:{qid}",
                              "item_type": "problem",
                              "payload": {"paper_id": str(r.arxiv_id), "title": str(r.arxiv_id),
                                          "url": f"https://arxiv.org/abs/{r.arxiv_id}", "quote": q["quote"],
                                          "statement": q["statement"], "section": q.get("section_kind", "")},
                              "context": {"route": route}, "overlap": False})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items)) if items else []
    items = [items[int(i)] for i in order]
    name = round_name or f"compare-extract-{date.today().isoformat()}"
    save_round(cfg, new_round(name, "extraction", items, {"routes": routes, "reference": ref}))
    table = list(per_route.values())
    _write_report(cfg, name, table, ref)
    return {"round": name, "routes": table, "reference": ref}


def _matches(statement: str, others: List[str], embedder: Any) -> bool:
    if not others:
        return False
    if embedder is not None:
        from ..utils import l2_normalize

        v = l2_normalize(np.asarray(embedder.transform([statement, *others]), dtype=np.float32))
        return bool((v[1:] @ v[0]).max() >= 0.8)
    from rapidfuzz import fuzz

    return max(fuzz.token_set_ratio(statement, o) for o in others) >= 80


def _write_report(cfg: Config, name: str, table: List[Dict], ref: str) -> str:
    cols = ["route", "papers", "llm_papers", "verified_quote_rate", "problems_per_paper", "task_dataset_jaccard",
            "ref_problems_matched", "requests_per_paper", "truncation_or_fallback_rate", "seconds_per_paper",
            "quota_stopped"]
    lines = [f"# Extraction route comparison ({name})", "", f"Reference route: `{ref}`", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in table]
    lines += ["", "## Decision rule", "", f"> {DECISION_RULE}", "",
              f"Human-rated problem precision per route: rate the blind round `{name}` on the Evaluate page, then "
              f"`python -m scholargrid.evaluation report --batch {name}`."]
    path = os.path.join(cfg.abspath(cfg["evaluation"]["report_dir"]), f"{name}.compare.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
