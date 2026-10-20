"""run_fulltext: fetch/parse in-scope papers incrementally; returns stats for meta.json."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import pandas as pd

from ..config import Config
from ..scope import select_scope
from ..utils import get_logger
from .fetch import Fetcher
from .sections import find_cues
from .store import FulltextStore
from .synthetic import synthetic_sections

log = get_logger("fulltext")


def run_fulltext(df: pd.DataFrame, cfg: Config,
                 progress: Optional[Callable[[int, int], None]] = None) -> Dict:
    f = cfg["fulltext"]
    stats: Dict = {"enabled": bool(f["enabled"]), "in_scope": 0, "fetched_now": 0, "ok": 0,
                   "unavailable": 0, "errors": 0, "by_source": {}, "coverage": 0.0}
    if not f["enabled"] or f["source"] == "none":
        return stats
    scope = select_scope(df, cfg)
    stats["in_scope"] = int(len(scope))
    store = FulltextStore(cfg)
    synthetic = f["source"] == "synthetic" or cfg["data"]["source"] == "synthetic"
    todo = store.needs_fetch(scope["arxiv_id"].astype(str), int(f["refetch_failed_days"]))
    if f.get("max_papers_per_run"):
        todo = todo[: int(f["max_papers_per_run"])]
    rows = scope.set_index(scope["arxiv_id"].astype(str))
    fetcher = None if synthetic else Fetcher(cfg)
    log.info("Full text: %d in scope, %d to fetch (%s).", len(scope), len(todo),
             "synthetic" if synthetic else "arXiv")
    try:
        for n, aid in enumerate(todo, start=1):
            r = rows.loc[aid]
            if isinstance(r, pd.DataFrame):          # duplicate ids: use the first row
                r = r.iloc[0]
            if synthetic:
                secs = synthetic_sections(aid, str(r["title"]), str(r["abstract"]), str(r["primary_category"]))
                store.add(aid, "ok", "synthetic", secs, find_cues(secs))
            else:
                assert fetcher is not None
                res = fetcher.fetch(aid)
                store.add(aid, res.status, res.source, res.sections, find_cues(res.sections),
                          res.http_status, res.error)
            stats["fetched_now"] += 1
            if n % int(f["flush_every"]) == 0:
                store.flush()
                if progress:
                    progress(n, len(todo))
    finally:
        store.flush()        # Ctrl-C keeps everything fetched so far
    st = store.status[store.status["arxiv_id"].isin(set(scope["arxiv_id"].astype(str)))]
    stats["ok"] = int((st["status"] == "ok").sum())
    stats["unavailable"] = int((st["status"] == "unavailable").sum())
    stats["errors"] = int((st["status"] == "error").sum())
    stats["by_source"] = {str(k): int(v) for k, v in st.loc[st["status"] == "ok", "source"].value_counts().items()}
    stats["coverage"] = round(stats["ok"] / max(1, stats["in_scope"]), 4)
    return stats
