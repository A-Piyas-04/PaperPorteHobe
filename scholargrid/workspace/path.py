"""Topic-specific reading path: plan 04's backbone + reading_path() over the workspace's papers.

Lift still uses corpus-wide frequencies, so universal citations stay down-weighted.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..foundations import corpus_references, reading_path, reference_lists
from ..foundations.backbone import BACKBONE_COLUMNS, subset_backbone

_REF_CACHE: Dict[Tuple[int, str], Tuple[List[List[str]], int, Counter]] = {}


def _ref_lists(bundle: Any, cfg: Config) -> Tuple[List[List[str]], int, Counter]:
    key = (id(bundle), str(bundle.meta.get("generated_at", "")))
    if key not in _REF_CACHE:
        lists = reference_lists(corpus_references(bundle.df, cfg))
        glob: Counter = Counter()
        for r in lists:
            glob.update(set(r))
        _REF_CACHE.clear()
        _REF_CACHE[key] = (lists, sum(1 for r in lists if r), glob)
    return _REF_CACHE[key]


def workspace_reading_path(bundle: Any, idx: np.ndarray, cfg: Config, lookup: Any
                           ) -> Tuple[List[Dict], Dict[str, Any]]:
    """(reading path items, {reference_coverage, n_citing}) for the workspace's collection papers."""
    f = cfg["foundations"]
    idx = np.asarray(sorted(set(int(i) for i in idx)), dtype=int)
    if not len(idx):
        return [], {"reference_coverage": 0.0, "n_citing": 0}
    lists, n_all, glob = _ref_lists(bundle, cfg)
    n_citing = int(sum(1 for i in idx if lists[int(i)]))
    rows = (subset_backbone(lists, idx, min_citing=max(2, len(idx) // 50), min_share=0.02, top=30,
                            eps=float(f["lift_epsilon"]), glob=glob, n_all=n_all)
            if n_citing else pd.DataFrame(columns=BACKBONE_COLUMNS))
    works = pd.DataFrame()
    if len(rows) and lookup is not None:
        works = lookup.get(rows["openalex_id"].tolist(), deadline=time.time() + float(f["online_deadline_s"]))
    problems_by_paper: Dict[str, int] = (bundle.problems["arxiv_id"].astype(str).value_counts().to_dict()
                                         if len(getattr(bundle, "problems", [])) else {})
    items = reading_path(bundle.df, idx, bundle.embeddings, works, rows, problems_by_paper,
                         {k: int(v) for k, v in f["path_sizes"].items()}, int(f["recent_months"]),
                         int(f["frontier_months"]), n_base=max(1, n_citing))
    return items, {"reference_coverage": round(n_citing / len(idx), 4), "n_citing": n_citing,
                   "n_collection": int(len(idx))}
