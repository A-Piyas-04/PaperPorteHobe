"""FR-07  Publication-growth analysis.

For each cluster, publication activity over time, a normalised ``relative_growth``
and monthly counts. Growth is always reported with absolute counts so volume
and rate are never conflated (FR-14).

    raw_growth(c)      = (recent_count(c) + 1) / (previous_count(c) + 1)
    relative_growth(c) = raw_growth(c) / raw_growth(all_CS)

>1 means the cluster grew faster than the overall CS corpus across the windows.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .utils import get_logger

log = get_logger("growth")


def _window_counts(dates: pd.Series, ref: pd.Timestamp, months: int) -> tuple[int, int]:
    recent_start = ref - pd.DateOffset(months=months)
    prev_start = ref - pd.DateOffset(months=2 * months)
    recent = int(((dates > recent_start) & (dates <= ref)).sum())
    previous = int(((dates > prev_start) & (dates <= recent_start)).sum())
    return recent, previous


def _raw_growth(recent: int, previous: int) -> float:
    return (recent + 1.0) / (previous + 1.0)


def compute_growth(df: pd.DataFrame, labels: np.ndarray, cfg: Config,
                   reference_date: Optional[str] = None,
                   windows: Optional[List[int]] = None) -> Dict:
    gcfg = cfg["growth"]
    ref_cfg = reference_date or gcfg["reference_date"]
    ref = df["date"].max() if ref_cfg == "auto" else pd.Timestamp(ref_cfg)
    windows = windows or list(gcfg["windows_months"])

    all_dates = df["date"]
    corpus_raw = {w: _raw_growth(*_window_counts(all_dates, ref, w)) for w in windows}

    cluster_ids = sorted({int(c) for c in labels if c != -1})
    clusters: Dict[int, Dict] = {}
    for cid in cluster_ids:
        cdates = df.loc[labels == cid, "date"]
        per_window = {}
        for w in windows:
            recent, previous = _window_counts(cdates, ref, w)
            raw = _raw_growth(recent, previous)
            per_window[str(w)] = {
                "recent_count": recent,
                "previous_count": previous,
                "raw_growth": round(raw, 4),
                "relative_growth": round(raw / corpus_raw[w], 4),
            }
        rel_values = [per_window[str(w)]["relative_growth"] for w in windows]
        clusters[cid] = {
            "cluster_id": cid,
            "reference_date": ref.strftime("%Y-%m-%d"),
            "windows": per_window,
            "monthly_counts": _monthly_counts(cdates),
            "stability": _stability(rel_values),
            "relative_growth_default": per_window[str(gcfg["default_window"])]["relative_growth"]
            if str(gcfg["default_window"]) in per_window else rel_values[0],
        }
    return {
        "reference_date": ref.strftime("%Y-%m-%d"),
        "windows_months": windows,
        "default_window": gcfg["default_window"],
        "corpus_raw_growth": {str(w): round(v, 4) for w, v in corpus_raw.items()},
        "clusters": clusters,
    }


def _monthly_counts(dates: pd.Series) -> Dict[str, int]:
    if len(dates) == 0:
        return {}
    counts = dates.dt.strftime("%Y-%m").value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def _stability(values: List[float]) -> Dict:
    """Coefficient of variation of relative growth across windows; lower = more
    stable. Flags clusters whose growth signal swings across windows."""
    arr = np.array(values, dtype=float)
    mean = float(arr.mean()) if len(arr) else 0.0
    std = float(arr.std()) if len(arr) else 0.0
    cv = std / mean if mean > 0 else float("inf")
    return {
        "relative_growth_values": [round(v, 4) for v in values],
        "mean": round(mean, 4),
        "std": round(std, 4),
        "coef_variation": round(cv, 4) if np.isfinite(cv) else None,
        "stable": bool(np.isfinite(cv) and cv < 0.35),
    }
