"""FR-07  Publication-growth analysis.

For each cluster and window ``w`` (months) the recent period ``(ref-w, ref]``
is compared with the previous one ``(ref-2w, ref-w]``:

    raw_growth(c)      = recent(c) / previous(c)
    relative_growth(c) = raw_growth(c) / raw_growth(all papers)

Counts are weighted by ``sample_weight`` so stratified samples estimate the
full corpus. No ``+1`` smoothing: a growth value is only produced when

* the corpus covers both periods (``sufficient``), and
* the cluster has at least ``min_papers_per_period`` papers in each period.

A Poisson bootstrap gives a confidence interval; a cluster is only called
"growing" when the lower bound is above 1 (``declining`` when the upper bound
is below 1). Growth must also be stable across alternative reference dates
(coefficient of variation <= ``max_cv``) to be marked ``reliable``. A
Poisson log-linear trend on monthly counts (offset by the corpus) gives a
second, window-free estimate. Absolute counts are always reported (FR-14).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .utils import get_logger

log = get_logger("growth")


def _weights(df: pd.DataFrame) -> pd.Series:
    if "sample_weight" in df.columns:
        return df["sample_weight"].astype(float).fillna(1.0)
    return pd.Series(1.0, index=df.index)


def _period_masks(dates: pd.Series, ref: pd.Timestamp, months: int):
    recent_start = ref - pd.DateOffset(months=months)
    prev_start = ref - pd.DateOffset(months=2 * months)
    recent = (dates > recent_start) & (dates <= ref)
    previous = (dates > prev_start) & (dates <= recent_start)
    return recent, previous


def window_counts(dates: pd.Series, ref: pd.Timestamp, months: int,
                  weights: Optional[pd.Series] = None) -> Dict[str, float]:
    recent, previous = _period_masks(dates, ref, months)
    w = weights if weights is not None else pd.Series(1.0, index=dates.index)
    return {"recent": int(recent.sum()), "previous": int(previous.sum()),
            "recent_w": float(w[recent].sum()), "previous_w": float(w[previous].sum())}


def bootstrap_ci(rec_c: float, prev_c: float, rec_all: float, prev_all: float,
                 n: int, level: float, rng: np.random.Generator) -> Optional[List[float]]:
    """Percentile CI of relative growth under independent Poisson counts."""
    if n <= 0 or prev_c <= 0 or prev_all <= 0:
        return None
    rc = rng.poisson(rec_c, n)
    pc = rng.poisson(prev_c, n)
    ra = rng.poisson(rec_all, n)
    pa = rng.poisson(prev_all, n)
    ok = (pc > 0) & (pa > 0) & (ra > 0)
    if ok.sum() < n * 0.5:
        return None
    rel = (rc[ok] / pc[ok]) / (ra[ok] / pa[ok])
    lo, hi = np.quantile(rel, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [round(float(lo), 4), round(float(hi), 4)]


def poisson_trend(cluster_monthly: pd.Series, corpus_monthly: pd.Series,
                  level: float = 0.9) -> Optional[Dict]:
    """Fit log E[y_t] = a + b*t + log(N_t) by IRLS; report the annual rate
    ratio exp(12b) relative to the corpus, with a Wald interval."""
    months = corpus_monthly.index
    y = cluster_monthly.reindex(months, fill_value=0).to_numpy(dtype=float)
    n = corpus_monthly.to_numpy(dtype=float)
    keep = n > 0
    if keep.sum() < 6 or y[keep].sum() < 10:
        return None
    y, n = y[keep], n[keep]
    t = np.arange(len(y), dtype=float)
    t -= t.mean()
    X = np.column_stack([np.ones_like(t), t])
    offset = np.log(n)
    beta = np.array([np.log(max(y.sum(), 1) / n.sum()), 0.0])
    for _ in range(50):
        mu = np.exp(X @ beta + offset)
        z = X @ beta + (y - mu) / np.maximum(mu, 1e-9)
        W = mu
        XtW = X.T * W
        try:
            new = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            return None
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    mu = np.exp(X @ beta + offset)
    try:
        cov = np.linalg.inv((X.T * mu) @ X)
    except np.linalg.LinAlgError:
        return None
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    from scipy.stats import norm

    zq = float(norm.ppf(1 - (1 - level) / 2))
    b = float(beta[1])
    return {"annual_rate_ratio": round(float(np.exp(12 * b)), 4),
            "ci": [round(float(np.exp(12 * (b - zq * se))), 4), round(float(np.exp(12 * (b + zq * se))), 4)],
            "months": int(len(y))}


def _direction(ci: Optional[List[float]]) -> Optional[str]:
    if not ci:
        return None
    if ci[0] > 1:
        return "growing"
    if ci[1] < 1:
        return "declining"
    return "no clear change"


def _cv(values: List[float]) -> Optional[float]:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if len(arr) < 2 or arr.mean() <= 0:
        return None
    return float(arr.std() / arr.mean())


def compute_growth(df: pd.DataFrame, labels: np.ndarray, cfg: Config,
                   reference_date: Optional[str] = None,
                   windows: Optional[List[int]] = None,
                   with_cutoffs: bool = True) -> Dict:
    g = cfg["growth"]
    ref_cfg = reference_date or g["reference_date"]
    ref = df["date"].max() if ref_cfg == "auto" else pd.Timestamp(ref_cfg)
    windows = windows or list(g["windows_months"])
    dw = int(g["default_window"])
    min_n = int(g["min_papers_per_period"])
    rng = np.random.default_rng(cfg["seed"])
    weights = _weights(df)

    dates = df["date"]
    earliest = dates.min() if len(dates) else ref
    history_days = int((ref - earliest).days) + 1 if len(dates) else 0
    sufficient = {w: earliest <= ref - pd.DateOffset(months=2 * w) for w in windows}
    corpus = {w: window_counts(dates, ref, w, weights) for w in windows}
    corpus_raw = {w: (c["recent_w"] / c["previous_w"]) if c["previous_w"] > 0 else None
                  for w, c in corpus.items()}
    corpus_monthly = weights.groupby(df["year_month"]).sum().sort_index()

    cutoff_values: Dict[int, List[Optional[float]]] = {}
    if with_cutoffs:
        for months_back in cfg["validation"]["alternate_cutoffs_months"]:
            alt_ref = ref - pd.DateOffset(months=int(months_back))
            alt = compute_growth(df[df["date"] <= alt_ref], labels[(df["date"] <= alt_ref).to_numpy()],
                                 cfg, reference_date=alt_ref.strftime("%Y-%m-%d"),
                                 windows=[dw], with_cutoffs=False) if (df["date"] <= alt_ref).any() else None
            for cid, info in (alt or {}).get("clusters", {}).items():
                cutoff_values.setdefault(int(cid), []).append(info["windows"][str(dw)]["relative_growth"])

    cluster_ids = sorted({int(c) for c in labels if c != -1})
    clusters: Dict[int, Dict] = {}
    for cid in cluster_ids:
        mask = labels == cid
        cdates, cw = dates[mask], weights[mask]
        per_window = {}
        for w in windows:
            c = window_counts(cdates, ref, w, cw)
            gate = bool(sufficient[w]) and c["recent"] >= min_n and c["previous"] >= min_n
            raw = c["recent_w"] / c["previous_w"] if c["previous_w"] > 0 else None
            rel = (raw / corpus_raw[w]) if (gate and raw is not None and corpus_raw[w]) else None
            ci = (bootstrap_ci(c["recent_w"], c["previous_w"], corpus[w]["recent_w"],
                               corpus[w]["previous_w"], int(g["bootstrap_samples"]),
                               float(g["ci_level"]), rng) if rel is not None else None)
            per_window[str(w)] = {
                "recent_count": c["recent"], "previous_count": c["previous"],
                "recent_weighted": round(c["recent_w"], 2), "previous_weighted": round(c["previous_w"], 2),
                "raw_growth": round(raw, 4) if raw is not None and gate else None,
                "relative_growth": round(rel, 4) if rel is not None else None,
                "ci": ci,
                "direction": _direction(ci),
                "sufficient": bool(sufficient[w]),
                "passes_gate": gate,
            }
        default = per_window.get(str(dw)) or per_window[str(windows[0])]
        window_values = [per_window[str(w)]["relative_growth"] for w in windows]
        cut_vals = [default["relative_growth"]] + cutoff_values.get(cid, [])
        cv_windows, cv_cutoffs = _cv(window_values), _cv(cut_vals)
        stable = (cv_cutoffs is not None and cv_cutoffs <= float(g["max_cv"])
                  and (cv_windows is None or cv_windows <= float(g["max_cv"])))
        reliable = bool(default["passes_gate"] and default["ci"] is not None
                        and (stable or not g["require_stable"]))
        monthly_w = cw.groupby(df.loc[mask, "year_month"]).sum().sort_index()
        clusters[cid] = {
            "cluster_id": cid,
            "reference_date": ref.strftime("%Y-%m-%d"),
            "windows": per_window,
            "monthly_counts": {str(k): int(v) for k, v in cdates.dt.strftime("%Y-%m").value_counts().sort_index().items()},
            "monthly_weighted": {str(k): round(float(v), 2) for k, v in monthly_w.items()},
            "trend": poisson_trend(monthly_w, corpus_monthly, float(g["ci_level"])),
            "stability": {
                "relative_growth_values": window_values,
                "cutoff_values": cut_vals,
                "cv_windows": round(cv_windows, 4) if cv_windows is not None else None,
                "cv_cutoffs": round(cv_cutoffs, 4) if cv_cutoffs is not None else None,
                "stable": bool(stable),
            },
            "reliable": reliable,
            "relative_growth_default": default["relative_growth"],
        }

    n_reliable = sum(1 for c in clusters.values() if c["reliable"])
    return {
        "reference_date": ref.strftime("%Y-%m-%d"),
        "windows_months": windows,
        "default_window": dw,
        "corpus_raw_growth": {str(w): (round(v, 4) if v is not None else None) for w, v in corpus_raw.items()},
        "history_days": history_days,
        "sufficient_history": bool(sufficient.get(dw, False)),
        "gate": {"min_papers_per_period": min_n, "ci_level": g["ci_level"],
                 "max_cv": g["max_cv"], "require_stable": g["require_stable"]},
        "n_reliable": n_reliable,
        "clusters": clusters,
    }


def backtest(df: pd.DataFrame, labels: np.ndarray, cfg: Config) -> Dict:
    """Compute growth as of ``backtest_months`` ago and check whether clusters
    flagged "growing" then actually grew faster than the corpus afterwards."""
    g = cfg["growth"]
    dw = int(g["default_window"])
    months = int(g["backtest_months"])
    if df.empty:
        return {"available": False, "reason": "empty corpus"}
    ref = df["date"].max() - pd.DateOffset(months=months)
    if df["date"].min() > ref - pd.DateOffset(months=2 * dw):
        return {"available": False,
                "reason": f"needs {months + 2 * dw} months of history"}
    past_mask = (df["date"] <= ref).to_numpy()
    past = compute_growth(df[past_mask], labels[past_mask], cfg,
                          reference_date=ref.strftime("%Y-%m-%d"), windows=[dw], with_cutoffs=False)
    flagged = [cid for cid, c in past["clusters"].items()
               if c["windows"][str(dw)]["direction"] == "growing"]
    weights = _weights(df)
    after = (df["date"] > ref) & (df["date"] <= ref + pd.DateOffset(months=months))
    before = (df["date"] > ref - pd.DateOffset(months=months)) & (df["date"] <= ref)
    total_after, total_before = float(weights[after].sum()), float(weights[before].sum())
    hits = []
    for cid in flagged:
        m = labels == cid
        a, b = float(weights[after & m].sum()), float(weights[before & m].sum())
        if b > 0 and total_before > 0 and total_after > 0:
            hits.append((a / total_after) / (b / total_before) > 1.0)
    return {
        "available": True,
        "reference_date": ref.strftime("%Y-%m-%d"),
        "flagged_growing": len(flagged),
        "evaluated": len(hits),
        "hit_rate": round(float(np.mean(hits)), 4) if hits else None,
    }
