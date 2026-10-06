import numpy as np
import pandas as pd

from scholargrid.growth import backtest, bootstrap_ci, compute_growth, poisson_trend


def _corpus(n_months=36, growing_rate=1.06, base=60, flat=200, seed=0):
    """Cluster 0 grows ~6%/month; cluster 1 is flat."""
    rng = np.random.default_rng(seed)
    rows = []
    months = pd.date_range("2022-01-01", periods=n_months, freq="MS")
    for m, start in enumerate(months):
        for cid, lam in ((0, base * growing_rate ** m), (1, flat)):
            for _ in range(rng.poisson(lam)):
                rows.append({"date": start + pd.Timedelta(days=int(rng.integers(0, 28))), "cid": cid})
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["year_month"] = df["date"].dt.strftime("%Y-%m")
    return df, df["cid"].to_numpy()


def test_growing_cluster_is_reliable_and_flat_is_not_growing(cfg):
    df, labels = _corpus()
    g = compute_growth(df, labels, cfg)
    assert g["sufficient_history"]
    up = g["clusters"][0]["windows"]["6"]
    flat = g["clusters"][1]["windows"]["6"]
    assert up["passes_gate"] and up["relative_growth"] > 1.05
    assert up["ci"][0] > 1 and up["direction"] == "growing"
    assert g["clusters"][0]["reliable"]
    assert flat["relative_growth"] < 1 and flat["direction"] == "declining"
    assert g["clusters"][0]["trend"]["annual_rate_ratio"] > 1.2


def test_small_clusters_get_no_ratio(cfg):
    df, labels = _corpus(base=1, flat=200)
    cfg.raw["growth"]["min_papers_per_period"] = 30
    g = compute_growth(df, labels, cfg)
    w = g["clusters"][0]["windows"]["3"]
    assert not w["passes_gate"]
    assert w["relative_growth"] is None and w["raw_growth"] is None
    assert not g["clusters"][0]["reliable"]


def test_insufficient_history_blocks_growth(cfg):
    df, labels = _corpus(n_months=8)
    g = compute_growth(df, labels, cfg)
    assert not g["sufficient_history"]
    assert all(c["windows"]["12"]["relative_growth"] is None for c in g["clusters"].values())


def test_sample_weights_restore_population_ratio(cfg):
    df, labels = _corpus()
    recent = df["date"] > df["date"].max() - pd.DateOffset(months=6)
    keep = ~(recent & (df["cid"] == 1)) | (np.arange(len(df)) % 2 == 0)
    sampled = df[keep].copy()
    sampled["sample_weight"] = np.where(recent[keep] & (sampled["cid"] == 1), 2.0, 1.0)
    full = compute_growth(df, labels, cfg, with_cutoffs=False)
    weighted = compute_growth(sampled.reset_index(drop=True), sampled["cid"].to_numpy(), cfg,
                              with_cutoffs=False)
    a = full["clusters"][1]["windows"]["6"]["relative_growth"]
    b = weighted["clusters"][1]["windows"]["6"]["relative_growth"]
    assert abs(a - b) / a < 0.1


def test_bootstrap_ci_brackets_truth():
    rng = np.random.default_rng(0)
    lo, hi = bootstrap_ci(200, 100, 1000, 1000, 2000, 0.9, rng)
    assert lo < 2.0 < hi and lo > 1.0


def test_poisson_trend_needs_data():
    idx = pd.Index(["2024-01", "2024-02"])
    assert poisson_trend(pd.Series([1, 2], index=idx), pd.Series([10, 10], index=idx)) is None


def test_backtest_reports_hit_rate(cfg):
    df, labels = _corpus(n_months=40)
    bt = backtest(df, labels, cfg)
    assert bt["available"]
    assert bt["flagged_growing"] >= 1
    assert bt["hit_rate"] == 1.0
    short, short_labels = _corpus(n_months=12)
    assert not backtest(short, short_labels, cfg)["available"]
