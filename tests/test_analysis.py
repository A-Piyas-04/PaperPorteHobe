"""Clustering, cluster ID matching, labels, enrichment parsing, validation maths."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from scholargrid.cluster_match import match_clusters
from scholargrid.clustering import cluster, cluster_stability, min_cluster_size
from scholargrid.enrich import citation_velocity, is_stale, match_stats, parse_work, refresh_interval_days
from scholargrid.labeling import _c_tf_idf, pretty_label
from scholargrid.validation import evaluate_gates, search_metrics


def _blobs(n=300, k=3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=10, size=(k, 4))
    labels = np.repeat(np.arange(k), n // k)
    return (centers[labels] + rng.normal(size=(len(labels), 4))).astype(np.float32), labels


def test_min_cluster_size_auto(cfg):
    cfg.raw["cluster"].update(min_cluster_size="auto", min_cluster_fraction=0.001)
    assert min_cluster_size(cfg, 50000) == 50
    assert min_cluster_size(cfg, 1000) == 10


def test_kmeans_stability_high_on_clean_blobs(cfg):
    cfg.raw["cluster"]["kmeans_k_range"] = [2, 6]
    x, _ = _blobs()
    labels, backend = cluster(x, cfg)
    assert len(set(labels) - {-1}) >= 3
    st = cluster_stability(x, labels, cfg, backend)
    assert st["mean_ari"] > 0.8 and st["unstable_clusters"] == []


def test_cluster_ids_survive_relabelling():
    x, truth = _blobs()
    ids = pd.Series([f"p{i}" for i in range(len(truth))])
    previous = pd.DataFrame({"arxiv_id": ids, "cluster_id": truth + 10})
    shuffled = np.array([2, 0, 1])[truth]
    out = match_clusters(ids, shuffled, x, previous)
    assert (out["labels"] == truth + 10).all()
    assert out["report"]["kept_fraction"] == 1.0

    split = truth.copy()
    split[(truth == 0) & (np.arange(len(truth)) % 2 == 0)] = 5
    rep = match_clusters(ids, split, x, previous)["report"]
    assert rep["new_clusters"] == 1 and rep["splits"]


def test_pretty_label_and_ctfidf():
    assert pretty_label(["kv", "cache", "kv cache", "llm"]) == "KV Cache & LLM"
    assert pretty_label([]) == "Unlabeled area"
    scores = _c_tf_idf(np.array([[5.0, 0.0, 1.0], [0.0, 5.0, 1.0]]))
    assert scores[0, 0] > scores[0, 2] and scores[1, 1] > scores[1, 2]


def test_parse_openalex_work_fixture():
    work = {"id": "https://openalex.org/W1", "cited_by_count": 12,
            "counts_by_year": [{"year": 2025, "cited_by_count": 9}],
            "referenced_works": ["https://openalex.org/W2"],
            "primary_location": {"source": {"display_name": "NeurIPS", "type": "conference"}},
            "authorships": [{"author": {"id": "https://openalex.org/A9"}}],
            "primary_topic": {"display_name": "Graph learning"}, "type": "article"}
    rec = parse_work(work)
    assert rec["openalex_id"] == "W1" and rec["venue"] == "NeurIPS"
    assert rec["references"] == ["W2"] and rec["author_ids"] == ["A9"]
    assert rec["counts_by_year"] == {"2025": 9}
    arxiv = parse_work({**work, "primary_location": {"source": {"display_name": "arXiv (Cornell)"}}})
    assert arxiv["venue"] == "" and arxiv["venue_type"] == "arxiv-only"


def test_refresh_schedule_by_age(cfg):
    sched = cfg["enrich"]["refresh_schedule"]
    today = date(2026, 1, 1)
    assert refresh_interval_days(pd.Timestamp("2025-12-01"), today, sched) == 7
    assert refresh_interval_days(pd.Timestamp("2025-01-01"), today, sched) == 30
    assert refresh_interval_days(pd.Timestamp("2020-01-01"), today, sched) == 90
    e = cfg["enrich"]
    hit = {"at": "2025-12-20", "rec": {"cited_by_count": 1}}
    assert is_stale(hit, pd.Timestamp("2025-12-01"), today, e)        # young: weekly
    assert not is_stale(hit, pd.Timestamp("2020-01-01"), today, e)    # old: quarterly
    assert is_stale({"at": "2025-12-30", "rec": None}, pd.Timestamp("2020-01-01"), today, e)
    assert is_stale(None, pd.Timestamp("2020-01-01"), today, e)


def test_citation_velocity_and_match_stats():
    h = pd.DataFrame({"arxiv_id": ["a", "a", "b"], "snapshot_date": ["2026-01-01", "2026-01-31", "2026-01-31"],
                      "cited_by_count": [10, 40, 3]})
    v = citation_velocity(h)
    assert v == {"a": pytest.approx(30.0)}
    df = pd.DataFrame({"openalex_id": ["W1", "", "W3", "W4"], "reference_count": [3, 0, 0, 5],
                       "date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-12-01", "2025-12-15"]),
                       "year_month": ["2025-01", "2025-02", "2025-12", "2025-12"],
                       "primary_category": ["cs.LG"] * 4})
    s = match_stats(df)
    assert s["match_rate"] == 0.75 and s["match_rate_older_3m"] == 0.5
    assert s["reference_coverage_older_6m"] == 0.5


def test_search_metrics_known_values():
    grades = np.array([2, 0, 1, 0, 2, 0])
    m = search_metrics([0, 4, 2, 1], grades, k=3, recall_k=4)
    assert m["ndcg_at_k"] == pytest.approx(1.0)
    assert m["precision_at_k"] == 1.0 and m["mrr"] == 1.0 and m["recall_at_k"] == 1.0
    worse = search_metrics([1, 3, 0], grades, k=3, recall_k=3)
    assert worse["mrr"] == pytest.approx(1 / 3, abs=1e-3) and worse["ndcg_at_k"] < 0.5


def test_gates_required_vs_advisory(cfg):
    cfg.raw["validation"].update(gates={"search_ndcg_at_10": 0.75, "max_noise_fraction": 0.3},
                                 required_gates=["search_ndcg_at_10"])
    report = {"search": {"ndcg_at_10": 0.8}, "clusters": {"noise_fraction": 0.5},
              "data": {"quality": {"passed": True}}}
    g = evaluate_gates(cfg, report)
    assert g["passed"]
    assert any(r["gate"] == "max_noise_fraction" and not r["passed"] for r in g["results"])
    report["search"]["ndcg_at_10"] = 0.5
    assert evaluate_gates(cfg, report)["failed_required"] == ["search_ndcg_at_10"]
