import pandas as pd
import pytest

from scholargrid.quality import DataQualityError, run_checks
from scholargrid.storage import PaperStore


def _papers(ids, when="2024-03-05", version=1, title="t"):
    return pd.DataFrame({"arxiv_id": ids, "version": version, "title": [f"{title} {i}" for i in ids],
                         "abstract": "a" * 300, "authors": "", "categories": "cs.LG",
                         "primary_category": "cs.LG", "first_submitted": when, "last_updated": when})


@pytest.mark.parametrize("backend", ["parquet", "duckdb"])
def test_store_upsert_dedups_and_reads_window(cfg, backend):
    cfg.raw["storage"]["backend"] = backend
    store = PaperStore(cfg)
    store.upsert(_papers(["a", "b"], when="2024-03-05"))
    store.upsert(_papers(["b"], when="2024-03-05", version=2, title="updated"))
    store.upsert(_papers(["c"], when="2024-05-01"))
    all_rows = store.read()
    assert sorted(all_rows["arxiv_id"]) == ["a", "b", "c"]
    assert all_rows.set_index("arxiv_id").loc["b", "title"] == "updated b"
    march = store.read("2024-03-01", "2024-03-31")
    assert sorted(march["arxiv_id"]) == ["a", "b"]


def test_citation_history_appends_and_dedups(cfg):
    store = PaperStore(cfg)
    store.append_citations(pd.DataFrame({"arxiv_id": ["a"], "snapshot_date": ["2024-01-01"],
                                         "cited_by_count": [1]}))
    store.append_citations(pd.DataFrame({"arxiv_id": ["a", "a"], "snapshot_date": ["2024-01-01", "2024-02-01"],
                                         "cited_by_count": [2, 5]}))
    h = store.citation_history().sort_values("snapshot_date")
    assert list(h["cited_by_count"]) == [2, 5]


def test_quality_checks_pass_and_fail(cfg, papers):
    cfg.raw["quality"].update(min_span_months=12, min_papers_per_month=5, max_category_share=0.5)
    ok = run_checks(papers, cfg, previous_count=410)
    assert ok["passed"], ok["failed"]

    cfg.raw["quality"]["min_span_months"] = 60
    bad = run_checks(papers, cfg, previous_count=1000)
    assert set(bad["failed"]) == {"time_span_months", "row_count_change"}

    cfg.raw["quality"]["fail_on_error"] = True
    with pytest.raises(DataQualityError):
        run_checks(papers, cfg)


def test_quality_detects_month_gaps_and_duplicates(cfg, papers):
    cfg.raw["quality"]["min_papers_per_month"] = 1
    gappy = papers[~papers["year_month"].isin(["2023-06", "2023-07"])]
    dup = pd.concat([gappy, gappy.head(1)])
    res = run_checks(dup, cfg)
    assert "min_papers_per_month" in res["failed"]
    assert "duplicates_after_dedup" in res["failed"]
