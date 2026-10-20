"""Scope selection: categories, min_date, max_papers and the three priorities."""
from __future__ import annotations

import pandas as pd
from conftest import make_config

from scholargrid.scope import scope_mask, select_scope


def _df():
    return pd.DataFrame({
        "arxiv_id": ["a", "b", "c", "d"],
        "primary_category": ["cs.CL", "cs.CL", "cs.LG", "cs.CL"],
        "date": pd.to_datetime(["2023-01-01", "2024-06-01", "2024-01-01", "2022-01-01"]),
        "cited_by_count": [5, 50, 100, 1],
    })


def test_categories_and_min_date(tmp_path):
    cfg = make_config(tmp_path, scope={"categories": ["cs.CL"], "min_date": "2023-01-01"})
    assert scope_mask(_df(), cfg).tolist() == [True, True, False, False]


def test_most_cited_and_cap(tmp_path):
    cfg = make_config(tmp_path, scope={"priority": "most_cited", "max_papers": 2})
    assert select_scope(_df(), cfg)["arxiv_id"].tolist() == ["c", "b"]


def test_newest(tmp_path):
    cfg = make_config(tmp_path, scope={"priority": "newest"})
    assert select_scope(_df(), cfg)["arxiv_id"].tolist() == ["b", "c", "a", "d"]


def test_random_is_seeded(tmp_path):
    cfg = make_config(tmp_path, scope={"priority": "random"})
    first = select_scope(_df(), cfg)["arxiv_id"].tolist()
    assert sorted(first) == ["a", "b", "c", "d"] and first == select_scope(_df(), cfg)["arxiv_id"].tolist()


def test_most_cited_without_citations_falls_back_to_newest(tmp_path):
    cfg = make_config(tmp_path, scope={"priority": "most_cited"})
    df = _df().drop(columns=["cited_by_count"])
    assert select_scope(df, cfg)["arxiv_id"].tolist() == ["b", "c", "a", "d"]
