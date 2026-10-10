"""Shared fixtures: a fast, offline, fully isolated configuration."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scholargrid.config import Config, load_config  # noqa: E402


def make_config(tmp_path, **overrides) -> Config:
    """Synthetic, scikit-learn-only config writing everything under tmp_path."""
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    raw = cfg.raw
    for key in ("raw_dir", "processed_dir", "reports_dir", "stages_dir", "releases_dir", "live_dir"):
        raw["paths"][key] = str(tmp_path / key)
    raw["search"]["live"] = False
    raw["data"].update(source="synthetic", profile="custom", max_papers=600,
                       date_start="2022-01-01", date_end="2025-12-31", language_filter=True)
    raw["enrich"]["openalex"] = False
    raw["embedding"]["backend"] = "tfidf"
    raw["reduce"]["backend"] = "pca"
    raw["landscape"]["backend"] = "pca"
    raw["cluster"]["backend"] = "kmeans"
    raw["cluster"]["stability_seeds"] = [0, 1]
    raw["sparse"]["min_corpus_size"] = 100
    raw["sparse"]["robustness_seeds"] = [0]
    raw["growth"]["bootstrap_samples"] = 200
    raw["growth"]["min_papers_per_period"] = 10
    raw["storage"]["path"] = str(tmp_path / "store.duckdb")
    for section, values in overrides.items():
        if isinstance(values, dict):
            raw[section].update(values)
        else:
            raw[section] = values
    return cfg


@pytest.fixture
def cfg(tmp_path) -> Config:
    return make_config(tmp_path)


@pytest.fixture
def papers() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", "2024-12-31", periods=400)
    cats = rng.choice(["cs.LG", "cs.CL", "cs.CV"], size=len(dates))
    return pd.DataFrame({
        "arxiv_id": [f"2301.{i:05d}" for i in range(len(dates))],
        "title": [f"Paper {i} about {c}" for i, c in enumerate(cats)],
        "abstract": ["We study a method for learning representations. " * 6] * len(dates),
        "authors": ["A. Author"] * len(dates),
        "categories": [f"{c} cs.AI" for c in cats],
        "primary_category": cats,
        "date": dates,
        "year_month": dates.strftime("%Y-%m"),
        "sample_weight": 1.0,
    })
