"""The offline demo build produces the whole journey's content."""
from __future__ import annotations

import os
import sys

from conftest import ROOT


def test_demo_build_has_reading_paths_and_gaps(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "app"))
    from scholargrid.artifacts import load_bundle
    from scholargrid.pipeline import run
    from views.onboarding import demo_config

    cfg = demo_config()
    for key in ("raw_dir", "processed_dir", "reports_dir", "stages_dir", "releases_dir", "live_dir"):
        cfg.raw["paths"][key] = str(tmp_path / key)
    cfg.raw["fulltext"]["store_dir"] = str(tmp_path / "fulltext")
    cfg.raw["extract"]["store_dir"] = str(tmp_path / "extract")
    cfg.raw["llm"]["cache_dir"] = str(tmp_path / "llm_cache")
    cfg.raw["llm"]["usage_dir"] = str(tmp_path / "llm_usage")
    cfg.raw["data"]["date_end"] = "2025-12-31"
    cfg.raw["data"]["date_start"] = "2023-01-01"
    run(cfg)
    b = load_bundle(cfg)
    assert b.reading_paths and any(b.reading_paths.values())
    assert sum(len(v) for v in b.gaps["areas"].values()) >= 1
    assert b.meta["snapshot"]["source"] == "synthetic"
