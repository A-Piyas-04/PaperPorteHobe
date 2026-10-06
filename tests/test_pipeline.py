"""Integration: the full staged pipeline on the synthetic corpus."""
import json
import os

import pytest

from conftest import make_config
from scholargrid.artifacts import load_bundle
from scholargrid.pipeline import main, run
from scholargrid.release import list_releases


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipe")
    cfg = make_config(tmp)
    meta, report = run(cfg)
    return cfg, meta, report


def test_bundle_shape(built):
    cfg, meta, _ = built
    out = os.listdir(cfg.processed_dir)
    for name in ("papers.parquet", "embeddings.npy", "clusters.json", "growth.json", "meta.json",
                 "bm25_tf.npz", "edges.json", "embedder"):
        assert name in out
    b = load_bundle(cfg)
    assert len(b.df) == meta["counts"]["papers"] == len(b.embeddings)
    assert b.bm25 is not None and b.embedder is not None
    assert meta["counts"]["clusters"] >= 3
    assert meta["snapshot"]["source"] == "synthetic"
    assert {"ingest", "embed", "cluster", "growth"} <= set(meta["stage_timings"])


def test_validation_report_and_release(built):
    cfg, _, report = built
    assert report["search"]["n_queries"] >= 100
    assert set(report["search"]["by_mode"]) == {"configured", "dense", "lexical"}
    assert report["clusters"]["stability"]["mean_ari"] is not None
    assert {"passed", "results"} <= set(report["gates"])
    assert os.path.exists(os.path.join(cfg.reports_dir, "validation_report.md"))
    rel = list_releases(cfg)
    assert len(rel) == 1
    manifest = json.load(open(os.path.join(cfg.releases_dir, rel[0], "manifest.json")))
    assert manifest["corpus"]["n_papers"] > 0 and manifest["libraries"]["numpy"]


def test_single_stage_rerun_uses_cached_inputs(built):
    cfg, meta, _ = built
    _, report = run(cfg, stages=["validate"])
    assert report["data"]["papers"] == meta["counts"]["papers"]


def test_second_run_keeps_cluster_ids(tmp_path):
    cfg = make_config(tmp_path)
    run(cfg)
    meta, _ = run(cfg, stages=["analyze"])
    assert meta["cluster_match"]["kept_fraction"] >= 0.8


def test_cli_rejects_bad_config(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("data:\n  source: nowhere\n", encoding="utf-8")
    with pytest.raises(Exception, match="source"):
        main(["--config", str(bad), "--stage", "ingest"])
