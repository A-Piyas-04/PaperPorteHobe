import os

import numpy as np
import pytest

from scholargrid.embeddings import Embedder, _EmbeddingCache
from scholargrid.release import PublishError, list_releases, prune, publish, resolve_bundle_dir, set_current


def test_tfidf_roundtrip_without_pickle(cfg, tmp_path):
    texts = [f"graph neural network paper number {i} about molecules" for i in range(30)]
    texts += [f"diffusion model image synthesis paper {i}" for i in range(30)]
    emb = Embedder("tfidf", cfg)
    vecs = emb.fit_transform(texts)
    emb.save(str(tmp_path / "e"))
    files = set(os.listdir(tmp_path / "e"))
    assert files == {"embedder_meta.json", "tfidf.joblib"}
    loaded = Embedder.load(str(tmp_path / "e"), cfg)
    assert np.allclose(loaded.transform(texts[:3]), vecs[:3], atol=1e-5)


def test_embedding_cache_reuses_unchanged_texts(cfg):
    calls = []

    def fn(texts):
        calls.append(len(texts))
        return np.ones((len(texts), 4), dtype=np.float32) * len(calls)

    cache = _EmbeddingCache(cfg, "model@rev")
    first = cache.encode(["a", "b"], ["text a", "text b"], fn)
    second = cache.encode(["a", "b", "c"], ["text a", "text b CHANGED", "text c"], fn)
    assert calls == [2, 2]
    assert np.allclose(second[0], first[0])
    assert np.allclose(second[1:], 2.0)


def _stage_bundle(cfg, n=3):
    os.makedirs(cfg.processed_dir, exist_ok=True)
    import pandas as pd

    pd.DataFrame({"arxiv_id": [f"p{i}" for i in range(n)], "date": ["2024-01-01"] * n,
                  "cluster_id": [0] * n}).to_parquet(os.path.join(cfg.processed_dir, "papers.parquet"))
    with open(os.path.join(cfg.processed_dir, "meta.json"), "w") as fh:
        fh.write("{}")


def test_publish_versions_prunes_and_rolls_back(cfg):
    cfg.raw["release"]["keep_last"] = 2
    _stage_bundle(cfg)
    meta = {"snapshot": {"n_papers": 3}, "counts": {"clusters": 1}}
    passing = {"gates": {"passed": True}}
    ids = [publish(cfg, meta, passing) for _ in range(3)]
    assert len(set(ids)) == 3
    assert list_releases(cfg) == sorted(ids)[-2:]
    assert resolve_bundle_dir(cfg).endswith(ids[-1])
    set_current(cfg, ids[1])
    assert resolve_bundle_dir(cfg).endswith(ids[1])
    cfg.raw["release"]["pin"] = ids[2]
    assert resolve_bundle_dir(cfg).endswith(ids[2])
    assert prune(cfg) == []


def test_publish_refuses_failing_gates_when_enforced(cfg):
    _stage_bundle(cfg)
    cfg.raw["validation"]["enforce_gates"] = True
    with pytest.raises(PublishError):
        publish(cfg, {}, {"gates": {"passed": False, "failed_required": ["search_ndcg_at_10"]}})
    assert list_releases(cfg) == []
    assert resolve_bundle_dir(cfg) == cfg.processed_dir


def test_missing_pin_is_an_error(cfg):
    cfg.raw["release"]["pin"] = "1999.01.01"
    with pytest.raises(FileNotFoundError):
        resolve_bundle_dir(cfg)
