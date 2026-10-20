"""Topic workspace: k selection, merging, micro-topics, keyword labels, build + cache, LLM fallback."""
from __future__ import annotations

import os
import time

import numpy as np
import pytest
from conftest import make_config

from scholargrid.workspace.labels import keyword_labels
from scholargrid.workspace.subtopics import choose_k, merge_small, subtopics


def _blobs(n_per=60, dirs=3, dim=16, noise=0.08, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.eye(dim)[:dirs]
    x = np.vstack([c + noise * rng.standard_normal((n_per, dim)) for c in centers])
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_choose_k_and_merge_small():
    x = _blobs()
    k, labels, sil = choose_k(x, (2, 6), 0)
    assert k == 3 and sil > 0.5
    labels = np.array([0] * 20 + [1] * 20 + [2] * 2)
    pts = np.vstack([_blobs(20, 2)[:40], _blobs(1, 1)[:1].repeat(2, axis=0)])
    merged = merge_small(pts, labels, 5)
    assert len(np.unique(merged)) == 2


def test_subtopics_micro_ids_do_not_collide(tmp_path):
    cfg = make_config(tmp_path)
    cfg.raw["workspace"].update(subtopic_k_range=[2, 3], micro_min_size=40, min_subtopic_size=8)
    # two macro groups, each made of two tight sub-groups -> micro-topics
    rng = np.random.default_rng(1)
    dim = 16
    base = np.eye(dim)
    parts = []
    for a, b in ((0, 1), (2, 3)):
        for c in (a, b):
            parts.append(base[a] * 3 + base[c + 8] + 0.05 * rng.standard_normal((50, dim)))
    x = np.vstack(parts)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    top, micro, info = subtopics(x, cfg)
    assert info["k"] >= 2
    tops = set(np.unique(top))
    micros = set(np.unique(micro[micro >= 0]))
    assert micros and not (tops & micros) and min(micros) > max(tops)
    cfg.raw["workspace"]["min_silhouette_micro"] = 0.99
    _, micro2, _ = subtopics(x, cfg)
    assert (micro2 == -1).all()


def test_keyword_labels_exclude_query_words():
    docs = {0: "graph neural networks for molecules molecules chemistry drug discovery",
            1: "graph neural networks for traffic forecasting traffic roads sensors"}
    kws = keyword_labels(docs, "graph neural networks")
    assert all("graph" not in k and "neural" not in k and "network" not in k for ks in kws.values() for k in ks)
    assert any("molecule" in k for k in kws[0]) and any("traffic" in k for k in kws[1])


@pytest.fixture(scope="module")
def bundle_cfg(tmp_path_factory):
    from scholargrid.artifacts import load_bundle
    from scholargrid.pipeline import run

    cfg = make_config(tmp_path_factory.mktemp("ws"))
    run(cfg)
    return cfg, load_bundle(cfg)


def _build(cfg, bundle, llm=None, **kw):
    from scholargrid.workspace import build_workspace

    return build_workspace("language models", bundle, None, cfg, bundle_version="v-test", llm=llm, **kw)


def test_build_workspace_with_fake_llm(bundle_cfg):
    from scholargrid.llm import FakeLLM

    cfg, bundle = bundle_cfg
    llm = FakeLLM(cfg)
    ws = _build(cfg, bundle, llm=llm, use_cache=False)
    assert len(ws.top_level()) >= 2
    assert all(p.subtopic >= 0 for p in ws.papers if p.bundle_index is not None)
    assert ws.llm_used and all(s.summary.startswith("Papers about") for s in ws.subtopics)
    assert ws.overview["summary"].startswith("An overview of language models")
    assert ws.reading_path
    assert {"retrieve", "subtopics", "labels", "activity", "overview", "reading_path"} <= set(ws.stats["timings"])
    assert all("abstract" not in it for it in ws.reading_path)


def test_cache_hit_and_expiry(bundle_cfg):
    from scholargrid.llm import FakeLLM

    cfg, bundle = bundle_cfg
    first = _build(cfg, bundle, llm=FakeLLM(cfg))
    llm = FakeLLM(cfg)
    again = _build(cfg, bundle, llm=llm)
    assert again.key == first.key and not llm.calls
    path = os.path.join(cfg.workspace_dir, f"{first.key}.json")
    old = time.time() - 100 * 3600
    os.utime(path, (old, old))
    _build(cfg, bundle, llm=llm)
    assert llm.calls


def test_llm_unavailable_falls_back_to_keywords(bundle_cfg):
    from scholargrid.llm import NullLLM

    cfg, bundle = bundle_cfg
    ws = _build(cfg, bundle, llm=NullLLM(cfg), use_cache=False)
    assert not ws.llm_used and all(s.name == s.keyword_label and not s.summary for s in ws.subtopics)
    assert any("summaries are unavailable" in w for w in ws.warnings)


def test_quota_exhausted_is_not_cached(bundle_cfg):
    from scholargrid.llm import FakeLLM, QuotaExhausted

    cfg, bundle = bundle_cfg

    class NoQuota(FakeLLM):
        def structured_many(self, task, prompt, schema, items, **kw):
            return {iid: QuotaExhausted("daily") for iid, _ in items}

    cfg.raw["workspace"]["max_papers"] = 450          # different key than the cached workspaces above
    try:
        ws = _build(cfg, bundle, llm=NoQuota(cfg))
        assert ws.stats["labels_pending"] and any("quota" in w for w in ws.warnings)
        assert not os.path.exists(os.path.join(cfg.workspace_dir, f"{ws.key}.json"))
    finally:
        cfg.raw["workspace"]["max_papers"] = 500


def test_json_round_trip(bundle_cfg):
    import json

    from scholargrid.llm import FakeLLM
    from scholargrid.workspace import TopicWorkspace

    cfg, bundle = bundle_cfg
    ws = _build(cfg, bundle, llm=FakeLLM(cfg), use_cache=False)
    again = TopicWorkspace.from_json(json.loads(json.dumps(ws.to_json())))
    assert again == ws
