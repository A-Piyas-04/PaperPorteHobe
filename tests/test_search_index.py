import numpy as np
import pandas as pd
import pytest

from scholargrid.bm25 import BM25Index, reciprocal_rank_fusion
from scholargrid.index import NeighborIndex, exact_topk
from scholargrid.search import SearchFilters, search
from scholargrid.utils import cosine_topk, l2_normalize


class FakeEmbedder:
    def __init__(self, vectors, texts):
        self.vectors, self.texts = vectors, texts

    def encode_query(self, q):
        hits = [i for i, t in enumerate(self.texts) if q.split()[0].lower() in t.lower()]
        return self.vectors[hits[0]] if hits else self.vectors[0]


class BrokenEmbedder:
    def encode_query(self, q):
        raise RuntimeError("model offline")


@pytest.fixture
def corpus():
    texts = ["GPT-4 evaluation on reasoning benchmarks", "Diffusion models for image synthesis",
             "Graph neural networks for molecules", "Federated learning with differential privacy",
             "Reinforcement learning for robot grasping", "Vision transformers for detection"]
    df = pd.DataFrame({
        "arxiv_id": [f"x{i}" for i in range(6)], "title": texts, "abstract": texts,
        "primary_category": ["cs.CL", "cs.CV", "cs.LG", "cs.CR", "cs.RO", "cs.CV"],
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01",
                                "2024-05-01", "2024-06-01"]),
        "cited_by_count": [50, 3, 10, 0, 7, 100],
    })
    rng = np.random.default_rng(0)
    vecs = l2_normalize(rng.normal(size=(6, 16)).astype(np.float32))
    return df, vecs, texts


def test_exact_topk_matches_bruteforce():
    rng = np.random.default_rng(1)
    m = l2_normalize(rng.normal(size=(300, 8)))
    idx, sims = exact_topk(m[:5], m, 10, chunk=2)
    for r in range(5):
        ref, _ = cosine_topk(m[r], m, 10)
        assert list(idx[r]) == list(ref)
    nself, _ = exact_topk(m[:5], m, 3, exclude_self=True)
    assert all(r not in nself[r] for r in range(5))


def test_faiss_index_recall(cfg):
    pytest.importorskip("faiss")
    cfg.raw["index"].update(backend="faiss_hnsw")
    m = l2_normalize(np.random.default_rng(2).normal(size=(3000, 32)))
    index = NeighborIndex.build(m, cfg)
    assert index.backend == "faiss_hnsw"
    assert index.recall_at_k(25, 100) >= 0.95


def test_bm25_ranks_exact_terms(corpus):
    df, _, texts = corpus
    bm = BM25Index.build(texts)
    idx, scores = bm.topk("gpt-4 reasoning", 3)
    assert idx[0] == 0 and scores[0] > 0
    assert len(bm.topk("nonexistentterm", 3)[0]) == 0


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([np.array([1, 2, 3]), np.array([2, 1, 4])])
    assert max(fused, key=fused.get) in (1, 2)
    assert fused[2] > fused[3] and fused[1] > fused[4]


def test_hybrid_search_and_degraded_fallback(cfg, corpus):
    df, vecs, texts = corpus
    labels = np.array([0, 1, 2, 3, 4, 1])
    bm = BM25Index.build(texts)
    res = search("diffusion image", FakeEmbedder(vecs, texts), vecs, df, labels, {}, top_k=3,
                 bm25=bm, cfg=cfg)
    assert res["mode"] == "hybrid" and res["results"][0]["arxiv_id"] == "x1"
    assert "diffusion" in res["results"][0]["matched_terms"]

    deg = search("federated privacy", BrokenEmbedder(), vecs, df, labels, {}, top_k=3, bm25=bm, cfg=cfg)
    assert deg["degraded"] and deg["mode"] == "lexical"
    assert deg["results"][0]["arxiv_id"] == "x3"
    none = search("federated privacy", None, vecs, df, labels, {}, top_k=3, bm25=bm, cfg=cfg)
    assert none["degraded"] and none["results"][0]["arxiv_id"] == "x3"


def test_filters_sorting_and_query_limit(cfg, corpus):
    df, vecs, texts = corpus
    labels = np.zeros(6, dtype=int)
    emb = FakeEmbedder(vecs, texts)
    f = SearchFilters(categories=["cs.CV"])
    res = search("vision", emb, vecs, df, labels, {}, top_k=5, cfg=cfg, filters=f)
    assert {r["primary_category"] for r in res["results"]} == {"cs.CV"}
    f2 = SearchFilters(date_from="2024-04-01", min_citations=5)
    res2 = search("graph", emb, vecs, df, labels, {}, top_k=5, cfg=cfg, filters=f2)
    assert {r["arxiv_id"] for r in res2["results"]} <= {"x4", "x5"}
    res3 = search("models", emb, vecs, df, labels, {}, top_k=6, cfg=cfg, sort="most_cited")
    cites = [r["cited_by_count"] for r in res3["results"]]
    assert cites == sorted(cites, reverse=True)
    long = search("x" * 5000, emb, vecs, df, labels, {}, top_k=2, cfg=cfg)
    assert len(long["query"]) == cfg["search"]["max_query_chars"]
