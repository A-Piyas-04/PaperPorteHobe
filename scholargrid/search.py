"""FR-06  Semantic search.

Embeds a query with the *same* embedder used for the corpus and retrieves the
nearest papers. With ``search.hybrid`` the dense ranking is fused with a BM25
lexical ranking by Reciprocal Rank Fusion, which helps exact-term queries
(model names, acronyms). An optional cross-encoder re-ranks the top
candidates. If the embedding model is unavailable the search degrades to BM25
and says so (``degraded: true``).

Results report their cluster distribution so query ambiguity across clusters
is exposed rather than hidden. Used at runtime (the app) and during search
validation (FR-16).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .bm25 import BM25Index, reciprocal_rank_fusion
from .utils import cosine_topk, get_logger

log = get_logger("search")

SORT_OPTIONS = ("relevance", "newest", "most_cited", "citation_velocity")


@dataclass
class SearchFilters:
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    categories: Sequence[str] = field(default_factory=list)
    min_citations: int = 0
    venue_only: bool = False

    def active(self) -> bool:
        return bool(self.date_from or self.date_to or self.categories
                    or self.min_citations > 0 or self.venue_only)

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(df), dtype=bool)
        if self.date_from:
            m &= (df["date"] >= pd.Timestamp(self.date_from)).to_numpy()
        if self.date_to:
            m &= (df["date"] < pd.Timestamp(self.date_to) + pd.Timedelta(days=1)).to_numpy()
        if self.categories:
            cats = set(self.categories)
            m &= df["primary_category"].isin(cats).to_numpy()
        if self.min_citations > 0 and "cited_by_count" in df.columns:
            m &= (df["cited_by_count"] >= self.min_citations).to_numpy()
        if self.venue_only and "venue" in df.columns:
            m &= (df["venue"].fillna("").astype(str) != "").to_numpy()
        return m


def clean_query(query: str, max_chars: int = 300) -> str:
    q = " ".join(str(query or "").split())
    return q[:max_chars]


def _dense(qvec: np.ndarray, embeddings: np.ndarray, k: int, index=None,
           mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    if mask is not None:
        allowed = np.flatnonzero(mask)
        if len(allowed) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        if index is None or index.backend == "exact":
            sub = np.asarray(embeddings[allowed], dtype=np.float32)
            idx, sims = cosine_topk(qvec, sub, min(k, len(allowed)))
            return allowed[idx], sims
        idx, sims = index.search(qvec[None, :], min(len(embeddings), k * 10))
        keep = [(i, s) for i, s in zip(idx[0], sims[0]) if i >= 0 and mask[i]][:k]
        return (np.array([i for i, _ in keep], dtype=np.int64),
                np.array([s for _, s in keep], dtype=np.float32))
    if index is not None:
        idx, sims = index.search(qvec[None, :], k)
        return idx[0], sims[0]
    return cosine_topk(qvec, np.asarray(embeddings, dtype=np.float32), k)


def _citation_boost(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    """log(1 + citations per month since publication): age-normalised."""
    if "cited_by_count" not in df.columns or len(idx) == 0:
        return np.zeros(len(idx))
    sub = df.iloc[idx]
    ref = df["date"].max()
    months = ((ref - sub["date"]).dt.days / 30.44).clip(lower=1).to_numpy()
    return np.log1p(sub["cited_by_count"].to_numpy() / months)


def search(query: str, embedder, embeddings: np.ndarray,
           df: pd.DataFrame, labels: np.ndarray,
           clusters_meta: Dict[int, Dict], top_k: int = 25, *,
           bm25: Optional[BM25Index] = None, index=None, cfg=None,
           filters: Optional[SearchFilters] = None, sort: str = "relevance",
           citation_weight: float = 0.0, reranker=None, mode: Optional[str] = None) -> Dict:
    scfg = (cfg["search"] if cfg is not None else {}) or {}
    query = clean_query(query, int(scfg.get("max_query_chars", 300)))
    hybrid = scfg.get("hybrid", True) if mode is None else mode == "hybrid"
    candidate_k = max(int(scfg.get("candidate_k", 100)), top_k)
    rrf_k = int(scfg.get("rrf_k", 60))
    mask = filters.mask(df) if filters is not None and filters.active() else None

    degraded = False
    dense_idx = np.array([], dtype=np.int64)
    dense_sims: Dict[int, float] = {}
    if query and embedder is not None and mode != "lexical":
        try:
            qvec = embedder.encode_query(query)
            dense_idx, sims = _dense(qvec, embeddings, candidate_k, index, mask)
            dense_sims = {int(i): float(s) for i, s in zip(dense_idx, sims)}
        except Exception as exc:
            log.warning("Dense search unavailable (%s); falling back to BM25.", exc)
            degraded = True
    elif embedder is None and mode != "lexical":
        degraded = True

    lex_idx = np.array([], dtype=np.int64)
    lex_scores: Dict[int, float] = {}
    if query and bm25 is not None and (hybrid or degraded or mode == "lexical"):
        li, ls = bm25.topk(query, candidate_k * (5 if mask is not None else 1))
        if mask is not None:
            keep = mask[li]
            li, ls = li[keep][:candidate_k], ls[keep][:candidate_k]
        lex_idx = li
        lex_scores = {int(i): float(s) for i, s in zip(li, ls)}

    if len(dense_idx) and len(lex_idx) and hybrid:
        fused = reciprocal_rank_fusion([dense_idx, lex_idx], k=rrf_k)
        order = sorted(fused, key=lambda i: -fused[i])
        used_mode = "hybrid"
    elif len(dense_idx):
        order = [int(i) for i in dense_idx]
        used_mode = "dense"
    else:
        order = [int(i) for i in lex_idx]
        used_mode = "lexical"

    if reranker is not None and order:
        head = order[:50]
        pairs = [[query, f"{df.iloc[i]['title']}. {df.iloc[i]['abstract']}"] for i in head]
        try:
            rr = np.asarray(reranker.predict(pairs), dtype=float)
            order = [head[j] for j in np.argsort(-rr)] + order[50:]
            used_mode += "+rerank"
        except Exception as exc:  # pragma: no cover - model dependent
            log.warning("Re-ranker failed (%s); keeping fused order.", exc)

    order_arr = np.array(order, dtype=np.int64)
    if citation_weight > 0 and len(order_arr):
        base = 1.0 / (np.arange(len(order_arr)) + rrf_k)
        boost = _citation_boost(df, order_arr)
        rescored = base * (1.0 + citation_weight * boost / max(boost.max(), 1e-9))
        order_arr = order_arr[np.argsort(-rescored)]
    order_arr = order_arr[:top_k]

    if sort != "relevance" and len(order_arr):
        sub = df.iloc[order_arr]
        key = {"newest": sub["date"].to_numpy(),
               "most_cited": sub.get("cited_by_count", pd.Series(0, index=sub.index)).to_numpy(),
               "citation_velocity": sub.get("citation_velocity", pd.Series(0.0, index=sub.index)).to_numpy(),
               }.get(sort)
        if key is not None:
            order_arr = order_arr[np.argsort(-key.astype("float64") if key.dtype.kind != "M"
                                             else -key.astype("int64"), kind="stable")]

    max_lex = max(lex_scores.values()) if lex_scores else 1.0
    terms = bm25.query_terms(query) if bm25 is not None else []
    results: List[Dict] = []
    for rank, i in enumerate(order_arr, start=1):
        i = int(i)
        r = df.iloc[i]
        cid = int(labels[i])
        score = dense_sims.get(i)
        if score is None:
            score = lex_scores.get(i, 0.0) / max_lex if max_lex else 0.0
        text = f"{r['title']} {r['abstract']}".lower()
        results.append({
            "rank": rank,
            "arxiv_id": r["arxiv_id"],
            "title": r["title"],
            "primary_category": r["primary_category"],
            "date": r["date"].strftime("%Y-%m-%d"),
            "cluster_id": cid,
            "cluster_label": clusters_meta.get(cid, {}).get("label", "noise" if cid == -1 else ""),
            "score": round(float(score), 4),
            "cited_by_count": int(r.get("cited_by_count", 0) or 0),
            "matched_terms": [t for t in terms if t in text],
            "arxiv_url": f"https://arxiv.org/abs/{r['arxiv_id']}",
        })

    dist = Counter(r["cluster_id"] for r in results)
    cluster_distribution = [
        {"cluster_id": cid, "count": n,
         "label": clusters_meta.get(cid, {}).get("label", "noise" if cid == -1 else "")}
        for cid, n in dist.most_common()
    ]
    return {
        "query": query,
        "mode": used_mode,
        "degraded": degraded,
        "result_indices": [int(i) for i in order_arr],
        "results": results,
        "cluster_distribution": cluster_distribution,
        "ambiguous": sum(1 for c in cluster_distribution if c["cluster_id"] != -1) > 1,
    }
