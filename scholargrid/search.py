"""FR-06  Semantic search.

Results come in two sections:

* **exact** - papers containing the query as a phrase, or one of its acronym /
  synonym expansions ("SATD" <-> "self-admitted technical debt"), ranked by
  match level (title, abstract, all words nearby), then by keyword and
  semantic score.
* **related** - papers with similar meaning that did not match exactly: the
  dense ranking fused with BM25 by Reciprocal Rank Fusion, keeping only papers
  whose similarity clearly stands out (``search.related_min_sim``, or an
  adaptive cutoff of mean + ``related_min_z`` standard deviations of the
  query's similarity to the whole corpus).

The query is embedded with the *same* embedder used for the corpus. An
optional cross-encoder re-ranks the top related candidates. If the embedding
model is unavailable the search degrades to BM25 and says so
(``degraded: true``). Results report their cluster distribution so query
ambiguity across clusters is exposed rather than hidden. Used at runtime (the
app, together with :mod:`scholargrid.live_search`) and during search
validation (FR-16).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .acronyms import Expansion, expand
from .bm25 import BM25Index, reciprocal_rank_fusion
from .phrase import PhraseIndex, matched_words
from .utils import cosine_topk, get_logger

log = get_logger("search")

SORT_OPTIONS = ("relevance", "newest", "most_cited", "citation_velocity")
_FULL_SCAN_MAX = 300_000      # corpora up to this size get an exact similarity for every paper
_AUTO_CUTOFF_MIN_DOCS = 200   # below this the corpus is too small to estimate a cutoff


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
            m &= (df["date"] >= pd.Timestamp(self.date_from)).fillna(False).to_numpy()
        if self.date_to:
            m &= (df["date"] < pd.Timestamp(self.date_to) + pd.Timedelta(days=1)).fillna(False).to_numpy()
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


def _top(scores: np.ndarray, k: int, mask: Optional[np.ndarray]) -> np.ndarray:
    s = scores if mask is None else np.where(mask, scores, -np.inf)
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return np.array([], dtype=np.int64)
    part = np.argpartition(-s, k - 1)[:k]
    return part[np.argsort(-s[part])].astype(np.int64)


def similarity_cutoff(sims: Optional[np.ndarray], scfg: Mapping, mask: Optional[np.ndarray] = None
                      ) -> Optional[float]:
    """Minimum similarity for the related section (``None`` = no cutoff).

    Adaptive: the paper must stand out from the corpus (mean + z * std) and be
    reasonably close to the best match (see :func:`relative_cutoff`).
    """
    fixed = scfg.get("related_min_sim")
    if fixed is not None:
        return float(fixed)
    if sims is None:
        return None
    vals = sims if mask is None else sims[mask]
    if len(vals) < _AUTO_CUTOFF_MIN_DOCS:
        return None
    base = float(vals.mean())
    adaptive = base + float(scfg.get("related_min_z", 2.0)) * float(vals.std())
    return max(adaptive, relative_cutoff(float(vals.max()), scfg, base) or adaptive)


def relative_cutoff(best: Optional[float], scfg: Mapping, base: float = 0.0) -> Optional[float]:
    """``related_rel_cutoff`` of the way from the typical similarity (``base``,
    the corpus mean) up to the best match, so it works for any embedding model."""
    if best is None or scfg.get("related_min_sim") is not None:
        return None
    return base + float(scfg.get("related_rel_cutoff", 0.5)) * (best - base)


def _citation_boost(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    """log(1 + citations per month since publication): age-normalised."""
    if "cited_by_count" not in df.columns or len(idx) == 0:
        return np.zeros(len(idx))
    sub = df.iloc[idx]
    ref = df["date"].max()
    months = ((ref - sub["date"]).dt.days / 30.44).clip(lower=1).to_numpy()
    return np.log1p(sub["cited_by_count"].to_numpy() / months)


def _boosted(order: List[int], df: pd.DataFrame, weight: float, rrf_k: int) -> List[int]:
    if weight <= 0 or not order:
        return order
    arr = np.array(order, dtype=np.int64)
    base = 1.0 / (np.arange(len(arr)) + rrf_k)
    boost = _citation_boost(df, arr)
    rescored = base * (1.0 + weight * boost / max(boost.max(), 1e-9))
    return [int(i) for i in arr[np.argsort(-rescored, kind="stable")]]


def sort_key(sort: str, rows: List[Dict]) -> Optional[np.ndarray]:
    """Descending sort key for non-relevance orders (``None`` for relevance)."""
    if sort == "newest":
        return np.array([pd.Timestamp(r.get("date") or "1900-01-01").value for r in rows], dtype="float64")
    if sort == "most_cited":
        return np.array([float(r.get("cited_by_count") or 0) for r in rows])
    if sort == "citation_velocity":
        return np.array([float(r.get("citation_velocity") or 0.0) for r in rows])
    return None


def search(query: str, embedder, embeddings: np.ndarray,
           df: pd.DataFrame, labels: np.ndarray,
           clusters_meta: Dict[int, Dict], top_k: Optional[int] = None, *,
           bm25: Optional[BM25Index] = None, index=None, cfg=None,
           filters: Optional[SearchFilters] = None, sort: str = "relevance",
           citation_weight: float = 0.0, reranker=None, mode: Optional[str] = None,
           phrase_index: Optional[PhraseIndex] = None, acronyms: Optional[Mapping] = None,
           expansion: Optional[Expansion] = None, qvec: Optional[np.ndarray] = None,
           top_k_exact: Optional[int] = None, top_k_related: Optional[int] = None) -> Dict:
    """Search the collection.

    ``top_k`` caps the whole list (exact matches first), which is what
    validation and the map use; without it each section has its own cap
    (``search.top_k_exact`` / ``search.top_k_related``). ``mode`` forces a
    single retrieval method ("dense", "lexical" or "hybrid") and skips the
    exact-phrase section; ``None`` is the configured behaviour.
    """
    scfg = (cfg["search"] if cfg is not None else {}) or {}
    query = clean_query(query, int(scfg.get("max_query_chars", 300)))
    exp = expansion or expand(query, acronyms, scfg.get("synonyms"))
    hybrid = scfg.get("hybrid", True) if mode is None else mode == "hybrid"
    if top_k is not None:
        cap_exact = cap_related = int(top_k)
    else:
        cap_exact = int(top_k_exact or scfg.get("top_k_exact", 100))
        cap_related = int(top_k_related if top_k_related is not None else scfg.get("top_k_related", 50))
    candidate_k = max(int(scfg.get("candidate_k", 100)), cap_related * 2, top_k or 0)
    rrf_k = int(scfg.get("rrf_k", 60))
    mask = filters.mask(df) if filters is not None and filters.active() else None

    # -- dense ---------------------------------------------------------------
    degraded = False
    sims_all: Optional[np.ndarray] = None
    dense_idx = np.array([], dtype=np.int64)
    dense_sims: Dict[int, float] = {}
    if query and embedder is not None and mode != "lexical":
        try:
            if qvec is None:
                qvec = embedder.encode_query(exp.embed_text)
            if len(embeddings) <= _FULL_SCAN_MAX:
                sims_all = np.asarray(embeddings @ qvec, dtype=np.float32)
                dense_idx = _top(sims_all, candidate_k, mask)
                dense_sims = {int(i): float(sims_all[i]) for i in dense_idx}
            else:
                dense_idx, sims = _dense(qvec, embeddings, candidate_k, index, mask)
                dense_sims = {int(i): float(s) for i, s in zip(dense_idx, sims)}
        except Exception as exc:
            log.warning("Dense search unavailable (%s); falling back to BM25.", exc)
            degraded, qvec = True, None
    elif embedder is None and mode != "lexical":
        degraded = True

    # -- lexical -------------------------------------------------------------
    lex_query = " ".join(exp.api_phrases) or query
    lex_all: Optional[np.ndarray] = None
    lex_idx = np.array([], dtype=np.int64)
    if query and bm25 is not None:
        lex_all = bm25.scores(lex_query)
        if hybrid or degraded or mode == "lexical":
            lex_idx = _top(np.where(lex_all > 0, lex_all, -np.inf), candidate_k, mask)
    max_lex = float(lex_all.max()) if lex_all is not None and len(lex_all) and lex_all.max() > 0 else 1.0

    # -- exact section -------------------------------------------------------
    levels: Dict[int, int] = {}
    if query and mode is None:
        pidx = phrase_index if phrase_index is not None else PhraseIndex.from_frame(df)
        levels = pidx.match(exp.phrases, mask, int(scfg.get("proximity_window", 8)))

    def exact_score(i: int) -> float:
        s = float(sims_all[i]) if sims_all is not None else dense_sims.get(i, 0.0)
        lex = float(lex_all[i]) / max_lex if lex_all is not None else 0.0
        return s + 0.5 * lex

    exact_order = sorted(levels, key=lambda i: (levels[i], -exact_score(i)))
    n_exact_total = len(exact_order)
    exact_order = _boosted(exact_order[:cap_exact], df, citation_weight, rrf_k)

    # -- related section -----------------------------------------------------
    exact_set = set(levels)
    d_idx = np.array([i for i in dense_idx if int(i) not in exact_set], dtype=np.int64)
    l_idx = np.array([i for i in lex_idx if int(i) not in exact_set], dtype=np.int64)
    if len(dense_idx) and len(lex_idx) and hybrid:
        fused = reciprocal_rank_fusion([d_idx, l_idx], k=rrf_k)
        related = sorted(fused, key=lambda i: -fused[i])
        used_mode = "hybrid"
    elif len(dense_idx):
        related = [int(i) for i in d_idx]
        used_mode = "dense"
    else:
        related = [int(i) for i in l_idx]
        used_mode = "lexical"

    cutoff = similarity_cutoff(sims_all, scfg, mask) if mode is None else None
    if cutoff is not None and sims_all is not None:
        related = [i for i in related if sims_all[i] >= cutoff]

    if reranker is not None and related:
        head = related[:50]
        pairs = [[query, f"{df.iloc[i]['title']}. {df.iloc[i]['abstract']}"] for i in head]
        try:
            rr = np.asarray(reranker.predict(pairs), dtype=float)
            related = [head[j] for j in np.argsort(-rr)] + related[50:]
            used_mode += "+rerank"
        except Exception as exc:  # pragma: no cover - model dependent
            log.warning("Re-ranker failed (%s); keeping fused order.", exc)
    related = _boosted(related[:cap_related], df, citation_weight, rrf_k)

    order = [(i, "exact") for i in exact_order] + [(i, "related") for i in related]
    if top_k is not None:
        order = order[:top_k]

    results: List[Dict] = []
    for i, match in order:
        r = df.iloc[i]
        cid = int(labels[i])
        sim = float(sims_all[i]) if sims_all is not None else dense_sims.get(i)
        if sim is not None:
            score = sim
        else:
            score = float(lex_all[i]) / max_lex if lex_all is not None else 0.0
        aid = str(r["arxiv_id"])
        results.append({
            "rank": 0,
            "paper_id": aid,
            "arxiv_id": aid,
            "doi": str(r.get("doi", "") or ""),
            "url": f"https://arxiv.org/abs/{aid}",
            "title": r["title"],
            "abstract": str(r.get("abstract", "") or ""),
            "authors": str(r.get("authors", "") or ""),
            "venue": str(r.get("venue", "") or ""),
            "primary_category": r["primary_category"],
            "date": r["date"].strftime("%Y-%m-%d"),
            "cluster_id": cid,
            "cluster_label": clusters_meta.get(cid, {}).get("label", "noise" if cid == -1 else ""),
            "score": round(float(score), 4),
            "sim": None if sim is None else round(sim, 4),
            "cited_by_count": int(r.get("cited_by_count", 0) or 0),
            "citation_velocity": float(r.get("citation_velocity", 0.0) or 0.0),
            "matched_terms": matched_words(exp.phrases, r["title"], r.get("abstract", "")),
            "match": match,
            "match_level": levels.get(i),
            "arxiv_url": f"https://arxiv.org/abs/{aid}",
            "sources": ["collection"],
            "is_new": False,
            "bundle_index": int(i),
        })

    key = sort_key(sort, results) if sort != "relevance" else None
    if key is not None:
        results = [results[j] for j in np.argsort(-key, kind="stable")]
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    return {
        "query": query,
        "mode": used_mode,
        "degraded": degraded,
        "result_indices": [r["bundle_index"] for r in results],
        "results": results,
        "cluster_distribution": cluster_distribution(results, clusters_meta),
        "ambiguous": sum(1 for c in cluster_distribution(results, clusters_meta)
                         if c["cluster_id"] != -1) > 1,
        "n_exact": sum(1 for r in results if r["match"] == "exact"),
        "n_related": sum(1 for r in results if r["match"] == "related"),
        "n_exact_total": n_exact_total,
        "expansion": list(exp.added),
        "phrases": list(exp.phrases),
        "api_phrases": list(exp.api_phrases),
        "cutoff": cutoff,
        "sim_base": float(sims_all.mean()) if sims_all is not None and len(sims_all) else None,
    }


def cluster_distribution(results: List[Dict], clusters_meta: Dict[int, Dict]) -> List[Dict]:
    dist = Counter(r["cluster_id"] for r in results if r.get("cluster_id") is not None)
    return [{"cluster_id": cid, "count": n,
             "label": clusters_meta.get(cid, {}).get("label", "noise" if cid == -1 else "")}
            for cid, n in dist.most_common()]
