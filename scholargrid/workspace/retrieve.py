"""Retrieve the papers of one topic: hybrid collection search plus one page of live sources.

Live papers have no ``bundle_index``. Their vectors come from the live store; up to
``embed_missing_live`` missing ones are embedded now, the rest stay unclustered ("found online").
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import Config
from ..utils import get_logger, l2_normalize
from .model import WsPaper

log = get_logger("workspace.retrieve")
_MAX_ABSTRACT = 1200


def _paper(r: Dict, is_new: bool) -> WsPaper:
    bi = r.get("bundle_index")
    return WsPaper(paper_id=str(r.get("paper_id") or r.get("arxiv_id") or ""), title=str(r.get("title") or ""),
                   abstract=str(r.get("abstract") or "")[:_MAX_ABSTRACT], date=str(r.get("date") or "")[:10],
                   cited_by_count=int(r.get("cited_by_count") or 0),
                   url=str(r.get("url") or r.get("arxiv_url") or ""),
                   bundle_index=None if bi is None else int(bi), is_new=is_new,
                   sim=None if r.get("sim") is None else round(float(r["sim"]), 4))


def retrieve(query: str, bundle: Any, searcher: Any, cfg: Config
             ) -> Tuple[List[WsPaper], np.ndarray, List[str]]:
    """Returns (papers, vectors aligned to the papers that have one, warnings). Papers without vectors
    are at the end of the list and have no row in ``vectors``."""
    from ..search import search

    w = cfg["workspace"]
    warnings: List[str] = []
    res = search(query, bundle.embedder, bundle.embeddings, bundle.df, bundle.labels, bundle.clusters_meta,
                 top_k=int(w["max_papers"]), bm25=bundle.bm25, index=bundle.index, cfg=cfg,
                 phrase_index=bundle.phrase, acronyms=bundle.acronyms)
    collection = [_paper(r, False) for r in res["results"]]
    if len(collection) < int(w["min_papers"]):
        warnings.append("Few papers in the collection match this topic; try a broader phrase.")
    vec_rows: List[np.ndarray] = []
    if collection:
        idx = np.array([p.bundle_index for p in collection], dtype=int)
        vec_rows.append(l2_normalize(np.asarray(bundle.embeddings[idx], dtype=np.float32)))

    with_vec: List[WsPaper] = []
    without_vec: List[WsPaper] = []
    if w["live"] and searcher is not None and getattr(searcher, "live_enabled", False):
        try:
            live = searcher.run(query, pages=1, live=True)
        except Exception as exc:          # live sources are a bonus; never fail the workspace
            log.warning("Live sources unavailable for the workspace (%s).", exc)
            warnings.append("Live sources (arXiv, OpenAlex, Semantic Scholar) did not answer; "
                            "showing the analysed collection only.")
            live = {"results": []}
        fresh = [r for r in live.get("results", []) if r.get("is_new") or r.get("bundle_index") is None]
        store = searcher.store
        dim = int(bundle.embeddings.shape[1]) if len(bundle.embeddings) else 0
        missing: List[Tuple[WsPaper, str]] = []
        live_vecs: List[np.ndarray] = []
        seen = {p.paper_id for p in collection}
        for r in fresh:
            p = _paper(r, True)
            if not p.paper_id or p.paper_id in seen:
                continue
            seen.add(p.paper_id)
            j: Optional[int] = store.find_row(p.paper_id) if len(store) else None
            if j is not None and bool(store.has_vec[j]) and store.vectors.shape[1] == dim:
                with_vec.append(p)
                live_vecs.append(np.asarray(store.vectors[j], dtype=np.float32))
            else:
                missing.append((p, f"{p.title}. {p.abstract}"))
        n_embed = int(w["embed_missing_live"])
        if missing and n_embed and bundle.embedder is not None:
            todo = missing[:n_embed]
            try:
                vecs = np.asarray(bundle.embedder.transform([t for _, t in todo]), dtype=np.float32)
                for (p, _), v in zip(todo, vecs):
                    with_vec.append(p)
                    live_vecs.append(v)
                missing = missing[n_embed:]
            except Exception as exc:
                log.warning("Embedding live papers failed (%s).", exc)
        without_vec = [p for p, _ in missing]
        if live_vecs:
            vec_rows.append(l2_normalize(np.vstack(live_vecs)))
    papers = collection + with_vec + without_vec
    dim = int(bundle.embeddings.shape[1]) if len(bundle.embeddings) else 1
    vectors = np.vstack(vec_rows) if vec_rows else np.zeros((0, dim), dtype=np.float32)
    return papers, vectors, warnings
