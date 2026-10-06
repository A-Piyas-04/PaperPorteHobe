"""Nearest-neighbour index for search and the paper graph.

* ``exact``      - chunked brute-force inner product (vectors are L2-normalised,
                   so inner product == cosine). Exact, fine up to ~20k papers.
* ``faiss_hnsw`` - FAISS ``IndexHNSWFlat`` (inner product). Built offline and
                   shipped in the bundle; recall@k vs. exact is measured on a
                   sample and recorded in the validation report.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from .config import Config
from .utils import ensure_dir, get_logger

log = get_logger("index")

INDEX_FILE = "ann.faiss"


def exact_topk(queries: np.ndarray, matrix: np.ndarray, k: int,
               exclude_self: bool = False, chunk: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k rows of ``matrix`` per query, in chunks to bound memory."""
    q = np.atleast_2d(np.asarray(queries, dtype=np.float32))
    m = np.asarray(matrix, dtype=np.float32)
    k = min(k, m.shape[0] - (1 if exclude_self else 0))
    all_idx = np.zeros((q.shape[0], k), dtype=np.int64)
    all_sim = np.zeros((q.shape[0], k), dtype=np.float32)
    for s in range(0, q.shape[0], chunk):
        sims = q[s:s + chunk] @ m.T
        if exclude_self:
            rows = np.arange(sims.shape[0])
            sims[rows, rows + s] = -np.inf
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        part_sims = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_sims, axis=1)
        all_idx[s:s + chunk] = np.take_along_axis(part, order, axis=1)
        all_sim[s:s + chunk] = np.take_along_axis(part_sims, order, axis=1)
    return all_idx, all_sim


class NeighborIndex:
    def __init__(self, backend: str, vectors: np.ndarray, cfg: Config, ann=None):
        self.backend = backend
        self.vectors = vectors
        self.cfg = cfg
        self._ann = ann

    @classmethod
    def build(cls, vectors: np.ndarray, cfg: Config) -> "NeighborIndex":
        backend = cfg.resolve_index_backend(len(vectors))
        if backend == "exact":
            return cls("exact", vectors, cfg)
        import faiss  # type: ignore

        icfg = cfg["index"]
        data = np.ascontiguousarray(vectors, dtype=np.float32)
        ann = faiss.IndexHNSWFlat(data.shape[1], int(icfg["m"]), faiss.METRIC_INNER_PRODUCT)
        ann.hnsw.efConstruction = int(icfg["ef_construction"])
        ann.add(data)
        ann.hnsw.efSearch = int(icfg["ef_search"])
        log.info("Built FAISS HNSW index over %d vectors (M=%s).", len(data), icfg["m"])
        return cls("faiss_hnsw", vectors, cfg, ann)

    def search(self, queries: np.ndarray, k: int,
               exclude_self: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if self.backend == "exact" or self._ann is None:
            return exact_topk(queries, self.vectors, k, exclude_self=exclude_self)
        q = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        extra = 1 if exclude_self else 0
        sims, idx = self._ann.search(q, k + extra)
        if exclude_self:
            out_i = np.zeros((len(q), k), dtype=np.int64)
            out_s = np.zeros((len(q), k), dtype=np.float32)
            for r in range(len(q)):
                keep = [(i, s) for i, s in zip(idx[r], sims[r]) if i != r and i >= 0][:k]
                out_i[r, :len(keep)] = [i for i, _ in keep]
                out_s[r, :len(keep)] = [s for _, s in keep]
            return out_i, out_s
        return idx.astype(np.int64), sims.astype(np.float32)

    def recall_at_k(self, k: int = 25, sample: int = 200, seed: int = 0) -> Optional[float]:
        if self.backend == "exact":
            return 1.0
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(self.vectors), size=min(sample, len(self.vectors)), replace=False)
        q = np.asarray(self.vectors[picks], dtype=np.float32)
        truth, _ = exact_topk(q, self.vectors, k)
        approx, _ = self.search(q, k)
        hits = [len(set(t) & set(a)) / k for t, a in zip(truth, approx)]
        return round(float(np.mean(hits)), 4)

    # -- persistence -----------------------------------------------------------
    def save(self, directory: str) -> None:
        if self.backend != "faiss_hnsw":
            return
        import faiss  # type: ignore

        ensure_dir(directory)
        faiss.write_index(self._ann, os.path.join(directory, INDEX_FILE))

    @classmethod
    def load(cls, directory: str, vectors: np.ndarray, cfg: Config) -> "NeighborIndex":
        path = os.path.join(directory, INDEX_FILE)
        if os.path.exists(path):
            try:
                import faiss  # type: ignore

                ann = faiss.read_index(path)
                ann.hnsw.efSearch = int(cfg["index"]["ef_search"])
                return cls("faiss_hnsw", vectors, cfg, ann)
            except Exception as exc:  # pragma: no cover - faiss missing at runtime
                log.warning("Could not load ANN index (%s); using exact search.", exc)
        return cls("exact", vectors, cfg)
