"""Paper graph for the Explore view.

Each paper is linked to its nearest neighbours in embedding space. When
OpenAlex references are available, pairs that cite the same works get a
weight boost (bibliographic coupling) - the same similarity idea Connected
Papers uses. Output is an undirected edge list plus a per-paper list of the
most similar papers.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .utils import get_logger, l2_normalize

log = get_logger("graph")


def build_graph(df: pd.DataFrame, embeddings: np.ndarray, cfg: Config) -> Tuple[List, Dict]:
    gcfg = cfg.get("graph", {}) or {}
    k = int(gcfg.get("k_neighbors", 6))
    boost = float(gcfg.get("coupling_boost", 0.15))
    n = len(embeddings)
    if n < 2:
        return [], {}
    k = min(k, n - 1)

    unit = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    sims = unit @ unit.T
    np.fill_diagonal(sims, -np.inf)
    nbr = np.argpartition(-sims, k - 1, axis=1)[:, :k]

    refs = None
    if "references" in df.columns:
        refs = [set(str(r).split()) if isinstance(r, str) and r else set()
                for r in df["references"].tolist()]

    edges: Dict[Tuple[int, int], float] = {}
    neighbors: Dict[str, List] = {}
    for i in range(n):
        order = nbr[i][np.argsort(-sims[i, nbr[i]])]
        neighbors[str(i)] = [[int(j), round(float(sims[i, j]), 4)] for j in order]
        for j in order:
            a, b = (i, int(j)) if i < j else (int(j), i)
            w = float(sims[i, j])
            if refs is not None and refs[a] and refs[b]:
                shared = len(refs[a] & refs[b])
                w += min(shared * boost, 0.5)
            edges[(a, b)] = max(edges.get((a, b), 0.0), w)

    edge_list = [[a, b, round(w, 4)] for (a, b), w in edges.items()]
    coupled = 0
    if refs is not None:
        coupled = sum(1 for (a, b) in edges if refs[a] & refs[b])
    log.info("Built paper graph: %d edges (k=%d), %d with shared references.",
             len(edge_list), k, coupled)
    return edge_list, neighbors
