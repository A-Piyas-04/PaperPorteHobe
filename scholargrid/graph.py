"""Paper graph for the Explore view.

Each paper is linked to its nearest neighbours in embedding space, found with
the shared :class:`~scholargrid.index.NeighborIndex` (exact or ANN), so no
n x n similarity matrix is ever materialised. When OpenAlex references are
available, pairs that cite the same works get a weight boost (bibliographic
coupling). Output is an undirected edge list plus a per-paper list of the most
similar papers.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .index import NeighborIndex
from .utils import get_logger, l2_normalize

log = get_logger("graph")


def build_graph(df: pd.DataFrame, embeddings: np.ndarray, cfg: Config,
                index: Optional[NeighborIndex] = None) -> Tuple[List, Dict]:
    gcfg = cfg["graph"]
    k = int(gcfg["k_neighbors"])
    boost = float(gcfg["coupling_boost"])
    n = len(embeddings)
    if n < 2:
        return [], {}
    k = min(k, n - 1)

    unit = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    index = index or NeighborIndex.build(unit, cfg)
    nbr, sims = index.search(unit, k, exclude_self=True)

    refs = None
    if "references" in df.columns:
        refs = [set(str(r).split()) if isinstance(r, str) and r else set()
                for r in df["references"].tolist()]

    edges: Dict[Tuple[int, int], float] = {}
    neighbors: Dict[str, List] = {}
    for i in range(n):
        neighbors[str(i)] = [[int(j), round(float(s), 4)] for j, s in zip(nbr[i], sims[i])]
        for j, s in zip(nbr[i], sims[i]):
            j = int(j)
            a, b = (i, j) if i < j else (j, i)
            w = float(s)
            if refs is not None and refs[a] and refs[b]:
                w += min(len(refs[a] & refs[b]) * boost, 0.5)
            edges[(a, b)] = max(edges.get((a, b), 0.0), w)

    edge_list = [[a, b, round(w, 4)] for (a, b), w in edges.items()]
    coupled = sum(1 for (a, b) in edges if refs[a] & refs[b]) if refs is not None else 0
    log.info("Built paper graph: %d edges (k=%d, index=%s), %d with shared references.",
             len(edge_list), k, index.backend, coupled)
    return edge_list, neighbors
