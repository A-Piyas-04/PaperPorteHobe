"""FR-06  Semantic search.

Embeds a query with the *same* embedder used for the corpus, retrieves the
nearest papers by cosine similarity, and reports their cluster distribution so
query ambiguity across clusters is exposed rather than hidden. Used both at
runtime (the app) and during search validation (FR-16).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd

from .embeddings import Embedder
from .utils import cosine_topk


def search(query: str, embedder: Embedder, embeddings: np.ndarray,
           df: pd.DataFrame, labels: np.ndarray,
           clusters_meta: Dict[int, Dict], top_k: int = 25) -> Dict:
    qvec = embedder.encode_query(query)
    idx, scores = cosine_topk(qvec, embeddings, top_k)

    results: List[Dict] = []
    for rank, (i, score) in enumerate(zip(idx, scores), start=1):
        r = df.iloc[int(i)]
        cid = int(labels[int(i)])
        results.append({
            "rank": rank,
            "arxiv_id": r["arxiv_id"],
            "title": r["title"],
            "primary_category": r["primary_category"],
            "date": r["date"].strftime("%Y-%m-%d"),
            "cluster_id": cid,
            "cluster_label": clusters_meta.get(cid, {}).get("label", "noise" if cid == -1 else ""),
            "score": round(float(score), 4),
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
        "result_indices": [int(i) for i in idx],
        "results": results,
        "cluster_distribution": cluster_distribution,
        "ambiguous": sum(1 for c in cluster_distribution if c["cluster_id"] != -1) > 1,
    }
