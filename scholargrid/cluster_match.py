"""Stable cluster identities across refreshes.

New clusters are matched one-to-one to the previous release's clusters with
the Hungarian algorithm on a score that combines member overlap (Jaccard on
arXiv IDs) and centroid similarity. Previous centroids are recomputed from the
*current* embeddings of their surviving members, so a model change does not
break matching. Matched clusters inherit the previous ID; the rest get fresh
IDs above the previous maximum. Merges and splits are reported for the user
changelog.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import get_logger, l2_normalize

log = get_logger("match")


def _centroids(emb: np.ndarray, labels: np.ndarray, ids: List[int]) -> np.ndarray:
    out = np.zeros((len(ids), emb.shape[1]), dtype=np.float32)
    for r, cid in enumerate(ids):
        idx = np.where(labels == cid)[0]
        if len(idx):
            out[r] = emb[idx].mean(axis=0)
    return l2_normalize(out)


def match_clusters(arxiv_ids: pd.Series, labels: np.ndarray, embeddings: np.ndarray,
                   previous: Optional[pd.DataFrame], min_score: float = 0.3) -> Dict:
    """``previous`` has columns ``arxiv_id`` and ``cluster_id`` from the last
    release. Returns ``{"labels", "mapping", "report"}``."""
    new_ids = sorted({int(c) for c in labels if c != -1})
    if previous is None or previous.empty or not new_ids:
        return {"labels": labels, "mapping": {c: c for c in new_ids},
                "report": {"matched": 0, "previous_clusters": 0, "new_clusters": len(new_ids),
                           "kept_fraction": None, "merges": [], "splits": []}}

    prev_map = dict(zip(previous["arxiv_id"].astype(str), previous["cluster_id"].astype(int)))
    ids = arxiv_ids.astype(str).to_numpy()
    prev_labels = np.array([prev_map.get(a, -1) for a in ids])
    old_ids = sorted({int(c) for c in previous["cluster_id"] if int(c) != -1})

    members_new = {c: set(ids[labels == c]) for c in new_ids}
    members_old = {c: {a for a, p in prev_map.items() if p == c} for c in old_ids}
    emb = np.asarray(embeddings, dtype=np.float32)
    c_new = _centroids(emb, labels, new_ids)
    c_old = _centroids(emb, prev_labels, old_ids)
    sim = c_new @ c_old.T

    score = np.zeros((len(new_ids), len(old_ids)))
    for i, cn in enumerate(new_ids):
        for j, co in enumerate(old_ids):
            inter = len(members_new[cn] & members_old[co])
            union = len(members_new[cn] | members_old[co]) or 1
            score[i, j] = 0.6 * inter / union + 0.4 * max(0.0, float(sim[i, j]))

    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-score)
    mapping: Dict[int, int] = {}
    for r, c in zip(rows, cols):
        if score[r, c] >= min_score:
            mapping[new_ids[r]] = old_ids[c]
    next_id = max(old_ids + new_ids) + 1
    for cn in new_ids:
        if cn not in mapping:
            mapping[cn] = next_id
            next_id += 1
    remapped = np.array([mapping.get(int(c), -1) if c != -1 else -1 for c in labels])

    merges, splits = [], []
    for cn in new_ids:
        sources = [co for co in old_ids
                   if len(members_new[cn] & members_old[co]) >= 0.3 * max(1, len(members_new[cn]))]
        if len(sources) >= 2:
            merges.append({"new_id": mapping[cn], "from": sources})
    for co in old_ids:
        targets = [mapping[cn] for cn in new_ids
                   if len(members_new[cn] & members_old[co]) >= 0.3 * max(1, len(members_old[co]))]
        if len(targets) >= 2:
            splits.append({"old_id": co, "into": targets})

    matched = sum(1 for cn in new_ids if mapping[cn] in old_ids)
    report = {
        "matched": matched,
        "previous_clusters": len(old_ids),
        "new_clusters": len(new_ids) - matched,
        "kept_fraction": round(matched / len(old_ids), 4) if old_ids else None,
        "disappeared": sorted(set(old_ids) - set(mapping.values())),
        "merges": merges,
        "splits": splits,
    }
    log.info("Cluster matching: %d/%d previous clusters kept their ID, %d new, %d merges, %d splits.",
             matched, len(old_ids), report["new_clusters"], len(merges), len(splits))
    return {"labels": remapped, "mapping": mapping, "report": report}
