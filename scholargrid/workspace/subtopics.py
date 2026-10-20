"""Cluster a workspace into sub-topics and micro-topics.

Spherical k-means on unit vectors is stable for 50-3,000 points, fast and reproducible with a seed;
k is chosen by cosine silhouette. HDBSCAN on a few hundred raw vectors gives too much noise here.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _kmeans(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)


def choose_k(x: np.ndarray, k_range: Tuple[int, int], seed: int) -> Tuple[int, np.ndarray, float]:
    from sklearn.metrics import silhouette_score

    lo, hi = int(k_range[0]), int(k_range[1])
    lo = max(2, min(lo, len(x) - 1))
    hi = max(lo, min(hi, len(x) // 10, len(x) - 1))
    best: Tuple[int, np.ndarray, float] = (lo, _kmeans(x, lo, seed), -1.0)
    for k in range(lo, hi + 1):
        labels = _kmeans(x, k, seed)
        if len(set(labels)) < 2:
            continue
        sample = min(len(x), 2000)
        s = float(silhouette_score(x, labels, metric="cosine", sample_size=sample, random_state=seed))
        if s > best[2]:
            best = (k, labels, s)
    return best


def merge_small(x: np.ndarray, labels: np.ndarray, min_size: int) -> np.ndarray:
    labels = labels.copy()
    while True:
        ids, counts = np.unique(labels, return_counts=True)
        small = ids[counts < min_size]
        if len(small) == 0 or len(ids) <= 2:
            return labels
        cents = {c: x[labels == c].mean(axis=0) for c in ids}
        s = int(small[0])
        others = [c for c in ids if c != s]
        target = max(others, key=lambda c: float(cents[s] @ cents[c]))
        labels[labels == s] = target


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    ids, counts = np.unique(labels, return_counts=True)
    order = {int(c): i for i, c in enumerate(ids[np.argsort(-counts, kind="stable")])}
    return np.array([order[int(c)] for c in labels], dtype=int)


def subtopics(x: np.ndarray, cfg: Any) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Returns (top-level labels 0..K-1, micro labels as global ids >= K or -1, info)."""
    w = cfg["workspace"]
    seed = int(cfg["seed"])
    if len(x) < 2 * int(w["min_subtopic_size"]):
        return np.zeros(len(x), dtype=int), -np.ones(len(x), dtype=int), {"k": 1 if len(x) else 0,
                                                                          "silhouette": None}
    k, labels, sil = choose_k(x, tuple(w["subtopic_k_range"]), seed)
    labels = relabel_by_size(merge_small(x, labels, int(w["min_subtopic_size"])))
    micro = -np.ones(len(x), dtype=int)
    next_id = int(labels.max()) + 1
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < int(w["micro_min_size"]):
            continue
        _, ml, ms = choose_k(x[idx], tuple(w["micro_k_range"]), seed)
        if ms < float(w["min_silhouette_micro"]) or len(set(ml)) < 2:
            continue
        ml = merge_small(x[idx], ml, max(5, int(w["min_subtopic_size"]) // 2))
        if len(set(ml)) < 2:
            continue
        for m in np.unique(ml):
            micro[idx[ml == m]] = next_id
            next_id += 1
    return labels, micro, {"k": int(len(np.unique(labels))), "silhouette": round(float(sil), 4)}
