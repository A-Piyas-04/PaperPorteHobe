"""FR-03  Research clustering.

Density clustering on the analytically-reduced representation. HDBSCAN is used
when importable; otherwise DBSCAN with a k-distance-estimated ``eps`` provides a
density-based fallback. Both preserve noise (cluster id ``-1``) rather than
forcing every paper into a cluster.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .utils import get_logger

log = get_logger("cluster")


def cluster(reduced: np.ndarray, cfg: Config) -> tuple[np.ndarray, str]:
    backend = cfg.resolve_cluster_backend()
    min_clusters = int(cfg["cluster"].get("fallback_min_clusters", 4))
    if backend == "hdbscan":
        labels = _hdbscan(reduced, cfg)
    else:
        labels = _dbscan(reduced, cfg)
    n = len({int(c) for c in labels if c != -1})
    log.info("%s produced %d clusters, %d noise points (%.1f%%).",
             backend, n, int((labels == -1).sum()),
             100.0 * int((labels == -1).sum()) / max(1, len(labels)))
    if n < min_clusters:
        log.info("%s yielded %d clusters (<%d); using KMeans fallback.",
                 backend, n, min_clusters)
        labels = _kmeans(reduced, cfg)
        backend = "kmeans"
        n = len({int(c) for c in labels if c != -1})
        log.info("%s produced %d clusters, %d noise points (%.1f%%).",
                 backend, n, int((labels == -1).sum()),
                 100.0 * int((labels == -1).sum()) / max(1, len(labels)))
    return labels.astype(int), backend


def _kmeans(reduced: np.ndarray, cfg: Config) -> np.ndarray:
    """Silhouette-selected KMeans fallback. Preserves the noise concept (FR-03)
    by labelling points beyond a high centroid-distance percentile as -1."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    lo, hi = cfg["cluster"]["kmeans_k_range"]
    hi = min(int(hi), max(2, len(reduced) // int(cfg["cluster"]["min_cluster_size"])))
    lo = min(int(lo), hi)
    best = None
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, random_state=cfg["seed"], n_init=10).fit(reduced)
        if k == 1:
            continue
        score = silhouette_score(reduced, km.labels_, sample_size=min(2000, len(reduced)),
                                 random_state=cfg["seed"])
        if best is None or score > best[0]:
            best = (score, k, km)
    _, k, km = best
    labels = km.labels_.astype(int).copy()
    # Mark far-from-centroid points as noise.
    d = np.linalg.norm(reduced - km.cluster_centers_[labels], axis=1)
    thr = np.percentile(d, float(cfg["cluster"]["noise_percentile"]))
    labels[d > thr] = -1
    labels = _repack(labels, int(cfg["cluster"]["min_cluster_size"]))
    log.info("KMeans selected k=%d (silhouette=%.3f).", k, best[0])
    return labels


def _hdbscan(reduced: np.ndarray, cfg: Config) -> np.ndarray:
    import hdbscan  # type: ignore

    method = str(cfg["cluster"].get("hdbscan_cluster_selection_method", "leaf"))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(cfg["cluster"]["min_cluster_size"]),
        min_samples=int(cfg["cluster"]["min_samples"]),
        metric="euclidean",
        cluster_selection_method=method,
    )
    return clusterer.fit_predict(reduced)


def _dbscan(reduced: np.ndarray, cfg: Config) -> np.ndarray:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    min_samples = int(cfg["cluster"]["min_samples"])
    min_size = int(cfg["cluster"]["min_cluster_size"])

    # k-distance graph: each point's distance to its min_samples-th neighbour.
    k = min(min_samples, len(reduced) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(reduced)
    dists, _ = nn.kneighbors(reduced)
    kth = np.sort(dists[:, -1])

    # Candidate eps values: the knee of the k-distance curve plus a percentile
    # scan. A single global percentile is unreliable on diverse real text, so we
    # self-tune: pick the eps maximising the cluster count while keeping the
    # noise fraction reasonable.
    candidates = sorted({_knee(kth)} | {float(np.percentile(kth, p))
                                        for p in (40, 50, 60, 70, 80)})
    best_labels, best_eps, best_score = None, None, (-1, 1.0)
    for eps in candidates:
        if eps <= 0:
            continue
        raw = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(reduced)
        labels = _repack(raw, min_size)
        n_clusters = len({int(c) for c in labels if c != -1})
        noise_frac = float((labels == -1).mean())
        if n_clusters < 2 or noise_frac > 0.6:
            score = (n_clusters, -noise_frac)  # still track in case nothing better
        else:
            score = (n_clusters, -noise_frac)
        if score > best_score:
            best_score, best_labels, best_eps = score, labels, eps

    if best_labels is None:  # degenerate corpus: everything one blob
        best_labels = _repack(DBSCAN(eps=candidates[-1], min_samples=min_samples).fit_predict(reduced), min_size)
        best_eps = candidates[-1]
    log.info("DBSCAN selected eps=%.4f (min_samples=%d) -> %d clusters.",
             best_eps, min_samples, len({int(c) for c in best_labels if c != -1}))
    return best_labels


def _knee(sorted_vals: np.ndarray) -> float:
    """Kneedle-style knee: index of maximum perpendicular distance from the
    line joining the first and last points of the sorted k-distance curve."""
    n = len(sorted_vals)
    if n < 3:
        return float(sorted_vals[-1]) if n else 0.0
    x = np.arange(n, dtype=float)
    y = sorted_vals.astype(float)
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    denom = np.hypot(x1 - x0, y1 - y0) + 1e-12
    dist = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
    return float(y[int(np.argmax(dist))])


def _repack(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Send clusters smaller than min_size to noise (FR-03) and renumber the
    survivors to 0..K-1."""
    labels = labels.copy()
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    too_small = {int(u) for u, c in zip(unique, counts) if c < min_size}
    if too_small:
        labels = np.array([-1 if lb in too_small else lb for lb in labels])
    remaining = sorted({int(lb) for lb in labels if lb != -1})
    remap = {old: new for new, old in enumerate(remaining)}
    return np.array([remap.get(int(lb), -1) for lb in labels])
