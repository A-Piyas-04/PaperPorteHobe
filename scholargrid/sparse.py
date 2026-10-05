"""FR-08  Sparse-neighborhood exploration.

Candidate low-density regions located *near* populated research clusters, framed
as "investigation leads" — never as proven research gaps (FR-14 / constraints).

Pipeline:
  1. detect low-density anchors on the 2D landscape that border clustered papers,
  2. verify each in high-dimensional embedding space (reject 2D-only artifacts),
  3. attach nearby papers as supporting evidence,
  4. repeat the 2D projection across several seeds and keep only anchors that
     remain low-density under a majority of projections.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Config
from .reduce import project_2d
from .utils import get_logger

log = get_logger("sparse")


def _local_density_2d(coords: np.ndarray, radius: float) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(radius=radius).fit(coords)
    neigh = nn.radius_neighbors(coords, return_distance=False)
    return np.array([len(n) - 1 for n in neigh])  # exclude self


def _radius_from_knn(coords: np.ndarray, k: int = 10) -> float:
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(coords) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    dists, _ = nn.kneighbors(coords)
    return float(np.median(dists[:, -1]))


def _low_density_mask(coords: np.ndarray, percentile: float) -> np.ndarray:
    radius = _radius_from_knn(coords)
    dens = _local_density_2d(coords, radius)
    threshold = np.percentile(dens, percentile)
    return dens <= threshold


def detect_sparse(df: pd.DataFrame, embeddings: np.ndarray, coords2d: np.ndarray,
                  labels: np.ndarray, cfg: Config,
                  clusters_meta: Dict[int, Dict]) -> List[Dict]:
    from sklearn.neighbors import NearestNeighbors

    scfg = cfg["sparse"]
    pct = float(scfg["low_density_percentile"])

    # Step 1: 2D low-density anchors that border clustered papers.
    base_low = _low_density_mask(coords2d, pct)
    radius = _radius_from_knn(coords2d)
    nn2d = NearestNeighbors(radius=radius * 2.0).fit(coords2d)
    neigh = nn2d.radius_neighbors(coords2d, return_distance=False)
    clustered = labels != -1
    near_cluster = np.array([bool(clustered[n].any()) for n in neigh])
    candidates = np.where(base_low & near_cluster)[0]
    log.info("Step1: %d low-density anchors bordering clusters.", len(candidates))
    if len(candidates) == 0:
        return []

    # Step 2: high-dimensional verification. Keep anchors that are also sparse in
    # embedding space (above-median distance to their k-th nearest neighbour).
    k = min(int(scfg["hd_neighbors"]), len(embeddings) - 1)
    nn_hd = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    hd_dist, hd_idx = nn_hd.kneighbors(embeddings)
    hd_sparsity = hd_dist[:, -1]
    hd_threshold = float(np.median(hd_sparsity))
    verified = [c for c in candidates if hd_sparsity[c] >= hd_threshold]
    log.info("Step2: %d anchors survive high-dimensional check.", len(verified))
    if not verified:
        return []

    # Step 4: projection robustness. Re-project under several seeds AND
    # bootstrap subsamples (the latter also exercises deterministic PCA, whose
    # output is seed-invariant), keeping only anchors that remain low-density.
    survival = {c: 1 for c in verified}  # base projection counts as one
    seeds = list(scfg["robustness_seeds"])
    verified_set = list(verified)
    others = np.array([i for i in range(len(embeddings)) if i not in set(verified_set)])
    for seed in seeds:
        rng = np.random.default_rng(seed)
        keep = rng.choice(others, size=int(0.8 * len(others)), replace=False)
        subset = np.sort(np.concatenate([np.array(verified_set), keep]))
        alt = project_2d(embeddings[subset], cfg, seed=seed)[0]
        alt_low = _low_density_mask(alt, pct)
        pos_in_subset = {int(g): j for j, g in enumerate(subset)}
        for c in verified:
            if alt_low[pos_in_subset[c]]:
                survival[c] += 1
    total = len(seeds) + 1
    min_robust = float(scfg["min_robustness"])
    robust = [c for c in verified if survival[c] / total >= min_robust]
    log.info("Step4: %d anchors robust across >=%.0f%% of %d projections.",
             len(robust), min_robust * 100, total)

    # Rank by combined (2D + HD) sparsity, then de-duplicate nearby anchors.
    robust.sort(key=lambda c: -hd_sparsity[c])
    leads = _build_leads(df, labels, coords2d, hd_idx, robust, survival, total,
                         clusters_meta, int(scfg["max_leads"]), radius)
    return leads


def _build_leads(df, labels, coords2d, hd_idx, robust, survival, total,
                 clusters_meta, max_leads, min_sep) -> List[Dict]:
    leads: List[Dict] = []
    taken: List[np.ndarray] = []
    for c in robust:
        pos = coords2d[c]
        if any(np.linalg.norm(pos - t) < min_sep * 2 for t in taken):
            continue  # too close to an existing lead
        taken.append(pos)
        neighbours = [int(j) for j in hd_idx[c] if j != c]
        nearby_labels = [int(labels[j]) for j in neighbours if labels[j] != -1]
        nearest_cluster = max(set(nearby_labels), key=nearby_labels.count) if nearby_labels else None
        evidence = []
        for j in neighbours[:6]:
            r = df.iloc[j]
            evidence.append({
                "arxiv_id": r["arxiv_id"], "title": r["title"],
                "primary_category": r["primary_category"],
                "date": r["date"].strftime("%Y-%m-%d"),
                "cluster_id": int(labels[j]),
            })
        leads.append({
            "anchor_arxiv_id": df.iloc[c]["arxiv_id"],
            "anchor_title": df.iloc[c]["title"],
            "position_2d": [float(pos[0]), float(pos[1])],
            "robustness": round(survival[c] / total, 3),
            "nearest_cluster_id": nearest_cluster,
            "nearest_cluster_label": clusters_meta.get(nearest_cluster, {}).get("label")
            if nearest_cluster is not None else None,
            "evidence_papers": evidence,
            "note": "Sparse neighbourhood / investigation lead — not a proven research gap.",
        })
        if len(leads) >= max_leads:
            break
    return leads
