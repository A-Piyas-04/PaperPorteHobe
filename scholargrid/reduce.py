"""Dimensionality reduction.

* Analytical reduction (FR-03): embeddings -> ~5-10 dims for clustering.
* Landscape projection (FR-05): a *separate* 2D projection used only for
  visualisation, never as the sole analytical basis.

UMAP is used when importable; otherwise PCA provides a deterministic fallback.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .utils import get_logger

log = get_logger("reduce")


def reduce_analytical(embeddings: np.ndarray, cfg: Config) -> tuple[np.ndarray, str]:
    backend = cfg.resolve_reduce_backend("reduce")
    n_components = int(cfg["reduce"]["n_components"])
    n_components = min(n_components, embeddings.shape[1], max(2, embeddings.shape[0] - 1))
    if backend == "umap":
        return _umap(embeddings, n_components, cfg, "reduce", cfg["seed"]), "umap"
    return _pca(embeddings, n_components, cfg["seed"]), "pca"


def project_2d(embeddings: np.ndarray, cfg: Config, seed: int | None = None) -> tuple[np.ndarray, str]:
    backend = cfg.resolve_reduce_backend("landscape")
    seed = cfg["seed"] if seed is None else seed
    if backend == "umap":
        return _umap(embeddings, 2, cfg, "landscape", seed), "umap"
    return _pca(embeddings, 2, seed), "pca"


def _umap(x: np.ndarray, n_components: int, cfg: Config, section: str, seed: int) -> np.ndarray:
    import umap  # type: ignore

    section_cfg = cfg[section]
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=int(section_cfg.get("umap_n_neighbors", 15)),
        min_dist=float(section_cfg.get("umap_min_dist", 0.1)),
        metric="cosine",
        random_state=seed,
    )
    log.info("UMAP -> %dd (seed=%d)", n_components, seed)
    return reducer.fit_transform(x).astype(np.float32)


def _pca(x: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    log.info("PCA -> %dd (seed=%d)", n_components, seed)
    reducer = PCA(n_components=n_components, random_state=seed)
    return reducer.fit_transform(x).astype(np.float32)
