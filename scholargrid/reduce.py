"""Dimensionality reduction.

* Analytical reduction (FR-03): embeddings -> ~5-10 dims for clustering.
* Landscape projection (FR-05): a *separate* 2D projection used only for
  visualisation, never as the sole analytical basis.

UMAP is used when importable; otherwise PCA provides a deterministic fallback.
For large corpora the reducer is fitted on a random sample of ``fit_sample``
papers and the rest are mapped with ``transform``, which keeps 500k-paper runs
tractable on one machine.
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
    return _fit(embeddings, n_components, cfg, "reduce", cfg["seed"], backend), backend


def project_2d(embeddings: np.ndarray, cfg: Config, seed: int | None = None) -> tuple[np.ndarray, str]:
    backend = cfg.resolve_reduce_backend("landscape")
    seed = cfg["seed"] if seed is None else seed
    return _fit(embeddings, 2, cfg, "landscape", seed, backend), backend


def _make(backend: str, n_components: int, cfg: Config, section: str, seed: int, n_rows: int):
    if backend == "umap":
        import umap  # type: ignore

        section_cfg = cfg[section]
        return umap.UMAP(
            n_components=n_components,
            n_neighbors=min(int(section_cfg.get("umap_n_neighbors", 15)), max(2, n_rows - 1)),
            min_dist=float(section_cfg.get("umap_min_dist", 0.1)),
            metric="cosine",
            random_state=seed,
        )
    from sklearn.decomposition import PCA

    return PCA(n_components=n_components, random_state=seed)


def _fit(x: np.ndarray, n_components: int, cfg: Config, section: str, seed: int,
         backend: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    fit_sample = int(cfg[section].get("fit_sample", len(x)))
    if len(x) <= fit_sample:
        log.info("%s -> %dd on %d rows (seed=%d)", backend.upper(), n_components, len(x), seed)
        return _make(backend, n_components, cfg, section, seed, len(x)).fit_transform(x).astype(np.float32)

    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(len(x), size=fit_sample, replace=False))
    log.info("%s -> %dd fitted on %d of %d rows, transforming the rest (seed=%d)",
             backend.upper(), n_components, fit_sample, len(x), seed)
    reducer = _make(backend, n_components, cfg, section, seed, fit_sample)
    out = np.zeros((len(x), n_components), dtype=np.float32)
    out[sample] = reducer.fit_transform(x[sample])
    rest = np.setdiff1d(np.arange(len(x)), sample)
    for s in range(0, len(rest), 50000):
        part = rest[s:s + 50000]
        out[part] = reducer.transform(x[part])
    return out
