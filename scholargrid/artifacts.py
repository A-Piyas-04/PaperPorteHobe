"""Precomputed-artifact IO (NFR-03 / deployment).

The offline pipeline writes a bundle into ``paths.processed_dir`` (staging);
publishing copies it into a versioned release (:mod:`scholargrid.release`).
The app loads whichever bundle :func:`scholargrid.release.resolve_bundle_dir`
points to and only does query embedding + retrieval (NFR-01). Embeddings are
memory-mapped so large corpora do not need to fit in RAM twice.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import acronyms as acronym_io
from .bm25 import BM25Index
from .config import Config
from .embeddings import Embedder, paper_text
from .index import NeighborIndex
from .phrase import PhraseIndex
from .utils import ensure_dir, get_logger, load_json, save_json

log = get_logger("artifacts")

PAPERS_PARQUET = "papers.parquet"
PAPERS_CSV = "papers.csv"
EMBEDDINGS_NPY = "embeddings.npy"
REDUCED_NPY = "reduced.npy"
CLUSTERS_JSON = "clusters.json"
GROWTH_JSON = "growth.json"
SPARSE_JSON = "sparse.json"
EDGES_JSON = "edges.json"
NEIGHBORS_JSON = "neighbors.json"
META_JSON = "meta.json"
EMBEDDER_DIR = "embedder"
_CSV_MAX_ROWS = 20000
_MMAP_MIN_BYTES = 512 * 1024 * 1024

_OPTIONAL_DEFAULTS = {"authors": "", "venue": "", "venue_type": "", "references": "",
                      "openalex_id": "", "author_ids": "", "topic": "",
                      "cited_by_count": 0, "reference_count": 0,
                      "citation_velocity": 0.0, "sample_weight": 1.0}


def _save_array(path: str, arr: np.ndarray) -> None:
    try:
        np.save(path, arr)
    except OSError as exc:
        raise OSError(
            f"Could not write {path} ({exc}). On Windows this usually means a running "
            "ScholarGrid app has the old bundle open; stop it and rerun with "
            "`--from-stage analyze`.") from exc


def save_bundle(cfg: Config, df: pd.DataFrame, embeddings: np.ndarray,
                reduced: np.ndarray, coords2d: np.ndarray, labels: np.ndarray,
                clusters_meta: Dict, growth: Dict, sparse_leads: list,
                embedder: Embedder, meta: Dict,
                edges: Optional[List] = None, neighbors: Optional[Dict] = None,
                bm25: Optional[BM25Index] = None, index: Optional[NeighborIndex] = None,
                phrase: Optional[PhraseIndex] = None, acronyms: Optional[Dict] = None,
                out: Optional[str] = None) -> str:
    out = ensure_dir(out or cfg.processed_dir)
    table = df.copy()
    table["cluster_id"] = labels
    table["x2d"] = coords2d[:, 0]
    table["y2d"] = coords2d[:, 1]
    table.to_parquet(os.path.join(out, PAPERS_PARQUET), index=False)
    csv_path = os.path.join(out, PAPERS_CSV)
    if len(table) <= _CSV_MAX_ROWS:
        table.drop(columns=["references"], errors="ignore").to_csv(csv_path, index=False)
    elif os.path.exists(csv_path):
        os.remove(csv_path)

    dtype = np.float16 if cfg["embedding"]["dtype"] == "float16" else np.float32
    _save_array(os.path.join(out, EMBEDDINGS_NPY), np.asarray(embeddings).astype(dtype))
    _save_array(os.path.join(out, REDUCED_NPY), reduced.astype(np.float32))
    save_json({str(k): v for k, v in clusters_meta.items()}, os.path.join(out, CLUSTERS_JSON))
    save_json(growth, os.path.join(out, GROWTH_JSON))
    save_json(sparse_leads, os.path.join(out, SPARSE_JSON))
    save_json(edges or [], os.path.join(out, EDGES_JSON))
    save_json(neighbors or {}, os.path.join(out, NEIGHBORS_JSON))
    save_json(meta, os.path.join(out, META_JSON))
    embedder.save(os.path.join(out, EMBEDDER_DIR))
    if bm25 is not None:
        bm25.save(out)
    if index is not None:
        index.save(out)
    if phrase is not None:
        phrase.save(out)
    if acronyms is not None:
        acronym_io.save(acronyms, out)
    log.info("Saved artifact bundle to %s", out)
    return out


@dataclass
class Bundle:
    df: pd.DataFrame
    embeddings: np.ndarray
    labels: np.ndarray
    clusters_meta: Dict[int, Dict]
    growth: Dict
    sparse_leads: list
    meta: Dict
    embedder: Optional[Embedder]
    edges: List = field(default_factory=list)
    neighbors: Dict = field(default_factory=dict)
    bm25: Optional[BM25Index] = None
    index: Optional[NeighborIndex] = None
    path: str = ""
    embedder_error: Optional[str] = None
    phrase: Optional[PhraseIndex] = None
    acronyms: Dict = field(default_factory=dict)


def _load_optional(path: str, default):
    return load_json(path) if os.path.exists(path) else default


def load_papers(out: str) -> pd.DataFrame:
    pq = os.path.join(out, PAPERS_PARQUET)
    if os.path.exists(pq):
        df = pd.read_parquet(pq)
    else:
        df = pd.read_csv(os.path.join(out, PAPERS_CSV), dtype={"arxiv_id": str})
    df["arxiv_id"] = df["arxiv_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    if "year_month" not in df.columns:
        df["year_month"] = df["date"].dt.strftime("%Y-%m")
    for col, default in _OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)
    df["cited_by_count"] = df["cited_by_count"].astype(int)
    df["reference_count"] = df["reference_count"].astype(int)
    return df


def load_bundle(cfg: Config, directory: Optional[str] = None) -> Bundle:
    """Load a bundle. If the embedding model cannot be loaded the bundle still
    loads with ``embedder=None`` so the app can fall back to BM25 search."""
    from .release import resolve_bundle_dir

    out = directory or resolve_bundle_dir(cfg)
    df = load_papers(out)
    emb_path = os.path.join(out, EMBEDDINGS_NPY)
    # Memory-mapping locks the file on Windows, so only do it when it saves real memory.
    big = os.path.getsize(emb_path) > _MMAP_MIN_BYTES
    embeddings = np.load(emb_path, mmap_mode="r" if big else None)
    labels = df["cluster_id"].to_numpy().astype(int)
    clusters_meta = {int(k): v for k, v in load_json(os.path.join(out, CLUSTERS_JSON)).items()}
    growth = load_json(os.path.join(out, GROWTH_JSON))
    sparse_leads = load_json(os.path.join(out, SPARSE_JSON))
    meta = load_json(os.path.join(out, META_JSON))
    edges = _load_optional(os.path.join(out, EDGES_JSON), [])
    neighbors = _load_optional(os.path.join(out, NEIGHBORS_JSON), {})

    embedder, err = None, None
    try:
        embedder = Embedder.load(os.path.join(out, EMBEDDER_DIR), cfg)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        log.error("Embedding model failed to load (%s); search will use BM25 only.", err)

    bm25 = BM25Index.load(out)
    if bm25 is None and len(df) <= 50000:
        bm25 = BM25Index.build(paper_text(df))
    index = NeighborIndex.load(out, embeddings, cfg)
    phrase = PhraseIndex.load(out, len(df)) or PhraseIndex.from_frame(df)
    acronyms = acronym_io.load(out) or acronym_io.mine(df["abstract"].astype(str))
    return Bundle(df, embeddings, labels, clusters_meta, growth, sparse_leads, meta,
                  embedder, edges, neighbors, bm25, index, out, err, phrase, acronyms)


def bundle_exists(cfg: Config) -> bool:
    from .release import resolve_bundle_dir

    try:
        return os.path.exists(os.path.join(resolve_bundle_dir(cfg), META_JSON))
    except FileNotFoundError:
        return False
