"""Precomputed-artifact IO (FR / NFR-03 / deployment).

The expensive offline pipeline writes all results here; the deployed app only
loads these artifacts and performs lightweight query embedding + nearest-
neighbour retrieval (Deployment Requirements, NFR-01).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .config import Config
from .embeddings import Embedder
from .utils import ensure_dir, get_logger, load_json, save_json

log = get_logger("artifacts")

PAPERS_CSV = "papers.csv"
EMBEDDINGS_NPY = "embeddings.npy"
REDUCED_NPY = "reduced.npy"
CLUSTERS_JSON = "clusters.json"
GROWTH_JSON = "growth.json"
SPARSE_JSON = "sparse.json"
META_JSON = "meta.json"
EMBEDDER_DIR = "embedder"


def save_bundle(cfg: Config, df: pd.DataFrame, embeddings: np.ndarray,
                reduced: np.ndarray, coords2d: np.ndarray, labels: np.ndarray,
                clusters_meta: Dict, growth: Dict, sparse_leads: list,
                embedder: Embedder, meta: Dict) -> None:
    out = ensure_dir(cfg.processed_dir)
    table = df.copy()
    table["cluster_id"] = labels
    table["x2d"] = coords2d[:, 0]
    table["y2d"] = coords2d[:, 1]
    table.to_csv(os.path.join(out, PAPERS_CSV), index=False)

    np.save(os.path.join(out, EMBEDDINGS_NPY), embeddings.astype(np.float32))
    np.save(os.path.join(out, REDUCED_NPY), reduced.astype(np.float32))
    save_json({str(k): v for k, v in clusters_meta.items()}, os.path.join(out, CLUSTERS_JSON))
    save_json(growth, os.path.join(out, GROWTH_JSON))
    save_json(sparse_leads, os.path.join(out, SPARSE_JSON))
    save_json(meta, os.path.join(out, META_JSON))
    embedder.save(os.path.join(out, EMBEDDER_DIR))
    log.info("Saved artifact bundle to %s", out)


@dataclass
class Bundle:
    df: pd.DataFrame
    embeddings: np.ndarray
    labels: np.ndarray
    clusters_meta: Dict[int, Dict]
    growth: Dict
    sparse_leads: list
    meta: Dict
    embedder: Embedder


def load_bundle(cfg: Config) -> Bundle:
    out = cfg.processed_dir
    df = pd.read_csv(os.path.join(out, PAPERS_CSV))
    df["date"] = pd.to_datetime(df["date"])
    embeddings = np.load(os.path.join(out, EMBEDDINGS_NPY))
    labels = df["cluster_id"].to_numpy().astype(int)
    clusters_raw = load_json(os.path.join(out, CLUSTERS_JSON))
    clusters_meta = {int(k): v for k, v in clusters_raw.items()}
    growth = load_json(os.path.join(out, GROWTH_JSON))
    sparse_leads = load_json(os.path.join(out, SPARSE_JSON))
    meta = load_json(os.path.join(out, META_JSON))
    embedder = Embedder.load(os.path.join(out, EMBEDDER_DIR), cfg)
    return Bundle(df, embeddings, labels, clusters_meta, growth, sparse_leads, meta, embedder)


def bundle_exists(cfg: Config) -> bool:
    return os.path.exists(os.path.join(cfg.processed_dir, META_JSON))
