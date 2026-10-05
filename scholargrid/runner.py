"""Reusable pipeline orchestration.

Runs every offline stage and writes the precomputed artifact bundle. Exposed as
a function so both the CLI (``pipeline/run_pipeline.py``) and the app's
first-run "load demo data" button can call it.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Tuple

from . import __version__
from .artifacts import save_bundle
from .clustering import cluster
from .config import Config, capabilities
from .data_ingest import build_corpus, snapshot_info
from .embeddings import Embedder, paper_text
from .growth import compute_growth
from .labeling import generate_labels
from .reduce import project_2d, reduce_analytical
from .sparse import detect_sparse
from .utils import get_logger, set_seed
from .validation import validate

log = get_logger("pipeline")

ProgressFn = Optional[Callable[[float, str], None]]


def _tick(progress: ProgressFn, frac: float, msg: str) -> None:
    log.info(msg)
    if progress is not None:
        progress(frac, msg)


def run(cfg: Config, progress: ProgressFn = None) -> Tuple[Dict, Dict]:
    """Execute the full offline pipeline. Returns (meta, validation_report).

    ``progress`` is an optional callback(fraction 0..1, message) used by the UI.
    """
    t0 = time.time()
    set_seed(cfg["seed"])
    caps = capabilities()
    log.info("ScholarGrid v%s | backends available: %s", __version__, caps)

    _tick(progress, 0.05, "Ingesting arXiv corpus…")
    df = build_corpus(cfg)

    emb_backend = cfg.resolve_embedding_backend()
    _tick(progress, 0.25, f"Embedding {len(df)} papers ({emb_backend})…")
    embedder = Embedder(emb_backend, cfg)
    embeddings = embedder.fit_transform(paper_text(df))

    _tick(progress, 0.45, "Reducing dimensions and clustering…")
    reduced, reduce_backend = reduce_analytical(embeddings, cfg)
    labels, cluster_backend = cluster(reduced, cfg)

    _tick(progress, 0.6, "Labelling clusters…")
    clusters_meta = generate_labels(df, embeddings, labels, cfg)

    _tick(progress, 0.7, "Building 2D landscape…")
    coords2d, landscape_backend = project_2d(embeddings, cfg)

    _tick(progress, 0.78, "Computing publication growth…")
    growth = compute_growth(df, labels, cfg)
    for cid, info in growth["clusters"].items():
        if cid in clusters_meta:
            clusters_meta[cid]["growth"] = info

    _tick(progress, 0.85, "Detecting sparse neighbourhoods…")
    sparse_leads = detect_sparse(df, embeddings, coords2d, labels, cfg, clusters_meta)

    meta = {
        "pipeline_version": cfg["pipeline_version"],
        "scholargrid_version": __version__,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": cfg["seed"],
        "snapshot": snapshot_info(df, cfg),
        "backends": {
            "available": caps,
            "embedding": emb_backend,
            "reduce": reduce_backend,
            "cluster": cluster_backend,
            "landscape": landscape_backend,
        },
        "params": {
            "embedding": cfg["embedding"], "reduce": cfg["reduce"],
            "cluster": cfg["cluster"], "landscape": cfg["landscape"],
            "growth": cfg["growth"], "sparse": cfg["sparse"],
        },
        "counts": {
            "papers": int(len(df)), "clusters": len(clusters_meta),
            "noise": int((labels == -1).sum()), "sparse_leads": len(sparse_leads),
            "embedding_dim": int(embeddings.shape[1]),
        },
    }

    _tick(progress, 0.92, "Saving artifacts…")
    save_bundle(cfg, df, embeddings, reduced, coords2d, labels,
                clusters_meta, growth, sparse_leads, embedder, meta)

    _tick(progress, 0.96, "Validating…")
    report = validate(cfg, df, embeddings, reduced, labels, clusters_meta,
                      embedder, sparse_leads)

    dt = time.time() - t0
    log.info("Pipeline complete in %.1fs | papers=%d clusters=%d leads=%d precision@k=%s",
             dt, len(df), len(clusters_meta), len(sparse_leads),
             report["search"]["mean_precision_at_k"])
    _tick(progress, 1.0, "Done.")
    return meta, report
