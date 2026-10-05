#!/usr/bin/env python3
"""ScholarGrid offline analytical pipeline (CLI entry point).

Runs every offline stage end-to-end and writes the precomputed artifact bundle
consumed by the Streamlit app:

    Data -> Embeddings -> Analytical reduction -> Clustering -> Labels ->
    2D landscape -> Growth -> Semantic-search index -> Sparse neighbourhoods ->
    Validation -> Exported artifacts

Usage:
    python pipeline/run_pipeline.py [--config configs/config.yaml]
"""
from __future__ import annotations

import argparse
import os
import sys

# Make the scholargrid package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholargrid.config import load_config  # noqa: E402
from scholargrid.runner import run  # noqa: E402
from scholargrid.utils import get_logger  # noqa: E402

log = get_logger("pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="ScholarGrid offline pipeline")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    meta, report = run(cfg)

    log.info("=" * 64)
    log.info("Artifacts: %s", cfg.processed_dir)
    log.info("Reports:   %s", cfg.reports_dir)
    log.info("papers=%d clusters=%d noise=%d leads=%d",
             meta["counts"]["papers"], meta["counts"]["clusters"],
             meta["counts"]["noise"], meta["counts"]["sparse_leads"])
    log.info("search mean precision@k=%s | silhouette=%s",
             report["search"]["mean_precision_at_k"],
             report["clusters"]["silhouette_reduced"])


if __name__ == "__main__":
    main()
