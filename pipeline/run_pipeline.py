#!/usr/bin/env python3
"""ScholarGrid offline analytical pipeline (CLI entry point).

Runs the staged pipeline and writes the precomputed artifact bundle consumed by
the Streamlit app:

    ingest -> enrich -> embed -> analyze -> validate -> publish

Usage:
    python pipeline/run_pipeline.py [--config configs/config.yaml]
                                    [--stage NAME | --from-stage NAME] [--no-publish]

Equivalent to ``python -m scholargrid.pipeline``.
"""
from __future__ import annotations

import os
import sys

# Make the scholargrid package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholargrid.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
