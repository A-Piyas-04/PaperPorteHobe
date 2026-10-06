"""Backwards-compatible entry point: ``run(cfg, progress)`` runs every stage.

The staged implementation lives in :mod:`scholargrid.pipeline`.
"""
from __future__ import annotations

from .pipeline import ProgressFn, run

__all__ = ["run", "ProgressFn"]
