"""Hand-built GapContext tables for detector tests (no bundle, no I/O)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from conftest import make_config

from scholargrid.gaps.context import GapContext


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def papers_df(n: int, start: str = "2024-01-01", months_apart: int = 1, prefix: str = "p") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq=f"{months_apart}MS")
    return pd.DataFrame({"arxiv_id": [f"{prefix}{i}" for i in range(n)], "title": [f"Paper {prefix}{i}" for i in range(n)],
                         "abstract": [f"Abstract of paper {prefix}{i}." for i in range(n)], "date": dates,
                         "cited_by_count": list(range(n)), "author_ids": [f"A{i}" for i in range(n)],
                         "references": [""] * n, "openalex_id": [""] * n})


def make_ctx(tmp_path, papers: pd.DataFrame, *, facets: Optional[pd.DataFrame] = None,
             problems: Optional[pd.DataFrame] = None, problem_vecs: Optional[np.ndarray] = None,
             findings: Optional[pd.DataFrame] = None, profiles: Optional[pd.DataFrame] = None,
             vectors: Optional[np.ndarray] = None, value_vectors: Optional[Dict] = None,
             ref_date: Optional[str] = None, sub_labels: Optional[np.ndarray] = None,
             sub_names: Optional[Dict[int, str]] = None, **gap_overrides) -> GapContext:
    cfg = make_config(tmp_path)
    cfg.raw["gaps"].update(gap_overrides)
    if vectors is None:
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((len(papers), 8)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    papers = papers.copy()
    papers["bidx"] = np.arange(len(papers))
    return GapContext(
        cfg=cfg, scope={"kind": "test", "id": "t", "label": "Test scope"}, papers=papers.reset_index(drop=True),
        vectors=vectors, facets=facets if facets is not None else pd.DataFrame(columns=["arxiv_id", "facet", "value", "role"]),
        problems=problems if problems is not None else pd.DataFrame(), problem_vectors=problem_vecs,
        findings=findings if findings is not None else pd.DataFrame(),
        profiles=profiles if profiles is not None else pd.DataFrame(),
        ref_date=pd.Timestamp(ref_date) if ref_date else pd.Timestamp(papers["date"].max()),
        all_papers=papers, all_vectors=vectors, value_vectors=value_vectors or {}, sub_labels=sub_labels,
        sub_names=sub_names or {})


def problems_df(rows: List[dict]) -> pd.DataFrame:
    base = {"section": "S5", "anchor": "S5", "section_kind": "limitations", "problem_type": "limitation_of_this_work",
            "specificity": 2, "verified_ratio": 100.0, "source_mode": "llm"}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["problem_id"] = [f"q{i}" for i in range(len(df))]
    df["vrow"] = np.arange(len(df))
    return df


def facets_rows(pairs: List[tuple]) -> pd.DataFrame:
    """pairs = [(arxiv_id, facet, value, role)]"""
    return pd.DataFrame(pairs, columns=["arxiv_id", "facet", "value", "role"])
