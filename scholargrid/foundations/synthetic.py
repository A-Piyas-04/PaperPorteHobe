"""Deterministic fake references and works for the synthetic corpus (CI/demo only).

Each paper in topic t cites 4-8 works from a per-topic pool of 20 (``WSYN{t}{k:02d}``), Zipf-weighted
so a few works are clearly foundational. ``WSYNALL01`` / ``WSYNALL02`` are cited by every topic and must
lose on lift.
"""
from __future__ import annotations

import hashlib
from typing import List

import numpy as np
import pandas as pd

_TOPIC = {"cs.CL": 1, "cs.CV": 2, "cs.LG": 3, "cs.CR": 4, "cs.RO": 5, "cs.DC": 6}


def synthetic_references(df: pd.DataFrame) -> pd.Series:
    out = []
    weights = 1.0 / np.arange(1, 21) ** 1.1
    weights /= weights.sum()
    for aid, cat in zip(df["arxiv_id"].astype(str), df["primary_category"].astype(str)):
        rng = np.random.default_rng(int(hashlib.sha1(aid.encode()).hexdigest()[:8], 16))
        t = _TOPIC.get(cat, 9)
        picks = rng.choice(20, size=int(rng.integers(4, 9)), replace=False, p=weights)
        refs = [f"WSYN{t}{k:02d}" for k in sorted(picks)] + ["WSYNALL01", "WSYNALL02"]
        out.append(" ".join(refs))
    return pd.Series(out, index=df.index)


def synthetic_works(ids: List[str]) -> pd.DataFrame:
    from .lookup import COLUMNS

    rows = []
    for oid in ids:
        if not oid.startswith("WSYN"):
            continue
        k = int(oid[-2:]) if oid[-2:].isdigit() else 0
        survey = k in (3, 7)
        rows.append({"openalex_id": oid, "title": (f"A Survey of Synthetic Topic {oid[4:-2]}" if survey else
                                                   f"Foundational Work {k} on Topic {oid[4:-2]}"),
                     "year": 2000 + k, "cited_by_count": 5000 // (k + 1), "doi": "", "arxiv_id": "",
                     "type": "review" if survey else "article", "venue": "Synthetic Journal",
                     "abstract": "Synthetic foundational paper used for offline tests.",
                     "authors": "A. Founder", "url": f"https://openalex.org/{oid}", "fetched_at": 0.0})
    return pd.DataFrame(rows, columns=COLUMNS)
