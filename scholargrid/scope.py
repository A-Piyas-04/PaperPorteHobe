"""Selects the papers that receive expensive processing (full text, LLM extraction).

Free LLM quota is counted in requests per day, so the scope decides both *which* papers
get the deep treatment and *in which order* (most useful first when quota runs out).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def scope_mask(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    s = cfg["scope"]
    mask = np.ones(len(df), dtype=bool)
    if s["categories"] != "all":
        mask &= df["primary_category"].astype(str).isin(set(s["categories"])).to_numpy()
    if s.get("min_date"):
        mask &= (pd.to_datetime(df["date"]) >= pd.Timestamp(s["min_date"])).to_numpy()
    return mask


def select_scope(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Papers in scope, ordered by ``scope.priority`` (so limited quota goes to the most useful
    papers first) and capped at ``scope.max_papers``."""
    s = cfg["scope"]
    out = df.loc[scope_mask(df, cfg)]
    if s["priority"] == "most_cited" and "cited_by_count" in out.columns:
        out = out.sort_values(["cited_by_count", "date"], ascending=[False, False], kind="stable")
    elif s["priority"] == "newest" or (s["priority"] == "most_cited" and "date" in out.columns):
        out = out.sort_values("date", ascending=False, kind="stable")
    elif s["priority"] == "random":
        out = out.sample(frac=1.0, random_state=int(cfg["seed"]))
    if s.get("max_papers"):
        out = out.head(int(s["max_papers"]))
    return out
