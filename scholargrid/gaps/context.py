"""Everything a detector may read for one scope. Detectors never touch disk or network."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..utils import l2_normalize


@dataclass
class GapContext:
    cfg: Any
    scope: Dict                         # {"kind", "id", "label"}
    papers: pd.DataFrame                # bundle rows in scope (keeps original bundle index in column "bidx")
    vectors: np.ndarray                 # unit vectors aligned with papers
    facets: pd.DataFrame                # arxiv_id, facet, value, role (canonical)
    problems: pd.DataFrame              # problems in scope (+ "vrow": row into problem_vectors)
    problem_vectors: Optional[np.ndarray]
    findings: pd.DataFrame
    profiles: pd.DataFrame
    ref_date: pd.Timestamp
    all_papers: pd.DataFrame = field(default_factory=pd.DataFrame)   # whole bundle df (bridge/novelty)
    all_vectors: Optional[np.ndarray] = None                         # unit float32, aligned with all_papers
    value_vectors: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)  # facet -> value -> unit vec
    sub_labels: Optional[np.ndarray] = None   # sub-group labels within scope (areas / sub-topics) for bridges
    sub_names: Dict[int, str] = field(default_factory=dict)
    facet_aliases: Dict[str, Dict[str, str]] = field(default_factory=dict)   # facet -> raw -> canonical
    _rows: Dict[str, int] = field(default_factory=dict, repr=False)

    def paper_row(self, arxiv_id: str) -> Optional[pd.Series]:
        if not self._rows and len(self.papers):
            self._rows = {str(a): i for i, a in reversed(list(enumerate(self.papers["arxiv_id"].astype(str))))}
        i = self._rows.get(str(arxiv_id))
        return None if i is None else self.papers.iloc[i]


def _aliases(vocab: Dict) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for facet, values in (vocab or {}).items():
        m: Dict[str, str] = {}
        for canon, info in (values or {}).items():
            m[canon] = canon
            for a in (info or {}).get("aliases", []):
                m[str(a)] = canon
        out[facet] = m
    return out


def build_context(bundle: Any, idx: np.ndarray, scope: Dict, cfg: Any, embedder: Any = None,
                  sub_labels: Optional[np.ndarray] = None, sub_names: Optional[Dict[int, str]] = None,
                  all_vectors: Optional[np.ndarray] = None) -> GapContext:
    idx = np.asarray(idx, dtype=int)
    papers = bundle.df.iloc[idx].copy()
    papers["bidx"] = idx
    ids = set(papers["arxiv_id"].astype(str))

    def sel(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["arxiv_id"].astype(str).isin(ids)].reset_index(drop=True) if len(df) else df

    probs = bundle.problems.copy() if len(bundle.problems) else pd.DataFrame()
    pvec = bundle.problem_embeddings
    if len(probs):
        probs["vrow"] = np.arange(len(probs))
        if pvec is None and embedder is not None:
            pvec = embedder.transform(probs["statement"].astype(str).tolist())
        probs = probs[probs["arxiv_id"].astype(str).isin(ids)].reset_index(drop=True)
    if all_vectors is None:
        all_vectors = l2_normalize(np.asarray(bundle.embeddings, dtype=np.float32))
    facets = sel(bundle.facets)
    ctx = GapContext(cfg, scope, papers.reset_index(drop=True), np.asarray(all_vectors[idx], dtype=np.float32),
                     facets, probs, None if pvec is None else l2_normalize(np.asarray(pvec, dtype=np.float32)),
                     sel(bundle.findings), sel(bundle.profiles), pd.Timestamp(bundle.df["date"].max()),
                     bundle.df, all_vectors, sub_labels=sub_labels, sub_names=dict(sub_names or {}),
                     facet_aliases=_aliases(getattr(bundle, "facet_vocab", {}) or {}))
    if embedder is not None and len(facets):
        col = "value" if "value" in facets.columns else "value_raw"
        for facet in ("task", "method", "domain", "dataset", "language", "modality"):
            vals = facets.loc[facets["facet"] == facet, col].astype(str).value_counts().head(200).index.tolist()
            if vals:
                vecs = l2_normalize(np.asarray(embedder.transform(vals), dtype=np.float32))
                ctx.value_vectors[facet] = dict(zip(vals, vecs))
    if "value" not in ctx.facets.columns and len(ctx.facets):
        ctx.facets = ctx.facets.assign(value=ctx.facets["value_raw"])
    return ctx
