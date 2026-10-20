"""Area-specific co-citation backbone.

For an area A with n_A citing papers: share_A(W) = freq_A(W) / n_A, lift(W) = (share_A + eps) /
(share_all + eps), score(W) = freq_A(W) * log(1 + lift). High lift means W is specific to this area,
not a universal citation like an optimizer paper. Universal citations are not blocklisted: they can be
legitimate foundations for some areas.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BACKBONE_COLUMNS = ["openalex_id", "freq", "n_citing_base", "share", "lift", "score"]


def reference_lists(refs: pd.Series) -> List[List[str]]:
    return [str(r).split() if isinstance(r, str) and r else [] for r in refs.tolist()]


def _score(cnt: Counter, n: int, glob: Counter, n_all: int, min_citing: int, min_share: float,
           top: int, eps: float) -> pd.DataFrame:
    rows = []
    for w, f in cnt.items():
        share = f / n
        if f < min_citing or share < min_share:
            continue
        lift = (share + eps) / (glob[w] / n_all + eps)
        rows.append({"openalex_id": w, "freq": f, "n_citing_base": n, "share": round(share, 4),
                     "lift": round(lift, 3), "score": round(f * float(np.log1p(lift)), 4)})
    if not rows:
        return pd.DataFrame(columns=BACKBONE_COLUMNS)
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(top).reset_index(drop=True)


def backbone(ref_lists: List[List[str]], labels: np.ndarray, min_citing: int, min_share: float,
             top: int, eps: float) -> Dict[int, pd.DataFrame]:
    """{area_id: DataFrame[openalex_id, freq, n_citing_base, share, lift, score]} for every clustered area."""
    has = np.array([bool(r) for r in ref_lists])
    n_all = int(has.sum())
    if n_all == 0:
        return {}
    glob: Counter = Counter()
    for r in ref_lists:
        glob.update(set(r))
    out: Dict[int, pd.DataFrame] = {}
    for cid in sorted({int(c) for c in labels if c != -1}):
        idx = np.where((labels == cid) & has)[0]
        if len(idx) == 0:
            continue
        cnt: Counter = Counter()
        for i in idx:
            cnt.update(set(ref_lists[i]))
        scored = _score(cnt, len(idx), glob, n_all, min_citing, min_share, top, eps)
        if len(scored):
            out[cid] = scored
    return out


def subset_backbone(ref_lists: List[List[str]], idx: np.ndarray, min_citing: int, min_share: float,
                    top: int, eps: float, glob: Optional[Counter] = None, n_all: Optional[int] = None
                    ) -> pd.DataFrame:
    """Backbone for an arbitrary subset (an online topic workspace), with lift against the whole corpus."""
    if glob is None or n_all is None:
        glob = Counter()
        for r in ref_lists:
            glob.update(set(r))
        n_all = sum(1 for r in ref_lists if r)
    sub = [ref_lists[int(i)] for i in idx if ref_lists[int(i)]]
    if not sub or not n_all:
        return pd.DataFrame(columns=BACKBONE_COLUMNS)
    cnt: Counter = Counter()
    for r in sub:
        cnt.update(set(r))
    return _score(cnt, len(sub), glob, n_all, min_citing, min_share, top, eps)
