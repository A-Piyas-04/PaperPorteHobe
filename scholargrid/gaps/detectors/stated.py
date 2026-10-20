"""Recurring author-stated open problems. A cluster of similar problem statements from >= min_support
distinct papers is a candidate. Later papers that look like they address it become 'related_attempt'
evidence (the novelty step makes the final call)."""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd

from ..evidence import evidence_for, number_evidence
from ..model import GapCandidate

_SECTION_NAMES = {"limitations": "Limitations", "future_work": "Future work", "discussion": "Discussion",
                  "conclusion": "Conclusion", "threats_to_validity": "Threats to validity",
                  "broader_impact": "Broader impact", "abstract": "Abstract", "introduction": "Introduction"}


def _cluster(x: np.ndarray, distance: float, seed: int) -> np.ndarray:
    """Average-linkage cosine clustering; k-means buckets first above 4,000 rows (O(n^2) memory)."""
    from sklearn.cluster import AgglomerativeClustering, KMeans

    if len(x) < 2:
        return np.zeros(len(x), dtype=int)
    if len(x) <= 4000:
        return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                       distance_threshold=distance).fit_predict(x)
    buckets = KMeans(n_clusters=max(2, len(x) // 1500), n_init=4, random_state=seed).fit_predict(x)
    out, offset = np.full(len(x), -1), 0
    for b in np.unique(buckets):
        idx = np.where(buckets == b)[0]
        sub = _cluster(x[idx], distance, seed)
        out[idx] = sub + offset
        offset += int(sub.max()) + 1
    return out


def _author_groups(ctx: Any, ids: List[str]) -> int:
    """Number of author-disjoint groups (greedy): two papers sharing an author id count once."""
    groups: List[set] = []
    for aid in ids:
        r = ctx.paper_row(aid)
        raw = str(r.get("author_ids", "") or "") if r is not None else ""
        authors = set(raw.split()) if raw.strip() else {aid}
        for g in groups:
            if g & authors:
                g |= authors
                break
        else:
            groups.append(set(authors))
    return len(groups)


def detect_stated(ctx: Any) -> List[GapCandidate]:
    g = ctx.cfg["gaps"]
    P = ctx.problems
    if P is None or len(P) < int(g["min_support"]) or ctx.problem_vectors is None:
        return []
    X = ctx.problem_vectors[P["vrow"].to_numpy()]
    nonzero = np.linalg.norm(X, axis=1) > 1e-6          # TF-IDF gives zero vectors for out-of-vocabulary text
    if not nonzero.all():
        P, X = P[nonzero].reset_index(drop=True), X[nonzero]
        if len(P) < int(g["min_support"]):
            return []
    labels = _cluster(X, float(g["problem_cluster_distance"]), int(ctx.cfg["seed"]))
    date_of = ctx.papers.drop_duplicates("arxiv_id").set_index("arxiv_id")["date"]
    dates = pd.to_datetime(P["arxiv_id"].map(date_of))
    paper_dates = pd.to_datetime(ctx.papers["date"])
    out: List[GapCandidate] = []
    for c in np.unique(labels):
        mask = labels == c
        rows = P[mask]
        papers = list(dict.fromkeys(rows["arxiv_id"].astype(str)))
        if len(papers) < int(g["min_support"]):
            continue
        cx = X[mask]
        centroid = cx.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        medoid = rows.iloc[int(np.argmax(cx @ centroid))]
        d = dates[mask]
        first, last = d.min(), d.max()
        months_open = (ctx.ref_date - first).days / 30.4
        recent = int((d >= ctx.ref_date - pd.DateOffset(months=12)).sum())
        long_standing = months_open >= int(g["longstanding_min_months"]) and recent >= 1
        # Later papers that may address it (scope only): similar abstract, dated after `first`.
        later = (paper_dates > first).to_numpy() & ~ctx.papers["arxiv_id"].astype(str).isin(papers).to_numpy()
        sims = ctx.vectors @ centroid if len(ctx.vectors) and ctx.vectors.shape[1] == len(centroid) else \
            np.zeros(len(ctx.papers))
        cand = np.where(later & (sims >= float(g["addressed_sim"])))[0]
        attempts = cand[np.argsort(-sims[cand], kind="stable")][:3]
        ev = []
        best = rows.sort_values(["specificity", "verified_ratio"], ascending=False).drop_duplicates("arxiv_id")
        for r in best.head(5).itertuples():
            ev.append(evidence_for(ctx, str(r.arxiv_id), "states_problem", quote=str(r.quote),
                                   section=_SECTION_NAMES.get(str(r.section_kind), str(r.section_kind)),
                                   anchor=str(r.anchor or "")))
        for j in attempts:
            ev.append(evidence_for(ctx, str(ctx.papers.iloc[int(j)]["arxiv_id"]), "related_attempt",
                                   note=f"similar topic (similarity {float(sims[j]):.2f}), published later"))
        support = len(papers)
        out.append(GapCandidate(
            type="long_standing_problem" if long_standing else "stated_open_problem",
            scope=ctx.scope, key_terms={"problem": str(medoid["statement"])[:120]},
            seed_statement=str(medoid["statement"]), evidence=number_evidence(ev),
            stats={"support": support, "author_groups": _author_groups(ctx, papers),
                   "first_seen": f"{first:%Y-%m}", "last_seen": f"{last:%Y-%m}", "recent_support": recent,
                   "months_open": round(months_open, 1), "related_attempts": int(len(attempts)),
                   "problem_types": {str(k): int(v) for k, v in rows["problem_type"].value_counts().items()}},
            specificity=float(rows["specificity"].astype(float).mean()) / 3.0,
            strength=float(1 - np.exp(-support / 3.0)),
            problem_ids=rows["problem_id"].astype(str).tolist()))
    return sorted(out, key=lambda c: -c.strength)
