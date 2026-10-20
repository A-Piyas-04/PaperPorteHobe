"""Missing bridges: semantically close groups (areas offline, sub-topics online) that rarely cite each other.

Expected cross-citations from g to h are proportional to h's share of the scope's papers with an OpenAlex
id: exp = refs_out[g] * |h| / N. ratio = (cross[g->h] + cross[h->g]) / (exp_gh + exp_hg). Groups need
reference coverage >= ``bridge_min_ref_coverage``; a pair is a candidate when ratio <= ``bridge_max_ratio``.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..evidence import evidence_for, number_evidence
from ..model import GapCandidate

_MIN_EXPECTED = 2.0          # below this the ratio is noise


def detect_bridge(ctx: Any) -> List[GapCandidate]:
    g = ctx.cfg["gaps"]
    labels = ctx.sub_labels
    P = ctx.papers
    if labels is None or not len(P) or "references" not in P.columns or "openalex_id" not in P.columns:
        return []
    labels = np.asarray(labels)
    groups = [int(c) for c in np.unique(labels) if c >= 0]
    if len(groups) < 2:
        return []
    cents: Dict[int, np.ndarray] = {}
    for c in groups:
        v = ctx.vectors[labels == c].mean(axis=0)
        cents[c] = v / (np.linalg.norm(v) + 1e-9)
    pairs = [(a, b, float(cents[a] @ cents[b])) for i, a in enumerate(groups) for b in groups[i + 1:]]
    if not pairs:
        return []
    threshold = float(np.quantile([s for _, _, s in pairs], float(g["bridge_sim_quantile"])))
    oa = P["openalex_id"].fillna("").astype(str).to_numpy()
    group_of = {o: int(labels[i]) for i, o in enumerate(oa) if o and labels[i] >= 0}
    n_with_id = len(group_of)
    if not n_with_id:
        return []
    refs = [str(r).split() if isinstance(r, str) and r.strip() else [] for r in P["references"].tolist()]
    size = {c: int(sum(1 for o, lab in zip(oa, labels) if o and lab == c)) for c in groups}
    coverage = {c: float(np.mean([bool(refs[i]) for i in np.where(labels == c)[0]])) for c in groups}
    refs_out = {c: 0 for c in groups}
    cross: Dict[tuple, int] = {}
    for i, rl in enumerate(refs):
        src = int(labels[i])
        if src < 0:
            continue
        for w in rl:
            dst = group_of.get(w)
            if dst is None:
                continue
            refs_out[src] += 1
            if dst != src:
                cross[(src, dst)] = cross.get((src, dst), 0) + 1
    out: List[GapCandidate] = []
    min_cov = float(g["bridge_min_ref_coverage"])
    for a, b, sim in pairs:
        if sim < threshold or coverage[a] < min_cov or coverage[b] < min_cov:
            continue
        exp_ab = refs_out[a] * size[b] / n_with_id
        exp_ba = refs_out[b] * size[a] / n_with_id
        expected = exp_ab + exp_ba
        if expected < _MIN_EXPECTED:
            continue
        observed = cross.get((a, b), 0) + cross.get((b, a), 0)
        ratio = observed / expected
        if ratio > float(g["bridge_max_ratio"]):
            continue
        name_a = ctx.sub_names.get(a, f"Group {a}")
        name_b = ctx.sub_names.get(b, f"Group {b}")
        ev = []
        for side, c in (("side_a", a), ("side_b", b)):
            rows = np.where(labels == c)[0]
            central = rows[np.argsort(-(ctx.vectors[rows] @ cents[c]), kind="stable")[:2]]
            ev += [evidence_for(ctx, str(P.iloc[int(j)]["arxiv_id"]), side,
                                note=f"central paper of '{ctx.sub_names.get(c, c)}'") for j in central]
        out.append(GapCandidate(
            type="missing_bridge", scope=ctx.scope, key_terms={"area_a": name_a, "area_b": name_b},
            seed_statement=(f"{name_a} and {name_b} study closely related ideas (similarity {sim:.2f}) but cite "
                            f"each other {ratio:.0%} as often as expected."),
            evidence=number_evidence(ev),
            stats={"similarity": round(sim, 3), "ratio": round(ratio, 3), "observed": int(observed),
                   "expected": round(expected, 2), "coverage_a": round(coverage[a], 3),
                   "coverage_b": round(coverage[b], 3)},
            specificity=0.5, strength=float(max(0.0, 1 - ratio))))
    return out
