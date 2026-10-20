"""Method x task (and similar) holes that are statistically unexpected AND plausibly bridgeable.

For a value pair (a, b) with marginals r_a and c_b among N papers that have both facet types, the
expected co-occurrence under independence is E = r_a * c_b / N; with observed O <= max_observed,
p = PoissonCDF(O; E). With ``require_bridge`` a candidate also needs a bridge: a is used for a task
similar to b, or b is approached with a method similar to a. Without one it is usually two unrelated
communities, not a gap.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
from scipy.stats import poisson

from ..evidence import evidence_for, number_evidence
from ..model import GapCandidate

_GENERIC = {"general", "other", "", "unknown", "none"}


def _incidence(F: pd.DataFrame, facet: str, top: int, exclude_baseline: bool) -> Tuple[List[str], Dict[str, set]]:
    if F is None or not len(F):
        return [], {}
    sub = F[F["facet"] == facet]
    if exclude_baseline and "role" in sub:
        sub = sub[sub["role"] != "baseline"]
    sub = sub[~sub["value"].astype(str).str.lower().isin(_GENERIC)]
    if sub.empty:
        return [], {}
    counts = sub.groupby("value")["arxiv_id"].nunique().sort_values(ascending=False).head(top)
    papers = {str(v): set(sub.loc[sub["value"] == v, "arxiv_id"].astype(str)) for v in counts.index}
    return [str(v) for v in counts.index], papers


def _bridge(ctx: Any, facet: str, target: str, partners: List[str]) -> Tuple[str, float]:
    vv = ctx.value_vectors.get(facet, {})
    if target not in vv:
        return "", 0.0
    best, s = "", 0.0
    for p in partners:
        if p in vv and p != target:
            sim = float(vv[target] @ vv[p])
            if sim > s:
                best, s = p, sim
    return best, s


def detect_combinatorial(ctx: Any) -> List[GapCandidate]:
    g = ctx.cfg["gaps"]
    out: List[GapCandidate] = []
    for pair in g["combo_pairs"]:
        fa, fb = pair.split(":")
        A_vals, A = _incidence(ctx.facets, fa, int(g["combo_top_values"]), fa == "method")
        B_vals, B = _incidence(ctx.facets, fb, int(g["combo_top_values"]), fb == "method")
        has_a = set().union(*A.values()) if A else set()
        has_b = set().union(*B.values()) if B else set()
        universe = has_a & has_b
        N = len(universe)
        if N < 10:
            continue
        for a in A_vals:
            ra = len(A[a] & universe)
            if ra < int(g["min_marginal"]):
                continue
            used_with = [b for b in B_vals if A[a] & B[b]]             # b' that a is already used for
            for b in B_vals:
                cb = len(B[b] & universe)
                if cb < int(g["min_marginal"]) or a.lower() in b.lower() or b.lower() in a.lower():
                    continue
                obs = len(A[a] & B[b])
                exp = ra * cb / N
                if obs > int(g["max_observed"]) or exp < float(g["min_expected"]):
                    continue
                p = float(poisson.cdf(obs, exp))
                if p > float(g["max_p"]):
                    continue
                b_sim_name, b_sim = _bridge(ctx, fb, b, used_with)
                approaches_b = [x for x in A_vals if A[x] & B[b]]
                a_sim_name, a_sim = _bridge(ctx, fa, a, approaches_b)
                has_bridge = max(b_sim, a_sim) >= float(g["min_bridge_sim"])
                if g["require_bridge"] and not has_bridge:
                    continue
                ev = []
                if b_sim_name and b_sim >= a_sim:
                    for pid in sorted(A[a] & B[b_sim_name])[:2]:
                        ev.append(evidence_for(ctx, pid, "bridge", note=f"applies {a} to {b_sim_name}"))
                elif a_sim_name:
                    for pid in sorted(A[a_sim_name] & B[b])[:2]:
                        ev.append(evidence_for(ctx, pid, "bridge", note=f"applies {a_sim_name} to {b}"))
                for pid in sorted(A[a] - B[b])[:2]:
                    ev.append(evidence_for(ctx, pid, "method_use", note=f"uses {a}"))
                top_b = ctx.papers[ctx.papers["arxiv_id"].astype(str).isin(B[b])].sort_values(
                    "cited_by_count", ascending=False)
                for pid in top_b["arxiv_id"].head(3):
                    ev.append(evidence_for(ctx, str(pid), "task_use", note=f"works on {b}"))
                if b_sim_name and b_sim >= a_sim:
                    bridge_txt = f"{a} is already used for the related {b_sim_name}"
                else:
                    bridge_txt = f"{b} is studied with the related {a_sim_name}"
                out.append(GapCandidate(
                    type="unexplored_combination", scope=ctx.scope, key_terms={fa: a, fb: b},
                    seed_statement=f"No paper in this collection uses {a} for {b}, although {bridge_txt}.",
                    evidence=number_evidence(ev),
                    stats={"pair": pair, "observed": obs, "expected": round(exp, 2), "p_value": round(p, 4),
                           f"n_{fa}": ra, f"n_{fb}": cb, "n_papers": N, "bridge": b_sim_name or a_sim_name,
                           "bridge_sim": round(max(a_sim, b_sim), 3)},
                    specificity=0.8, strength=float(min(1.0, (1 - p) * min(1.0, exp / 10)))))
    out.sort(key=lambda c: (c.stats["p_value"], -c.stats["expected"]))
    return out
