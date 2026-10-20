"""Evaluation gaps: methods with enough papers but only evaluated in narrow conditions.

Profile shares (real-world, robustness, efficiency) use only *known* values; a share is computed only
when known for >= ``eval_min_known`` of the method's papers. Unknown is never treated as "no".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..evidence import evidence_for, number_evidence
from ..model import GapCandidate

_MULTI_GPU = {"multi_gpu", "large_cluster"}


def _share(profile: pd.DataFrame, col: str, min_known: float) -> Optional[float]:
    if col not in profile.columns or not len(profile):
        return None
    known = profile[col].dropna()
    if len(known) < min_known * len(profile) or not len(known):
        return None
    return float(known.astype(bool).mean())


def detect_evaluation(ctx: Any) -> List[GapCandidate]:
    g = ctx.cfg["gaps"]
    F = ctx.facets
    if F is None or not len(F):
        return []
    methods = F[(F["facet"] == "method") & (F["role"] != "baseline")]
    counts = methods.groupby("value")["arxiv_id"].nunique()
    counts = counts[counts >= int(g["eval_min_papers"])].sort_values(ascending=False)
    out: List[GapCandidate] = []
    datasets_of: Dict[str, List[str]] = (F[F["facet"] == "dataset"].groupby("arxiv_id")["value"]
                                         .apply(lambda s: sorted(set(map(str, s)))).to_dict())
    langs_of = F[F["facet"] == "language"].groupby("arxiv_id")["value"].apply(lambda s: set(map(str, s))).to_dict()
    text_papers = set(F.loc[(F["facet"] == "modality") & (F["value"] == "text"), "arxiv_id"].astype(str))
    min_known = float(g["eval_min_known"])
    for m, n in counts.items():
        ids = sorted(set(methods.loc[methods["value"] == m, "arxiv_id"].astype(str)))
        profile = ctx.profiles[ctx.profiles["arxiv_id"].astype(str).isin(ids)] if len(ctx.profiles) else pd.DataFrame()
        ds = sorted({d for a in ids for d in datasets_of.get(a, [])})
        triggers = []
        if ds and len(ds) <= int(g["eval_max_datasets"]):
            k = len(ds)
            triggers.append(("narrow_benchmarks",
                             f"{m} has been evaluated on only {k} benchmark{'s' if k != 1 else ''} "
                             f"({', '.join(ds)}) across {n} papers."))
        if _share(profile, "eval_real_world", min_known) == 0.0:
            triggers.append(("no_real_world", f"None of the {n} papers on {m} evaluates it on real-world or "
                                              "deployment data."))
        if _share(profile, "eval_robustness", min_known) == 0.0:
            triggers.append(("no_robustness", f"Robustness of {m} under distribution shift or adversarial "
                                              "conditions is untested in these papers."))
        langs = [langs_of[a] for a in ids if a in langs_of]
        if langs and len(langs) >= min_known * len(ids) and set().union(*langs) == {"English"} \
                and any(a in text_papers for a in ids):
            triggers.append(("english_only", f"{m} has only been evaluated on English data."))
        heavy = (profile["compute"].isin(_MULTI_GPU).mean() if "compute" in profile.columns and len(profile)
                 else 0.0)
        if _share(profile, "eval_efficiency_reported", min_known) == 0.0 and heavy >= 0.5:
            triggers.append(("no_efficiency", f"The cost or latency of {m} is not reported."))
        for trig, seed in triggers:
            ev = [evidence_for(ctx, a, "narrow_eval",
                               note="evaluated on: " + (", ".join(datasets_of.get(a, [])) or "not stated"))
                  for a in ids[:5]]
            out.append(GapCandidate(
                type="evaluation_gap", scope=ctx.scope, key_terms={"method": str(m), "trigger": trig},
                seed_statement=seed, evidence=number_evidence(ev),
                stats={"trigger": trig, "n_papers": int(n), "datasets": ds, "n_datasets": len(ds)},
                specificity=0.7, strength=float(min(1.0, n / 10))))
    return out
