"""Opportunity view: areas placed by crowdedness x growth, sized by open problems, with plain advice.

Growth is used only where the growth statistics call it reliable; everything else is "no reliable
growth signal" and can never be called growing.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

QUADRANTS: Dict[str, str] = {
    "Emerging": "Growing and not yet crowded. Good entry point if the problems are feasible.",
    "Hot & competitive": "Many groups publish here. Pick a sharp niche, like an evaluation gap or a specific setting.",
    "Niche": "Fewer papers. Check that the problems matter (citations, venues) before committing.",
    "Mature": "Established area. Reproductions, evaluation gaps and unsettled questions are realistic first projects.",
}


def quadrant(crowdedness: float, reliable: bool, ci_low: Optional[float]) -> str:
    crowded = crowdedness >= 0.5
    growing = bool(reliable) and ci_low is not None and float(ci_low) > 1
    if growing:
        return "Hot & competitive" if crowded else "Emerging"
    return "Mature" if crowded else "Niche"


def _area_name(info: Dict, cid: int) -> str:
    from .labeling import pretty_label

    return pretty_label(info["keywords"]) if info.get("keywords") else info.get("label", f"Area {cid}")


def opportunity_table(bundle: Any) -> pd.DataFrame:
    """One row per area: name, size, recent_papers (last 12 months), crowdedness (recent_papers percentile
    0..1), growth (relative growth if reliable else NaN), growth_ci, reliable, open_problems (gap cards in
    bundle.gaps), stated_per_100 (verified stated problems per 100 extracted papers), quadrant."""
    df = bundle.df
    labels = np.asarray(bundle.labels)
    ref = pd.Timestamp(df["date"].max()) if len(df) else pd.Timestamp.now()
    recent_mask = (pd.to_datetime(df["date"]) >= ref - pd.DateOffset(months=12)).to_numpy()
    growth = bundle.growth or {}
    dw = str(growth.get("default_window", 6))
    gaps = (getattr(bundle, "gaps", None) or {}).get("areas") or {}
    problems = getattr(bundle, "problems", pd.DataFrame())
    profiles = getattr(bundle, "profiles", pd.DataFrame())
    prob_by_paper = problems["arxiv_id"].astype(str).value_counts() if len(problems) else pd.Series(dtype=int)
    extracted = set(profiles["arxiv_id"].astype(str)) if len(profiles) else set()
    ids = df["arxiv_id"].astype(str).to_numpy()
    rows = []
    for cid, info in sorted(bundle.clusters_meta.items()):
        mask = labels == int(cid)
        g = (growth.get("clusters") or {}).get(cid) or (growth.get("clusters") or {}).get(str(cid)) or {}
        w = (g.get("windows") or {}).get(dw) or {}
        reliable = bool(g.get("reliable")) and w.get("relative_growth") is not None
        ci = w.get("ci") if reliable else None
        area_ids = [a for a in ids[mask] if a in extracted]
        stated = int(sum(int(prob_by_paper.get(a, 0)) for a in area_ids))
        rows.append({"area_id": int(cid), "name": _area_name(info, int(cid)), "size": int(mask.sum()),
                     "recent_papers": int((mask & recent_mask).sum()),
                     "growth": float(w["relative_growth"]) if reliable else np.nan, "growth_ci": ci,
                     "reliable": reliable, "open_problems": len(gaps.get(str(int(cid)), [])),
                     "stated_per_100": round(100 * stated / len(area_ids), 1) if area_ids else np.nan})
    out = pd.DataFrame(rows, columns=["area_id", "name", "size", "recent_papers", "growth", "growth_ci", "reliable",
                                      "open_problems", "stated_per_100"])
    if len(out):
        out["crowdedness"] = out["recent_papers"].rank(pct=True, method="average").round(3)
        out["quadrant"] = [quadrant(c, r, (ci[0] if ci else None))
                           for c, r, ci in zip(out["crowdedness"], out["reliable"], out["growth_ci"])]
    else:
        out["crowdedness"] = pd.Series(dtype=float)
        out["quadrant"] = pd.Series(dtype=str)
    return out
