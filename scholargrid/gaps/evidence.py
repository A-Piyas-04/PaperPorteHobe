"""Evidence construction shared by every detector, so URLs, years and anchors are consistent."""
from __future__ import annotations

from typing import Any, List

import pandas as pd

from .model import Evidence


def evidence_for(ctx: Any, arxiv_id: str, role: str, *, quote: str = "", section: str = "", anchor: str = "",
                 note: str = "") -> Evidence:
    r = ctx.paper_row(arxiv_id)
    if r is None:
        hit = ctx.all_papers[ctx.all_papers["arxiv_id"].astype(str) == str(arxiv_id)]
        if hit.empty:
            raise KeyError(f"paper {arxiv_id} not in the bundle")
        r = hit.iloc[0]
    anchor_url = f"https://arxiv.org/html/{arxiv_id}#{anchor}" if anchor and anchor != "abstract" else ""
    try:
        year = int(pd.Timestamp(r["date"]).year)
    except (TypeError, ValueError):
        year = None
    return Evidence(eid="", paper_id=str(arxiv_id), title=str(r["title"]), year=year,
                    url=f"https://arxiv.org/abs/{arxiv_id}", role=role, quote=str(quote or "")[:300],  # type: ignore[arg-type]
                    section=section, anchor_url=anchor_url, note=note)


def number_evidence(items: List[Evidence], limit: int = 8) -> List[Evidence]:
    """Dedupe by (paper_id, role), cap, assign E1..En."""
    seen: set = set()
    out: List[Evidence] = []
    for e in items:
        k = (e.paper_id, e.role)
        if k in seen:
            continue
        seen.add(k)
        out.append(e.model_copy(update={"eid": f"E{len(out) + 1}"}))
        if len(out) >= limit:
            break
    return out
