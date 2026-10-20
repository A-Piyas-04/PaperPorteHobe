"""Ordered reading path: surveys -> foundations -> key recent -> frontier.

One function serves both offline (per area) and online (per workspace subset). Items are de-duplicated
across steps (a paper appears once, in its earliest step). ``papers`` must have a default RangeIndex
(``load_papers`` produces one), so a row label is its bundle index.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .surveys import is_survey


def _year(value: Any) -> Optional[int]:
    try:
        y = int(value)
    except (TypeError, ValueError):
        return None
    return y if y > 0 else None


def _age_years(year: Any, ref_year: int) -> float:
    y = _year(year)
    return 5.0 if y is None else max(0.0, float(ref_year - y))


def _item(step: str, *, paper_id: str, title: str, year: Any, url: str, cites: Any, reason: str,
          in_corpus: bool, bundle_index: Optional[int] = None, survey: bool = False,
          openalex_id: str = "", abstract: str = "") -> Dict:
    try:
        n_cites = int(cites or 0)
    except (TypeError, ValueError):
        n_cites = 0
    return {"step": step, "paper_id": paper_id, "title": str(title), "year": _year(year),
            "url": url, "cited_by_count": n_cites, "reason": reason, "in_corpus": bool(in_corpus),
            "bundle_index": bundle_index, "is_survey": survey, "openalex_id": openalex_id,
            "abstract": str(abstract or "")[:1200]}


def reading_path(papers: pd.DataFrame, idx: np.ndarray, embeddings: np.ndarray, works: pd.DataFrame,
                 backbone_rows: pd.DataFrame, problems_by_paper: Dict[str, int], sizes: Dict[str, int],
                 recent_months: int, frontier_months: int, n_base: int) -> List[Dict]:
    """``papers`` = full bundle df; ``idx`` = bundle row indices of the subset (area or workspace);
    ``works`` = WorkLookup rows for backbone ids; ``backbone_rows`` = scored backbone for the subset."""
    idx = np.asarray(idx, dtype=int)
    if len(idx) == 0:
        return []
    sub = papers.iloc[idx]
    ref = pd.Timestamp(papers["date"].max())
    ref_year = int(ref.year)
    seen: set = set()
    out: List[Dict] = []
    corpus_ids = set(papers["arxiv_id"].astype(str))

    def add(item: Dict) -> None:
        key = item["paper_id"]
        if key and key not in seen:
            seen.add(key)
            out.append(item)

    w_by_id = works.drop_duplicates("openalex_id").set_index("openalex_id") if len(works) else pd.DataFrame()
    known = set(w_by_id.index) if len(w_by_id) else set()
    bb = list(backbone_rows.itertuples()) if len(backbone_rows) else []

    # 1) surveys: in-subset surveys + backbone works that are reviews
    cand = []
    for i, r in zip(idx, sub.itertuples()):
        if is_survey(r.title, r.abstract):
            y = pd.Timestamp(r.date).year
            cand.append((int(r.cited_by_count) / (_age_years(y, ref_year) + 1), _item(
                "survey", paper_id=str(r.arxiv_id), title=r.title, year=y, url=f"https://arxiv.org/abs/{r.arxiv_id}",
                cites=int(r.cited_by_count), reason=f"Survey from {y} covering this topic "
                f"({int(r.cited_by_count)} citations).", in_corpus=True, bundle_index=int(i), survey=True,
                abstract=r.abstract)))
    for b in bb:
        if b.openalex_id in known:
            w = w_by_id.loc[b.openalex_id]
            if is_survey(w["title"], w["abstract"], w["type"]):
                cand.append((float(w["cited_by_count"] or 0) / (_age_years(w["year"], ref_year) + 1), _item(
                    "survey", paper_id=w["arxiv_id"] or b.openalex_id, title=w["title"], year=w["year"],
                    url=w["url"], cites=w["cited_by_count"], reason=f"Survey cited by {b.freq} papers here.",
                    in_corpus=bool(w["arxiv_id"] and w["arxiv_id"] in corpus_ids), survey=True,
                    openalex_id=b.openalex_id, abstract=w["abstract"])))
    for _, it in sorted(cand, key=lambda t: -t[0])[: sizes.get("survey", 3)]:
        add(it)

    # 2) foundations: backbone by score, displayed oldest first
    fnd = []
    for b in bb:
        if b.openalex_id not in known:
            continue
        w = w_by_id.loc[b.openalex_id]
        if not w["title"]:
            continue
        pid = w["arxiv_id"] or b.openalex_id
        if pid in seen:
            continue
        fnd.append(_item("foundation", paper_id=pid, title=w["title"], year=w["year"], url=w["url"],
                         cites=w["cited_by_count"],
                         reason=f"Cited by {b.freq} of the {n_base} papers here, a shared foundation.",
                         in_corpus=bool(w["arxiv_id"] and w["arxiv_id"] in corpus_ids), survey=False,
                         openalex_id=b.openalex_id, abstract=w["abstract"]))
    for it in sorted(fnd[: sizes.get("foundation", 6)], key=lambda x: x["year"] or 0):
        add(it)

    # 3) key recent: last recent_months, by citation velocity (fallback citations per month)
    dates = pd.to_datetime(sub["date"])
    rec = sub[dates >= ref - pd.DateOffset(months=recent_months)]
    if len(rec):
        months = ((ref - pd.to_datetime(rec["date"])).dt.days / 30.4).clip(lower=1)
        vel = rec["citation_velocity"].astype(float) if "citation_velocity" in rec else pd.Series(0.0, index=rec.index)
        rate = np.where(vel > 0, vel, rec["cited_by_count"].astype(float) / months)
        if float(np.max(rate)) > 0:          # without citation data "most-cited" would be meaningless
            for pos in np.argsort(-rate, kind="stable")[: sizes.get("recent", 6)]:
                r = rec.iloc[int(pos)]
                if float(rate[int(pos)]) <= 0:
                    break
                add(_item("recent", paper_id=str(r["arxiv_id"]), title=r["title"], year=pd.Timestamp(r["date"]).year,
                          url=f"https://arxiv.org/abs/{r['arxiv_id']}", cites=int(r["cited_by_count"]),
                          reason=f"One of the most-cited recent papers here ({float(rate[int(pos)]):.1f} "
                                 "citations/month).",
                          in_corpus=True, bundle_index=int(r.name), abstract=r["abstract"]))

    # 4) frontier: last frontier_months, central to the subset, prefer stated open problems
    fmask = (dates >= ref - pd.DateOffset(months=frontier_months)).to_numpy()
    if fmask.any():
        e = np.asarray(embeddings[idx], dtype=np.float32)
        centroid = e.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        sims = e @ centroid
        fidx = np.where(fmask)[0]
        bonus = np.array([0.05 if problems_by_paper.get(str(sub.iloc[j]["arxiv_id"]), 0) else 0.0 for j in fidx])
        taken = 0
        for j in fidx[np.argsort(-(sims[fidx] + bonus), kind="stable")]:
            if taken >= sizes.get("frontier", 5):
                break
            r = sub.iloc[int(j)]
            if str(r["arxiv_id"]) in seen:
                continue
            has_p = problems_by_paper.get(str(r["arxiv_id"]), 0)
            add(_item("frontier", paper_id=str(r["arxiv_id"]), title=r["title"], year=pd.Timestamp(r["date"]).year,
                      url=f"https://arxiv.org/abs/{r['arxiv_id']}", cites=int(r["cited_by_count"]),
                      reason=(f"Recent ({pd.Timestamp(r['date']):%b %Y}) and central to this topic"
                              + ("; it lists open problems you can build on." if has_p else ".")),
                      in_corpus=True, bundle_index=int(idx[int(j)]), abstract=r["abstract"]))
            taken += 1
    return out
