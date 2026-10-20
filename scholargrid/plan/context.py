"""Computed context for one gap: numbered references [R#], candidate baselines/datasets/metrics, venues,
people and code links. All of it is computed in Python; the LLM only chooses among the references.

People and venues are computed only from the related set (the referenced collection papers) and are
never sent to the LLM: they are facts to display, not material to embellish.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import l2_normalize


def _first_sentence(text: str) -> str:
    t = str(text or "").strip()
    return (t.split(". ")[0] + ".") if t else ""


def _facets_by_paper(bundle: Any, ids: List[str]) -> Dict[str, Dict[str, List[str]]]:
    F = getattr(bundle, "facets", None)
    out: Dict[str, Dict[str, List[str]]] = {i: {"methods": [], "datasets": [], "metrics": []} for i in ids}
    if F is None or not len(F):
        return out
    col = "value" if "value" in F.columns else "value_raw"
    sub = F[F["arxiv_id"].astype(str).isin(set(ids))]
    for r in sub.itertuples():
        d = out.get(str(r.arxiv_id))
        if d is None:
            continue
        v = str(getattr(r, col))
        if r.facet == "method" and r.role == "proposed" and v not in d["methods"]:
            d["methods"].append(v)
        elif r.facet == "dataset" and v not in d["datasets"]:
            d["datasets"].append(v)
        elif r.facet == "metric" and v not in d["metrics"]:
            d["metrics"].append(v)
    return out


def _scope_indices(card: Any, bundle: Any, workspace: Any) -> np.ndarray:
    if workspace is not None:
        return np.array(sorted(set(workspace.bundle_indices())), dtype=int)
    scope = card.scope or {}
    if scope.get("kind") == "area":
        try:
            return np.where(bundle.labels == int(str(scope.get("id"))))[0]
        except (TypeError, ValueError):
            pass
    return np.arange(len(bundle.df))


def _reading_items(card: Any, bundle: Any, workspace: Any, n: int) -> List[Dict]:
    if workspace is not None:
        path = list(workspace.reading_path)
    else:
        path = list((getattr(bundle, "reading_paths", None) or {}).get(str((card.scope or {}).get("id")), []))
    return [p for p in path if p.get("step") in ("foundation", "survey")][:n]


def plan_context(card: Any, bundle: Any, cfg: Config, embedder: Any, workspace: Any = None) -> Dict:
    p = cfg["plan"]
    df = bundle.df
    row_of = {str(a): i for i, a in reversed(list(enumerate(df["arxiv_id"].astype(str))))}
    refs: List[Dict] = []
    seen: set = set()

    def add(pid: str, role: str, title: str = "", year: Optional[int] = None, url: str = "") -> None:
        if not pid or pid in seen or len(refs) >= int(p["max_references"]):
            return
        seen.add(pid)
        i = row_of.get(pid)
        if i is not None:
            r = df.iloc[i]
            title = title or str(r["title"])
            year = year or int(pd.Timestamp(r["date"]).year)
            url = url or f"https://arxiv.org/abs/{pid}"
        refs.append({"rid": f"R{len(refs) + 1}", "paper_id": pid, "title": title, "year": year, "url": url,
                     "role": role, "in_corpus": i is not None, "bundle_index": i})

    for e in card.evidence:
        add(e.paper_id, e.role, e.title, e.year, e.url)
    evidence_ids = set(seen)
    if embedder is not None and int(p["related_top"]) > 0:
        idx = _scope_indices(card, bundle, workspace)
        idx = np.array([i for i in idx if str(df.iloc[int(i)]["arxiv_id"]) not in evidence_ids], dtype=int)
        if len(idx):
            v = l2_normalize(np.asarray(embedder.transform([card.statement]), dtype=np.float32))[0]
            sims = l2_normalize(np.asarray(bundle.embeddings[idx], dtype=np.float32)) @ v
            for j in idx[np.argsort(-sims, kind="stable")[: int(p["related_top"])]]:
                add(str(df.iloc[int(j)]["arxiv_id"]), "related")
    reading = _reading_items(card, bundle, workspace, int(p["foundations_top"]))
    for it in reading:
        add(str(it.get("paper_id")), it.get("step", "foundation"), str(it.get("title", "")), it.get("year"),
            str(it.get("url", "")))

    corpus_ids = [r["paper_id"] for r in refs if r["in_corpus"]]
    facets = _facets_by_paper(bundle, corpus_ids)
    profiles = getattr(bundle, "profiles", None)
    prof = (profiles.drop_duplicates("arxiv_id").set_index("arxiv_id") if profiles is not None and len(profiles)
            else pd.DataFrame())
    for r in refs:
        f = facets.get(r["paper_id"], {"methods": [], "datasets": [], "metrics": []})
        summary, code = "", ""
        if r["paper_id"] in getattr(prof, "index", []):
            summary = str(prof.loc[r["paper_id"], "summary"] or "")
            code = str(prof.loc[r["paper_id"], "code_url"] or "") if "code_url" in prof.columns else ""
        if not summary and r["in_corpus"]:
            summary = _first_sentence(df.iloc[r["bundle_index"]].get("abstract", ""))
        r.update(summary=summary[:300], methods=f["methods"][:6], datasets=f["datasets"][:6],
                 metrics=f["metrics"][:6], code_url=code)

    computed = _computed(refs, df, row_of, p)
    computed["reading"] = reading
    return {"references": refs, "computed": computed}


def _computed(refs: List[Dict], df: pd.DataFrame, row_of: Dict[str, int], p: Dict) -> Dict:
    ds: Counter = Counter()
    ms: Counter = Counter()
    meth: Counter = Counter()
    for r in refs:
        ds.update(r["datasets"])
        ms.update(r["metrics"])
        meth.update(r["methods"])
    rows = [df.iloc[row_of[r["paper_id"]]] for r in refs if r["paper_id"] in row_of]
    venues: Counter = Counter(str(r.get("venue", "") or "").strip() for r in rows)
    venues.pop("", None)
    people: Counter = Counter()
    link: Dict[str, str] = {}
    for r in rows:
        names = [a.strip() for a in str(r.get("authors", "") or "").split(";") if a.strip()]
        ids = str(r.get("author_ids", "") or "").split()
        for k, name in enumerate(names):
            people[name] += 1
            if len(ids) == len(names) and ids[k]:
                aid = ids[k].rsplit("/", 1)[-1]
                link.setdefault(name, f"https://openalex.org/{aid}")
    return {
        "dataset_counts": [[k, v] for k, v in ds.most_common(10)],
        "metric_counts": [[k, v] for k, v in ms.most_common(10)],
        "method_counts": [[k, v] for k, v in meth.most_common(10)],
        "code_links": [{"paper_id": r["paper_id"], "title": r["title"], "code_url": r["code_url"]}
                       for r in refs if r.get("code_url")],
        "venues": [[k, v] for k, v in venues.most_common(int(p["venues_top"]))],
        "people": [{"name": k, "papers": v, "url": link.get(k, "")} for k, v in people.most_common(int(p["people_top"]))],
        "n_related": len(rows),
    }
