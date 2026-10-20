"""build_workspace(query, bundle, searcher, cfg, progress) -> TopicWorkspace (cached on disk).

Steps: retrieve -> vectors -> sub-topics -> labels -> activity -> overview -> reading path -> cache.
The cache key includes the bundle version, so a new release invalidates old workspaces. A workspace
whose LLM labels failed only because the free quota ran out is not cached (``labels_pending``), so the
next visit after the reset retries them.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..config import Config
from ..utils import ensure_dir, get_logger
from . import prompts as _prompts  # noqa: F401  (registers subtopic_label / topic_overview)
from .activity import activity, newest_date
from .labels import evidence_lines, facets_text, keyword_label, keyword_labels, llm_labels, top_facets
from .model import Subtopic, TopicWorkspace
from .path import workspace_reading_path
from .retrieve import retrieve
from .subtopics import subtopics

log = get_logger("workspace")
Progress = Optional[Callable[[str, float], None]]
QUOTA_WARNING = "Summaries unavailable right now (free AI quota used up); showing keyword labels."
LLM_WARNING = "Plain-language summaries are unavailable right now; showing keyword labels."


def workspace_key(query: str, bundle_version: str, cfg: Config) -> str:
    q = " ".join(query.lower().split())
    w = json.dumps(cfg["workspace"], sort_keys=True)
    return hashlib.sha1(f"{bundle_version}|{q}|{w}|{cfg['llm']['provider']}".encode()).hexdigest()[:20]


def load_cached(key: str, cfg: Config) -> Optional[TopicWorkspace]:
    path = os.path.join(cfg.workspace_dir, f"{key}.json")
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > float(cfg["workspace"]["cache_ttl_hours"]) * 3600:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return TopicWorkspace.from_json(json.load(fh))
    except (OSError, ValueError, TypeError) as exc:
        log.warning("Ignoring unreadable workspace cache %s (%s).", path, exc)
        return None


def save_cached(ws: TopicWorkspace, cfg: Config) -> None:
    d = ensure_dir(cfg.workspace_dir)
    tmp = os.path.join(d, f"{ws.key}.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ws.to_json(), fh)
    os.replace(tmp, os.path.join(d, f"{ws.key}.json"))
    files = sorted((os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json")), key=os.path.getmtime)
    for f in files[: max(0, len(files) - int(cfg["workspace"]["max_cached"]))]:
        os.remove(f)


def _representative(x: np.ndarray, rows: np.ndarray, n: int) -> List[int]:
    if not len(rows):
        return []
    centroid = x[rows].mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    order = np.argsort(-(x[rows] @ centroid), kind="stable")[:n]
    return [int(rows[o]) for o in order]


def build_workspace(query: str, bundle: Any, searcher: Any, cfg: Config, *, bundle_version: str,
                    progress: Progress = None, lookup: Any = None, use_cache: bool = True,
                    llm: Any = None) -> TopicWorkspace:
    """``llm`` defaults to ``scholargrid.llm.get_llm(cfg)``; the app passes ``ui.get_llm()`` so the online
    budget guard and a user's own key apply. Steps below never call get_llm() themselves."""
    from ..llm import LLMError, QuotaExhausted, get_llm

    key = workspace_key(query, bundle_version, cfg)
    if use_cache:
        cached = load_cached(key, cfg)
        if cached is not None:
            return cached
    w = cfg["workspace"]
    llm = llm if llm is not None else get_llm(cfg)
    if lookup is None:
        from ..foundations import WorkLookup

        lookup = WorkLookup(cfg, synthetic=cfg["data"]["source"] == "synthetic")
    timings: Dict[str, float] = {}

    def step(msg: str, frac: float) -> float:
        if progress:
            progress(msg, frac)
        return time.time()

    t = step("Finding papers…", 0.1)
    papers, x, warnings = retrieve(query, bundle, searcher, cfg)
    timings["retrieve"] = round(time.time() - t, 2)
    n_vec = len(x)

    t = step("Grouping into sub-topics…", 0.4)
    top, micro, info = subtopics(x, cfg) if n_vec else (np.zeros(0, int), np.zeros(0, int), {"k": 0, "silhouette": None})
    for i in range(n_vec):
        papers[i].subtopic, papers[i].micro = int(top[i]), int(micro[i])
    timings["subtopics"] = round(time.time() - t, 2)

    t = step("Writing plain-language summaries…", 0.6)
    groups: Dict[int, np.ndarray] = {int(c): np.where(top == c)[0] for c in np.unique(top)} if n_vec else {}
    parent_of: Dict[int, Optional[int]] = {c: None for c in groups}
    for m in (np.unique(micro[micro >= 0]) if n_vec else []):
        rows = np.where(micro == m)[0]
        groups[int(m)] = rows
        parent_of[int(m)] = int(top[rows[0]])
    docs = {sid: " ".join(f"{papers[i].title}. {papers[i].abstract}" for i in rows) for sid, rows in groups.items()}
    kws = keyword_labels(docs, query)
    profiles = (dict(zip(bundle.profiles["arxiv_id"].astype(str), bundle.profiles["summary"].astype(str)))
                if len(getattr(bundle, "profiles", [])) else {})
    problem_counts = (bundle.problems["arxiv_id"].astype(str).value_counts().to_dict()
                      if len(getattr(bundle, "problems", [])) else {})
    newest = newest_date([p.date for p in papers])
    subs: List[Subtopic] = []
    t_activity = 0.0
    for sid, rows in sorted(groups.items()):
        members = [papers[i] for i in rows]
        ids = [p.paper_id for p in members if p.bundle_index is not None]
        t0 = time.time()
        monthly, recent = activity([p.date for p in members], newest)
        t_activity += time.time() - t0
        rep = _representative(x, rows, int(w["representative_per_subtopic"]))
        subs.append(Subtopic(
            id=sid, parent=parent_of[sid], keyword_label=keyword_label(kws.get(sid, [])),
            name=keyword_label(kws.get(sid, [])), keywords=kws.get(sid, []), size=len(rows),
            n_collection=len(ids), recent_share=recent, monthly=monthly,
            top_facets=top_facets(getattr(bundle, "facets", None), ids),
            representative=[papers[i].paper_id for i in rep],
            n_open_problems=int(sum(problem_counts.get(a, 0) for a in ids))))
    labels_pending = False
    llm_used = False
    if w["label_with_llm"] and subs and llm.available():
        by_id = {p.paper_id: p for p in papers}
        items = [(s.id, {"keywords": ", ".join(s.keywords[:8]), "facets": facets_text(s.top_facets),
                         "papers": evidence_lines([by_id[pid] for pid in s.representative], profiles,
                                                  int(w["representative_per_subtopic"]))}) for s in subs]
        got, errors = llm_labels(llm, query, items)
        for s in subs:
            lab = got.get(s.id)
            if lab is not None:
                s.name = lab.name or s.keyword_label
                s.summary = lab.summary
                s.key_concepts = [c.model_dump() for c in lab.key_concepts]
                s.beginner_entry = lab.beginner_entry
        llm_used = bool(got)
        if errors:
            quota = any(isinstance(e, QuotaExhausted) for e in errors)
            labels_pending = labels_pending or quota
            warnings.append(QUOTA_WARNING if quota else LLM_WARNING)
    elif w["label_with_llm"] and subs:
        warnings.append(LLM_WARNING)
    timings["activity"] = round(t_activity, 3)
    timings["labels"] = round(time.time() - t - t_activity, 2)

    t = step("Summarising the topic…", 0.75)
    overview: Dict = {}
    summarised = [s for s in subs if s.parent is None and s.summary]
    if w["overview_with_llm"] and summarised and llm.available():
        from ..llm.prompts import wrap_untrusted
        from .prompts import TopicOverview

        lines = "\n".join(f"- {s.name} ({s.size} papers): {s.summary}" for s in summarised)
        try:
            ov = llm.structured("label", "topic_overview", TopicOverview, query=query,
                                subtopics=wrap_untrusted("evidence", lines))
            overview = ov.value.model_dump()
        except QuotaExhausted:
            labels_pending = True
        except LLMError as exc:
            log.info("Topic overview unavailable (%s).", exc)
    timings["overview"] = round(time.time() - t, 2)

    t = step("Building your reading path…", 0.85)
    idx = np.array([p.bundle_index for p in papers if p.bundle_index is not None], dtype=int)
    path, path_info = workspace_reading_path(bundle, idx, cfg, lookup)
    if cfg["foundations"]["llm_reasons"] and path and llm.available():
        from ..foundations import attach_reasons

        attach_reasons(llm, path, {id(it): query for it in path})
    for it in path:
        it.pop("abstract", None)
    timings["reading_path"] = round(time.time() - t, 2)

    n_live = sum(1 for p in papers if p.is_new)
    ws = TopicWorkspace(
        key=key, query=query, bundle_version=bundle_version, created_at=time.time(), papers=papers,
        subtopics=subs, overview=overview, reading_path=path,
        stats={"timings": timings, "n_collection": int(len(idx)), "n_live": int(n_live),
               "n_live_clustered": int(sum(1 for p in papers[:n_vec] if p.is_new)),
               "k": int(info.get("k") or 0), "silhouette": info.get("silhouette"),
               "reference_coverage": path_info.get("reference_coverage", 0.0),
               "n_citing": path_info.get("n_citing", 0), "labels_pending": labels_pending},
        llm_used=llm_used, warnings=list(dict.fromkeys(warnings)))
    if progress:
        progress("Done", 1.0)
    if not labels_pending:
        save_cached(ws, cfg)
    return ws
