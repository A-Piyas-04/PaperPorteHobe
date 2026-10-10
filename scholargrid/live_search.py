"""Live multi-source search: the collection, plus arXiv, OpenAlex and Semantic
Scholar queried at search time.

One search:

1. expands the query (acronyms, synonyms) and embeds it once;
2. searches the local collection (:func:`scholargrid.search.search`);
3. asks every enabled source for the query phrases **in parallel**. Semantic
   Scholar gets ``search.s2_budget_s`` and is dropped when slow or rate-limited
   (the others continue); nothing waits past ``search.live_deadline_s``; a
   source that answered 429 is skipped for ``search.cooldown_s``;
4. de-duplicates the answers (DOI, arXiv ID, title + year), stores new papers
   in the :class:`~scholargrid.live_store.LiveStore` and embeds up to
   ``search.live_embed_max`` of them;
5. ranks everything into an **exact** section (phrase / expansion matches)
   and a **related** section (similar meaning above the similarity cutoff).

Each source's pages are cached on disk per query for ``search.live_cache_hours``,
so repeating a search or paging back is instant. "Load more" asks for the
next page of every source that still has results.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .acronyms import Expansion, expand
from .acronyms import merge as merge_acronyms
from .live_store import KeyIndex, LiveStore, merge_records
from .search import SearchFilters, cluster_distribution, relative_cutoff, search, sort_key
from .sources import (
    SOURCE_LABELS,
    RateLimited,
    SourcePage,
    SourceUnavailable,
    cooling_down,
    get_adapter,
    start_cooldown,
)
from .utils import ensure_dir, get_logger, load_json, save_json

log = get_logger("live_search")

S2 = "semantic_scholar"
_MAX_API_PHRASES = 4
_MIN_REQUEST_S = 1.0          # do not start another page with less time than this left
_FALLBACK_CUTOFF = 0.35       # related cutoff when the collection is too small to estimate one
ProgressFn = Optional[Callable[[str], None]]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
@dataclass
class LiveOutcome:
    records: List[Dict] = field(default_factory=list)       # de-duplicated
    totals: Dict[str, Optional[int]] = field(default_factory=dict)
    status: Dict[str, str] = field(default_factory=dict)    # ok|cached|slow|rate_limited|cooldown|error|done
    cursors: Dict[str, Any] = field(default_factory=dict)   # next page per source (None = no more)
    elapsed: float = 0.0

    @property
    def has_more(self) -> bool:
        return any(c is not None for c in self.cursors.values())


def _cache_path(cache_dir: str, phrases: Sequence[str], source: str, page: int, per_source: int) -> str:
    key = json.dumps([list(phrases), source, page, per_source])
    return os.path.join(cache_dir, hashlib.sha1(key.encode()).hexdigest()[:20] + ".json")


def _fetch_source(name: str, phrases: List[str], cursor: Any, want: int, deadline: float,
                  cfg) -> SourcePage:
    """Page through one source until ``want`` records, no more results, or the deadline."""
    adapter = get_adapter(name)
    out = SourcePage(name, next_cursor=cursor)
    first = True
    while len(out.records) < want and (first or out.next_cursor is not None):
        left = deadline - time.monotonic()
        if left < _MIN_REQUEST_S and not first:
            break
        page = adapter.fetch(phrases, out.next_cursor, want - len(out.records), max(left, 0.5),
                             cfg, deadline=deadline)
        out.records.extend(page.records)
        out.total = page.total if page.total is not None else out.total
        out.next_cursor = page.next_cursor
        first = False
        if not page.records:
            break
    return out


def fetch_live(exp: Expansion, cfg, *, page: int = 0, cursors: Optional[Dict[str, Any]] = None,
               cache_dir: Optional[str] = None, now: Callable[[], float] = time.time) -> LiveOutcome:
    """Fetch one page from every enabled source, in parallel, within the time budgets.

    ``cursors`` are the previous page's ``LiveOutcome.cursors``; sources whose
    cursor is ``None`` are finished and skipped.
    """
    scfg = cfg["search"]
    per_source = int(scfg["live_per_source"])
    api_phrases = exp.api_phrases[:_MAX_API_PHRASES]
    cache_dir = cache_dir or os.path.join(cfg.live_dir, "query_cache")
    ttl = float(scfg["live_cache_hours"]) * 3600
    t0 = time.monotonic()
    out = LiveOutcome()

    todo: Dict[str, Any] = {}
    for name in scfg["live_sources"]:
        cursor = None if cursors is None else cursors.get(name)
        if page > 0 and cursor is None:
            out.status[name], out.cursors[name] = "done", None
            continue
        path = _cache_path(cache_dir, api_phrases, name, page, per_source)
        if ttl > 0 and os.path.exists(path):
            try:
                cached = load_json(path)
                if now() - float(cached.get("at", 0)) < ttl:
                    out.records.extend(cached["records"])
                    out.totals[name] = cached.get("total")
                    out.cursors[name] = cached.get("next")
                    out.status[name] = "cached"
                    continue
            except Exception:  # corrupt cache entry: refetch
                pass
        if cooling_down(name):
            out.status[name], out.cursors[name] = "cooldown", cursor
            continue
        todo[name] = cursor

    if todo:
        deadline_all = t0 + float(scfg["live_deadline_s"])
        deadlines = {n: min(deadline_all, t0 + float(scfg["s2_budget_s"])) if n == S2 else deadline_all
                     for n in todo}
        pool = cf.ThreadPoolExecutor(max_workers=len(todo), thread_name_prefix="live")
        futures = {pool.submit(_fetch_source, n, [api_phrases[0]] if n == S2 else api_phrases,
                               c, per_source, deadlines[n], cfg): n for n, c in todo.items()}
        pending = set(futures)
        while pending:
            nearest = min(deadlines[futures[f]] for f in pending)
            done, pending = cf.wait(pending, timeout=max(0.0, nearest - time.monotonic()),
                                    return_when=cf.FIRST_COMPLETED)
            for f in done:
                name = futures[f]
                try:
                    res = f.result()
                except RateLimited:
                    start_cooldown(name, float(scfg["cooldown_s"]))
                    out.status[name], out.cursors[name] = "rate_limited", todo[name]
                    continue
                except SourceUnavailable:
                    out.status[name], out.cursors[name] = "slow", todo[name]
                    continue
                except Exception as exc:
                    log.warning("Live source %s failed: %s", name, exc)
                    out.status[name], out.cursors[name] = "error", todo[name]
                    continue
                out.records.extend(res.records)
                out.totals[name] = res.total
                out.cursors[name] = res.next_cursor
                out.status[name] = "ok"
                if ttl > 0:
                    save_json({"at": now(), "records": res.records, "total": res.total,
                               "next": res.next_cursor},
                              _cache_path(ensure_dir(cache_dir), api_phrases, name, page, per_source))
            for f in [f for f in pending if time.monotonic() >= deadlines[futures[f]]]:
                name = futures[f]
                out.status[name], out.cursors[name] = "slow", todo[name]
                pending.discard(f)
        pool.shutdown(wait=False, cancel_futures=True)

    out.records = merge_records(out.records)
    out.elapsed = round(time.monotonic() - t0, 2)
    log.info("Live lookup page %d for %r: %d papers in %.1fs %s", page, api_phrases[0],
             len(out.records), out.elapsed, out.status)
    return out


# ---------------------------------------------------------------------------
# Searching the collection + the live store together
# ---------------------------------------------------------------------------
def _store_result(store: LiveStore, j: int, match: str, level: Optional[int], sim: Optional[float],
                  phrases: Sequence[str], is_new: bool) -> Dict:
    from .phrase import matched_words

    r = store.row_record(j)
    if not pd.isna(r["date"]):
        date_str = r["date"].strftime("%Y-%m-%d")
    else:
        date_str = "" if pd.isna(r["year"]) else str(int(r["year"]))
    return {
        "rank": 0, "paper_id": r["paper_id"], "arxiv_id": r["arxiv_id"], "doi": r["doi"],
        "url": r["url"] or (f"https://arxiv.org/abs/{r['arxiv_id']}" if r["arxiv_id"] else ""),
        "title": r["title"], "abstract": r["abstract"], "authors": r["authors"], "venue": r["venue"],
        "primary_category": r["primary_category"], "date": date_str,
        "cluster_id": None, "cluster_label": "", "score": round(float(sim or 0.0), 4),
        "sim": None if sim is None else round(float(sim), 4),
        "cited_by_count": int(r["cited_by_count"]), "citation_velocity": 0.0,
        "matched_terms": matched_words(phrases, r["title"], r["abstract"]),
        "match": match, "match_level": level, "arxiv_url": "", "sources": r["sources"],
        "is_new": is_new, "bundle_index": None,
    }


class LiveSearcher:
    """Searches the bundle and the live store, optionally fetching from live sources."""

    def __init__(self, cfg, bundle, store: Optional[LiveStore] = None):
        self.cfg = cfg
        self.bundle = bundle
        fp = bundle.embedder.fingerprint if bundle.embedder is not None else None
        dim = int(bundle.embeddings.shape[1]) if len(bundle.embeddings) else None
        self.store = store if store is not None else LiveStore.load(cfg.live_dir, fp, dim)
        self.bundle_keys = KeyIndex.from_frame(bundle.df)

    @property
    def live_enabled(self) -> bool:
        return bool(self.cfg["search"]["live"]) and bool(self.cfg["search"]["live_sources"])

    def expansion(self, query: str) -> Expansion:
        acr = merge_acronyms(self.bundle.acronyms, self.store.acronyms)
        return expand(query, acr, self.cfg["search"].get("synonyms"))

    def _qvec(self, exp: Expansion) -> Optional[np.ndarray]:
        if self.bundle.embedder is None or not exp.query:
            return None
        try:
            return self.bundle.embedder.encode_query(exp.embed_text)
        except Exception as exc:
            log.warning("Query embedding failed (%s); keyword matching only.", exc)
            return None

    def run(self, query: str, *, filters: Optional[SearchFilters] = None, sort: str = "relevance",
            citation_weight: float = 0.0, pages: int = 1, live: bool = True,
            progress: ProgressFn = None) -> Dict:
        scfg = self.cfg["search"]
        b = self.bundle
        pages = max(1, int(pages))
        exp = self.expansion(query)
        qvec = self._qvec(exp)
        cap_exact = int(scfg["top_k_exact"]) * pages
        cap_related = int(scfg["top_k_related"]) * pages

        if progress:
            progress("Searching saved papers…")
        local = search(query, b.embedder, b.embeddings, b.df, b.labels, b.clusters_meta,
                       bm25=b.bm25, index=b.index, cfg=self.cfg, filters=filters,
                       citation_weight=citation_weight, phrase_index=b.phrase, expansion=exp,
                       qvec=qvec, top_k_exact=cap_exact, top_k_related=cap_related)

        outcome = LiveOutcome()
        fetched_rows: List[int] = []
        in_collection = 0
        if live and self.live_enabled and exp.query:
            if progress:
                progress("Checking " + ", ".join(SOURCE_LABELS[s] for s in scfg["live_sources"])
                         + " for more papers…")
            cursors = None
            for p in range(pages):
                o = fetch_live(exp, self.cfg, page=p, cursors=cursors)
                outcome.records.extend(o.records)
                if p == 0:
                    outcome.totals = o.totals
                outcome.status.update({k: v for k, v in o.status.items() if v != "done" or p == 0})
                outcome.cursors = o.cursors
                outcome.elapsed += o.elapsed
                cursors = o.cursors
                if not o.has_more:
                    break
            fresh = []
            for rec in merge_records(outcome.records):
                if self.bundle_keys.find(rec) is not None:
                    in_collection += 1
                else:
                    fresh.append(rec)
            if fresh:
                fetched_rows = self.store.upsert(fresh)
            if progress and fetched_rows:
                progress(f"Reading {len(set(fetched_rows))} papers found online…")
            embedded = self.store.embed_missing(b.embedder, fetched_rows, int(scfg["live_embed_max"]),
                                                float(scfg["live_embed_budget_s"]))
            if embedded or fresh:
                self.store.save()

        pool_exact, pool_related = self._search_store(exp, qvec, local.get("cutoff"), filters,
                                                      cap_exact, cap_related)
        new_rows = set(fetched_rows)
        exact = [r for r in local["results"] if r["match"] == "exact"]
        related = [r for r in local["results"] if r["match"] == "related"]
        exact += [_store_result(self.store, j, "exact", lvl, s, exp.phrases, j in new_rows)
                  for j, lvl, s in pool_exact]
        related += [_store_result(self.store, j, "related", None, s, exp.phrases, j in new_rows)
                    for j, s in pool_related]
        exact.sort(key=lambda r: (r["match_level"] or 9, -(r["sim"] if r["sim"] is not None else r["score"])))
        related.sort(key=lambda r: -(r["sim"] if r["sim"] is not None else r["score"]))
        best = max((r["sim"] for r in exact + related if r["sim"] is not None), default=None)
        floor = (relative_cutoff(best, scfg, local.get("sim_base") or 0.0)
                 if len(exact) + len(related) >= 2 else None)
        if floor is not None:
            related = [r for r in related if r["sim"] is None or r["sim"] >= floor]
        results = exact[:cap_exact] + related[:cap_related]
        key = sort_key(sort, results) if sort != "relevance" else None
        if key is not None:
            results = [results[i] for i in np.argsort(-key, kind="stable")]
        for rank, r in enumerate(results, start=1):
            r["rank"] = rank

        n_exact_total = int(local["n_exact_total"]) + len(pool_exact)
        totals = {k: v for k, v in outcome.totals.items() if v is not None}
        about = max([n_exact_total, *totals.values()]) if totals else n_exact_total
        return {
            **{k: local[k] for k in ("query", "mode", "degraded", "expansion", "phrases", "cutoff")},
            "results": results,
            "result_indices": [r["bundle_index"] for r in results if r["bundle_index"] is not None],
            "cluster_distribution": cluster_distribution(results, b.clusters_meta),
            "n_exact": sum(1 for r in results if r["match"] == "exact"),
            "n_related": sum(1 for r in results if r["match"] == "related"),
            "n_exact_total": n_exact_total,
            "about_total": int(about),
            "source_totals": totals,
            "source_status": dict(outcome.status),
            "has_more": (outcome.has_more or n_exact_total > cap_exact
                         or len(exact) > cap_exact or len(related) > cap_related),
            "live": bool(live and self.live_enabled),
            "live_seconds": round(outcome.elapsed, 1),
            "n_found_online": len(set(fetched_rows)) + in_collection,
            "n_store": len(self.store),
            "pending_embeddings": int((~self.store.has_vec).sum()) if len(self.store) else 0,
        }

    def _search_store(self, exp: Expansion, qvec: Optional[np.ndarray], cutoff: Optional[float],
                      filters: Optional[SearchFilters], cap_exact: int, cap_related: int):
        store = self.store
        with store.lock:
            if not len(store) or not exp.query:
                return [], []
            df = store.df
            mask = filters.mask(df) if filters is not None and filters.active() else None
            levels = store.phrase.match(exp.phrases, mask, int(self.cfg["search"]["proximity_window"]))
            has_vec = store.has_vec
            sims = None
            if qvec is not None and has_vec.any() and store.vectors.shape[1] == len(qvec):
                sims = np.where(has_vec, store.vectors @ qvec, -np.inf).astype(np.float32)
        exact = sorted(((j, lvl, None if sims is None or not np.isfinite(sims[j]) else float(sims[j]))
                        for j, lvl in levels.items()),
                       key=lambda t: (t[1], -(t[2] or 0.0)))[:cap_exact * 2]
        related: List[tuple] = []
        if sims is not None:
            cut = cutoff if cutoff is not None else _FALLBACK_CUTOFF
            ok = np.isfinite(sims) & (sims >= cut)
            if mask is not None:
                ok &= mask
            cand = [int(j) for j in np.flatnonzero(ok) if int(j) not in levels]
            cand.sort(key=lambda j: -sims[j])
            related = [(j, float(sims[j])) for j in cand[:cap_related * 2]]
        return exact, related

    def similar(self, paper_id: str, top_k: int = 25) -> Dict:
        """Papers like ``paper_id`` (from the collection or the live store), by meaning."""
        b, store = self.bundle, self.store
        vec, self_bundle, self_store, title = None, None, None, paper_id
        hits = np.flatnonzero(b.df["arxiv_id"].astype(str).to_numpy() == paper_id)
        if len(hits):
            self_bundle = int(hits[0])
            vec = np.asarray(b.embeddings[self_bundle], dtype=np.float32)
            title = str(b.df.iloc[self_bundle]["title"])
        else:
            j = store.find_row(paper_id)
            if j is not None:
                self_store, title = j, str(store.df.iloc[j]["title"])
                if not store.df.at[j, "has_vec"]:
                    store.embed_missing(b.embedder, [j], 1)
                if store.df.at[j, "has_vec"]:
                    vec = store.vectors[j]
        out = {"query": f"papers like “{title}”", "mode": "similar", "degraded": False,
               "results": [], "result_indices": [], "cluster_distribution": [], "n_exact": 0,
               "n_related": 0, "n_exact_total": 0, "about_total": 0, "source_totals": {},
               "source_status": {}, "has_more": False, "live": False, "expansion": []}
        if vec is None:
            return out
        rows: List[Dict] = []
        if len(b.embeddings) and len(vec) == b.embeddings.shape[1]:
            sims = np.asarray(b.embeddings @ vec, dtype=np.float32)
            if self_bundle is not None:
                sims[self_bundle] = -np.inf
            for i in np.argsort(-sims)[:top_k]:
                rows.append(self._bundle_result(int(i), float(sims[i])))
        with store.lock:
            if len(store) and store.has_vec.any() and store.vectors.shape[1] == len(vec):
                ssims = np.where(store.has_vec, store.vectors @ vec, -np.inf)
                if self_store is not None:
                    ssims[self_store] = -np.inf
                for j in np.argsort(-ssims)[:top_k]:
                    if np.isfinite(ssims[j]):
                        rows.append(_store_result(store, int(j), "related", None, float(ssims[j]), [], False))
        rows.sort(key=lambda r: -r["sim"])
        rows = rows[:top_k]
        for rank, r in enumerate(rows, start=1):
            r["rank"] = rank
        out.update(results=rows, n_related=len(rows),
                   result_indices=[r["bundle_index"] for r in rows if r["bundle_index"] is not None],
                   cluster_distribution=cluster_distribution(rows, b.clusters_meta))
        return out

    def _bundle_result(self, i: int, sim: float) -> Dict:
        b = self.bundle
        r = b.df.iloc[i]
        aid = str(r["arxiv_id"])
        cid = int(b.labels[i])
        return {
            "rank": 0, "paper_id": aid, "arxiv_id": aid, "doi": str(r.get("doi", "") or ""),
            "url": f"https://arxiv.org/abs/{aid}", "title": r["title"],
            "abstract": str(r.get("abstract", "") or ""), "authors": str(r.get("authors", "") or ""),
            "venue": str(r.get("venue", "") or ""), "primary_category": r["primary_category"],
            "date": r["date"].strftime("%Y-%m-%d"), "cluster_id": cid,
            "cluster_label": b.clusters_meta.get(cid, {}).get("label", ""),
            "score": round(sim, 4), "sim": round(sim, 4),
            "cited_by_count": int(r.get("cited_by_count", 0) or 0), "citation_velocity": 0.0,
            "matched_terms": [], "match": "related", "match_level": None, "arxiv_url": "",
            "sources": ["collection"], "is_new": False, "bundle_index": i,
        }

    def paper(self, paper_id: str) -> Optional[Dict]:
        """A result-shaped record for a paper in the collection or the live store."""
        b = self.bundle
        hits = np.flatnonzero(b.df["arxiv_id"].astype(str).to_numpy() == paper_id)
        if len(hits):
            return self._bundle_result(int(hits[0]), 0.0)
        j = self.store.find_row(paper_id)
        return None if j is None else _store_result(self.store, j, "related", None, None, [], False)
