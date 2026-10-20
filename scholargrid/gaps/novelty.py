"""Novelty check: does recent work (collection, and live sources online) already address the gap?

"novel" means "no match among what was checked"; the card always says what was checked and when.
Without an LLM the verdict is heuristic and never "addresses".
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger, l2_normalize
from . import prompts as _prompts  # noqa: F401  (registers judge_addressed)
from .model import GapCard, NoveltyCheck, NoveltyHit

log = get_logger("gaps.novelty")
_WORD = re.compile(r"[a-z0-9]{3,}")
HEURISTIC_PARTIAL_SIM = 0.85


def _overlap(a: str, b: str) -> float:
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return len(wa & wb) / max(1, len(wa | wb))


def _collection_candidates(card: GapCard, ctx: Any, vec: np.ndarray, cfg: Config) -> List[Dict]:
    g = cfg["gaps"]
    allp = ctx.all_papers
    if ctx.all_vectors is None or not len(allp):
        return []
    dates = pd.to_datetime(allp["date"])
    recent = (dates >= ctx.ref_date - pd.DateOffset(months=int(g["novelty_recent_months"]))).to_numpy()
    exclude = {e.paper_id for e in card.evidence}
    mask = recent & ~allp["arxiv_id"].astype(str).isin(exclude).to_numpy()
    if not mask.any():
        return []
    rows = np.where(mask)[0]
    sims = np.asarray(ctx.all_vectors[rows], dtype=np.float32) @ vec
    top = rows[np.argsort(-sims, kind="stable")[: int(g["novelty_collection_top"])]]
    sim_of = dict(zip(rows.tolist(), sims.tolist()))
    out = []
    for i in top:
        r = allp.iloc[int(i)]
        aid = str(r["arxiv_id"])
        out.append({"paper_id": aid, "title": str(r["title"]), "abstract": str(r.get("abstract", "") or ""),
                    "url": f"https://arxiv.org/abs/{aid}", "date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                    "score": float(sim_of[int(i)])})
    return out


def _live_candidates(card: GapCard, ctx: Any, vec: np.ndarray, cfg: Config, searcher: Any,
                     queries: List[str]) -> List[Dict]:
    g = cfg["gaps"]
    cut = ctx.ref_date - pd.DateOffset(months=int(g["novelty_recent_months"]))
    exclude = {e.paper_id for e in card.evidence}
    out: Dict[str, Dict] = {}
    store = getattr(searcher, "store", None)
    for q in queries:
        try:
            res = searcher.run(q, pages=1, live=True)
        except Exception as exc:
            log.warning("Live novelty search failed for %r (%s).", q, exc)
            continue
        for r in res.get("results", []):
            pid = str(r.get("paper_id") or r.get("arxiv_id") or "")
            if not pid or pid in exclude or pid in out:
                continue
            try:
                d = pd.Timestamp(str(r.get("date") or "")[:10])
            except (ValueError, TypeError):
                continue
            if pd.isna(d) or d < cut:
                continue
            score: Optional[float] = None
            bi = r.get("bundle_index")
            if bi is not None and ctx.all_vectors is not None:
                score = float(np.asarray(ctx.all_vectors[int(bi)], dtype=np.float32) @ vec)
            elif store is not None and len(store):
                j = store.find_row(pid)
                if j is not None and bool(store.has_vec[j]) and store.vectors.shape[1] == len(vec):
                    score = float(np.asarray(store.vectors[j], dtype=np.float32) @ vec)
            if score is None:
                score = _overlap(card.statement, f"{r.get('title', '')} {r.get('abstract', '')}")
            out[pid] = {"paper_id": pid, "title": str(r.get("title") or ""), "abstract": str(r.get("abstract") or ""),
                        "url": str(r.get("url") or ""), "date": f"{d:%Y-%m-%d}", "score": score}
    return list(out.values())


def _judge(card: GapCard, cands: List[Dict], llm: Any) -> Tuple[List[NoveltyHit], str]:
    from ..extract.verify import locate
    from ..llm import LLMError
    from .prompts import AddressedJudgement

    def heuristic(c: Dict) -> NoveltyHit:
        verdict: Literal["partially", "unrelated"] = "partially" if c["score"] >= HEURISTIC_PARTIAL_SIM else "unrelated"
        return NoveltyHit(paper_id=c["paper_id"], title=c["title"], url=c["url"], date=c["date"], verdict=verdict,
                          reason=f"similarity {c['score']:.2f} (no AI judgment)", score=round(c["score"], 3))

    if llm is None or not llm.available() or not cands:
        return [heuristic(c) for c in cands], "heuristic"
    items = [(f"{card.id}|{c['paper_id']}", {"statement": card.statement, "title": c["title"],
                                             "abstract": c["abstract"][:2500]}) for c in cands]
    try:
        results = llm.structured_many("judge", "judge_addressed", AddressedJudgement, items)
    except LLMError:
        return [heuristic(c) for c in cands], "heuristic"
    hits: List[NoveltyHit] = []
    any_llm = False
    for (iid, _), c in zip(items, cands):
        res = results.get(iid)
        if isinstance(res, LLMError) or res is None:
            hits.append(heuristic(c))
            continue
        any_llm = True
        quote = res.value.quote
        if quote:
            found = locate(quote, c["abstract"], 90)
            quote = found[0] if found else ""
        hits.append(NoveltyHit(paper_id=c["paper_id"], title=c["title"], url=c["url"], date=c["date"],
                               verdict=res.value.verdict, reason=res.value.reason, quote=quote,
                               score=round(c["score"], 3)))
    return hits, "llm" if any_llm else "heuristic"


def novelty_check(card: GapCard, ctx: Any, embedder: Any, llm: Any, cfg: Config, searcher: Any = None,
                  live: bool = False) -> NoveltyCheck:
    g = cfg["gaps"]
    if embedder is None:
        return NoveltyCheck(status="not_checked", scope="none", method="none")
    vec = l2_normalize(np.asarray(embedder.transform([card.statement]), dtype=np.float32))[0]
    cands = _collection_candidates(card, ctx, vec, cfg)
    queries = list(dict.fromkeys(q for q in (card.novelty_queries or [" ".join(card.key_terms.values())]) if q))
    queries = queries[: int(g["novelty_live_queries"])]
    use_live = bool(live and searcher is not None and queries)
    if use_live:
        known = {c["paper_id"] for c in cands}
        cands += [c for c in _live_candidates(card, ctx, vec, cfg, searcher, queries) if c["paper_id"] not in known]
    cands.sort(key=lambda c: -c["score"])
    judged = cands[: int(g["novelty_top_judge"])]
    hits, method = _judge(card, judged, llm)
    verdicts = {h.verdict for h in hits}
    status = ("already_addressed" if "addresses" in verdicts else
              "partially_explored" if "partially" in verdicts else "novel")
    return NoveltyCheck(status=status, scope="collection+live" if use_live else "collection",  # type: ignore[arg-type]
                        checked_at=time.time(), queries=queries if use_live else [], n_candidates=len(cands),
                        hits=hits, method=method)  # type: ignore[arg-type]
