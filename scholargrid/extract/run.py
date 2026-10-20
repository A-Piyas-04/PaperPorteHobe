"""run_extract(df, cfg) -> stats. Incremental, quota-aware, resumable across days; never drops a paper.

- mode=auto -> LLM if ``get_llm(cfg).available()``, else heuristic.
- To do: in-scope papers whose stored row is not current (input hash or prompt version changed, or a
  heuristic row while an LLM is available - that is how yesterday's heuristic fill-ins get upgraded).
  Papers whose ``llm_attempts`` reached ``max_llm_attempts`` are skipped.
- Papers are processed in ``select_scope`` priority order, so limited quota goes to the most useful first.
- After a quota stop the rest follow ``extract.on_quota``: heuristic (default; upgraded later), skip, stop.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from ..config import Config
from ..fulltext import FulltextStore
from ..llm import BudgetExceeded, LLMError, QuotaExhausted, get_llm
from ..llm.prompts import PROMPTS, wrap_untrusted
from ..scope import select_scope
from ..utils import get_logger
from . import prompts as _prompts  # noqa: F401  (registers "extract_paper")
from .document import PaperDocument, build_document
from .heuristic import heuristic_extract_from_text
from .schemas import PaperExtraction
from .store import ExtractionStore
from .verify import verify_quotes

log = get_logger("extract")
PROMPT = "extract_paper"
CHUNK = 40          # papers per flush (a crash or Ctrl-C keeps finished work)
_EMPTY_SECTIONS = pd.DataFrame(columns=["idx", "kind", "heading", "text", "anchor"])


def documents(scope: pd.DataFrame, cfg: Config) -> List[PaperDocument]:
    e = cfg["extract"]
    store = FulltextStore(cfg)
    ids = scope["arxiv_id"].astype(str).tolist()
    secs, cues = store.sections(ids), store.cues(ids)
    by_sec = {str(k): g for k, g in secs.groupby("arxiv_id")} if len(secs) else {}
    by_cue = {str(k): g for k, g in cues.groupby("arxiv_id")} if len(cues) else {}
    docs = []
    for r in scope.itertuples():                      # keeps select_scope priority order
        aid = str(r.arxiv_id)
        s = by_sec.get(aid, _EMPTY_SECTIONS)
        if s.empty and not e["abstract_only_fallback"]:
            continue
        docs.append(build_document(aid, str(r.title), str(r.abstract), s, by_cue.get(aid, pd.DataFrame()),
                                   e["section_budget"], int(e["max_input_chars"]), int(e["max_cue_sentences"])))
    return docs


def _record(store: ExtractionStore, doc: PaperDocument, ext: PaperExtraction, mode: str, model: str,
            cfg: Config, attempts: int) -> Tuple[int, int]:
    e = cfg["extract"]
    verified, dropped = verify_quotes(ext, doc, int(e["quote_min_ratio"]), int(e["max_problems_per_paper"]))
    problems = [{"statement": v.problem.statement, "quote": v.quote, "section": v.section_label,
                 "anchor": doc.anchors.get(v.section_label, ""), "section_kind": doc.kinds.get(v.section_label, ""),
                 "problem_type": v.problem.problem_type, "specificity": v.problem.specificity,
                 "verified_ratio": v.ratio} for v in verified]
    payload = ext.model_dump(mode="json")
    payload["findings"] = payload["findings"][: int(e["max_findings_per_paper"])]
    suffix = "" if doc.has_fulltext else "_abstract"
    store.add(arxiv_id=doc.arxiv_id, input_hash=doc.input_hash, prompt_version=PROMPTS[PROMPT].version,
              mode=mode + suffix, model=model, payload=payload, problems=problems,
              n_problems_raw=len(ext.open_problems), n_problems_verified=len(problems), cost_usd=0.0,
              llm_attempts=attempts)
    return len(problems), dropped


def run_extract(df: pd.DataFrame, cfg: Config, llm: Any = None) -> Dict:
    e = cfg["extract"]
    stats: Dict[str, Any] = {"enabled": bool(e["enabled"]), "in_scope": 0, "done_now": 0, "llm": 0,
                             "heuristic": 0, "problems_verified": 0, "problems_dropped": 0,
                             "quota_stopped": False, "pending_llm": 0, "requests_used": 0,
                             "quote_verified_rate": None}
    if not e["enabled"]:
        return stats
    scope = select_scope(df, cfg)
    stats["in_scope"] = int(len(scope))
    llm = llm if llm is not None else get_llm(cfg)
    use_llm = e["mode"] == "llm" or (e["mode"] == "auto" and llm.available())
    store = ExtractionStore(cfg)
    version = PROMPTS[PROMPT].version
    docs = [d for d in documents(scope, cfg)
            if not store.is_current(d.arxiv_id, d.input_hash, version, use_llm)
            and not (use_llm and store.attempts(d.arxiv_id) >= int(e["max_llm_attempts"]))]
    log.info("Extraction: %d in scope, %d to process (%s).", len(scope), len(docs), "llm" if use_llm else "heuristic")
    ledger = getattr(llm, "ledger", None)
    req_before = getattr(ledger, "requests", 0)

    def heuristic(doc: PaperDocument, attempts: int) -> None:
        v, d = _record(store, doc, heuristic_extract_from_text(doc.text), "heuristic", "heuristic", cfg, attempts)
        stats["heuristic"] += 1
        stats["problems_verified"] += v
        stats["problems_dropped"] += d

    skipped = 0

    def after_quota(doc: PaperDocument) -> None:
        nonlocal skipped
        stats["quota_stopped"] = True
        stats["pending_llm"] += 1
        if e["on_quota"] == "heuristic":
            heuristic(doc, store.attempts(doc.arxiv_id))      # upgraded by a later run
        else:
            skipped += 1

    try:
        if not use_llm:
            for doc in docs:
                heuristic(doc, store.attempts(doc.arxiv_id))
        else:
            by_id = {d.arxiv_id: d for d in docs}
            items = [(d.arxiv_id, {"arxiv_id": d.arxiv_id, "document": wrap_untrusted("paper", d.text)})
                     for d in docs]
            for start in range(0, len(items), CHUNK):
                chunk = items[start:start + CHUNK]
                if stats["quota_stopped"]:                         # quota gone: no more LLM calls this run
                    for iid, _ in chunk:
                        after_quota(by_id[iid])
                    continue
                try:
                    results = llm.structured_many("extract", PROMPT, PaperExtraction, chunk)
                except BudgetExceeded as exc:
                    results = {iid: QuotaExhausted(str(exc)) for iid, _ in chunk}
                for iid, _ in chunk:
                    doc, res = by_id[iid], results.get(iid)
                    if isinstance(res, (QuotaExhausted, BudgetExceeded)):
                        if e["on_quota"] == "stop":
                            raise res
                        after_quota(doc)
                    elif isinstance(res, LLMError) or res is None:
                        heuristic(doc, store.attempts(iid) + 1)
                    else:
                        v, d = _record(store, doc, res.value, "llm", res.route, cfg, 0)
                        stats["llm"] += 1
                        stats["problems_verified"] += v
                        stats["problems_dropped"] += d
                store.flush()
    finally:
        store.flush()
    stats["done_now"] = len(docs) - skipped
    stats["requests_used"] = int(getattr(ledger, "requests", 0) - req_before)
    total = stats["problems_verified"] + stats["problems_dropped"]
    stats["quote_verified_rate"] = round(stats["problems_verified"] / total, 4) if total else None
    if stats["quota_stopped"]:
        log.warning("Free quota used up: %d papers wait for an LLM pass; rerun after the quota resets "
                    "(`python -m scholargrid.llm quota`).", stats["pending_llm"])
    return stats
