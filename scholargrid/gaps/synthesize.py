"""Gap Card text: LLM drafts grounded in the candidate's evidence, verified, with an honest template fallback.

Per item: LLMError -> template; draft.reject -> dropped (counted); verify_text failure -> one synchronous
retry naming the foreign entities; second failure -> template. Quota stops turn the rest into templates
(``synthesized=False``); LLM results are cached, so a later run re-synthesizes only what is missing.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from ..config import Config
from ..llm.prompts import PROMPTS
from ..utils import get_logger
from . import prompts as _prompts  # noqa: F401  (registers the gap prompts)
from .model import TYPE_HELP, TYPE_LABELS, GapCandidate
from .prompts import GapDraft
from .verify import verify_text

log = get_logger("gaps.synthesize")

_CAVEATS = {
    "stated_open_problem": "Later papers may have solved it without saying so in these terms; check the novelty "
                           "result and recent work.",
    "long_standing_problem": "It may have stayed open because it is very hard or poorly defined, not because "
                             "nobody tried.",
    "unexplored_combination": "The combination may be infeasible or uninteresting for reasons the statistics "
                              "can't see (e.g. incompatible data or assumptions).",
    "evaluation_gap": "Evaluations may exist in papers or appendices outside what was analysed here.",
    "unsettled_question": "The results may differ because of setup details (splits, model sizes, metrics) "
                          "rather than a real disagreement.",
    "missing_bridge": "The two lines of work may cite each other through venues or papers outside this collection.",
}


def evidence_lines(c: GapCandidate) -> str:
    out = []
    for e in c.evidence:
        line = f"{e.eid} [{e.role}] ({e.year or 'n.d.'}) \"{e.title}\""
        if e.section:
            line += f" | section: {e.section}"
        if e.quote:
            line += f" | quote: \"{e.quote}\""
        elif e.note:
            line += f" | note: {e.note}"
        out.append(line)
    return "\n".join(out)


def _vars(c: GapCandidate, extra: str = "") -> Dict[str, str]:
    return {"type_label": TYPE_LABELS.get(c.type, c.type), "type_help": TYPE_HELP.get(c.type, ""),
            "scope_label": str(c.scope.get("label", "")),
            "key_terms": "; ".join(f"{k}: {v}" for k, v in c.key_terms.items()),
            "stats": json.dumps(c.stats, default=str)[:1500], "evidence": evidence_lines(c),
            "seed": c.seed_statement + extra}


def template_card(c: GapCandidate) -> GapDraft:
    """No-LLM fallback: honest, generic wording per type."""
    terms = [v for k, v in c.key_terms.items() if k != "trigger"]
    title = (TYPE_LABELS.get(c.type, "Research gap") + ": " + " × ".join(terms))[:90]
    first_id = c.evidence[0].eid if c.evidence else "E1"
    return GapDraft(
        reject=False, reject_reason="", title=title, statement=c.seed_statement,
        why_it_matters=TYPE_HELP.get(c.type, ""),
        why_it_might_not_be_a_gap=["Work on this may exist outside the papers analysed here (other venues, "
                                   "industry, or different terminology).", _CAVEATS.get(c.type, "")],
        first_steps=[f"Read the evidence papers, starting with {first_id}.",
                     "Search for recent work using the key terms above.",
                     "Write down one concrete, testable question and how you would measure it."],
        skills=[], data_needs="", compute="unknown", feasibility="medium", beginner_friendly=False,
        evidence_used=[e.eid for e in c.evidence], novelty_queries=[" ".join(terms)[:120]] if terms else [])


def synthesize(cands: List[GapCandidate], llm: Any, cfg: Config, offline: bool
               ) -> Tuple[List[Tuple[GapCandidate, GapDraft, Dict]], Dict[str, Any]]:
    """Returns ([(candidate, draft, generated_by)], stats)."""
    from ..llm import BudgetExceeded, LLMError, QuotaExhausted

    stats: Dict[str, Any] = {"rejected_by_llm": 0, "text_fallbacks": 0, "quota_stopped": False, "synthesized": 0}
    version = PROMPTS["synthesize_gap"].version
    out: List[Tuple[GapCandidate, GapDraft, Dict]] = []

    def template(c: GapCandidate, why: str) -> None:
        stats["text_fallbacks"] += 1
        out.append((c, template_card(c), {"model": "template", "prompt_version": version, "synthesized": False,
                                          "reason": why}))

    use_llm = bool(cfg["gaps"]["synthesize_with_llm"]) and llm is not None and llm.available()
    if not use_llm:
        for c in cands:
            template(c, "llm unavailable")
        return out, stats
    results = llm.structured_many("synthesize_gap", "synthesize_gap", GapDraft,
                                  [(c.id, _vars(c)) for c in cands])
    for c in cands:
        res = results.get(c.id)
        if isinstance(res, (QuotaExhausted, BudgetExceeded)):
            stats["quota_stopped"] = True
            template(c, "quota")
            continue
        if isinstance(res, LLMError) or res is None:
            template(c, f"llm error: {res}"[:120])
            continue
        draft, route = res.value, res.route
        if draft.reject:
            stats["rejected_by_llm"] += 1
            log.info("Synthesizer rejected %s: %s", c.id, draft.reject_reason)
            continue
        issues = verify_text(draft, c)
        if issues:
            foreign = next((i.split(": ", 1)[1] for i in issues if i.startswith("entities not in evidence")), "")
            extra = (f"\nPrevious draft named entities not in evidence: {foreign}. Remove them." if foreign
                     else f"\nPrevious draft had problems: {'; '.join(issues)}. Fix them.")
            try:
                again = llm.structured("synthesize_gap", "synthesize_gap", GapDraft, **_vars(c, extra))
            except (QuotaExhausted, BudgetExceeded):
                stats["quota_stopped"] = True
                template(c, "quota")
                continue
            except LLMError as exc:
                template(c, f"retry failed: {exc}"[:120])
                continue
            if again.value.reject:
                stats["rejected_by_llm"] += 1
                continue
            if verify_text(again.value, c):
                template(c, "text verification failed twice")
                continue
            draft, route = again.value, again.route
        stats["synthesized"] += 1
        out.append((c, draft, {"model": route, "prompt_version": version, "synthesized": True}))
    return out, stats
