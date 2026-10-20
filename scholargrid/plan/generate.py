"""generate_plan: grounded LLM plan (1 request) -> verify -> repair, or an honest template without an LLM."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..config import Config
from ..llm.prompts import PROMPTS
from . import prompts as _prompts  # noqa: F401  (registers "research_plan")
from .context import plan_context
from .schemas import Choice, Milestone, PlanDocument, ResearchPlan, Risk

_RANGE = re.compile(r"(\d+)\s*(?:[-–—]|to)\s*(\d+)")
_SINGLE = re.compile(r"(\d+)")
DATASET_RISK = "Dataset choice needs confirmation: none of the references lists a suitable dataset"


def _gap_text(card: Any) -> str:
    nov = card.novelty
    lines = [f"Title: {card.title}", f"Type: {card.type}", f"Statement: {card.statement}",
             "Why it might not be a gap: " + "; ".join(card.why_it_might_not_be_a_gap),
             "First steps: " + "; ".join(card.first_steps),
             f"Novelty check: {nov.status} ({nov.scope})"]
    return "\n".join(lines)


def _refs_text(refs: List[Dict]) -> str:
    out = []
    for r in refs:
        out.append(f"[{r['rid']}] ({r.get('year') or 'n.d.'}) {r['title']} | {r.get('summary', '')} | methods: "
                   f"{', '.join(r.get('methods', [])) or '-'} | datasets: {', '.join(r.get('datasets', [])) or '-'} | "
                   f"metrics: {', '.join(r.get('metrics', [])) or '-'} | code: {'yes' if r.get('code_url') else 'no'}")
    return "\n".join(out)


def _computed_text(computed: Dict) -> str:
    ds = ", ".join(f"{k} ({v})" for k, v in computed.get("dataset_counts", [])) or "none"
    ms = ", ".join(f"{k} ({v})" for k, v in computed.get("metric_counts", [])) or "none"
    return f"Datasets used by the referenced papers: {ds}\nMetrics used: {ms}"


def parse_weeks(text: str) -> Optional[Tuple[int, int]]:
    t = str(text or "")
    m = _RANGE.search(t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))
    m = _SINGLE.search(t)
    return (int(m.group(1)), int(m.group(1))) if m else None


def _milestone_issue(plan: ResearchPlan, weeks: int) -> Optional[str]:
    covered: List[int] = []
    for m in plan.milestones:
        r = parse_weeks(m.weeks)
        if r is None:
            return f"milestone weeks '{m.weeks}' could not be read"
        covered += list(range(r[0], r[1] + 1))
    if sorted(covered) != list(range(1, weeks + 1)):
        missing = sorted(set(range(1, weeks + 1)) - set(covered))
        extra = sorted({w for w in covered if covered.count(w) > 1} | {w for w in covered if w > weeks or w < 1})
        return f"milestones do not cover weeks 1-{weeks} exactly (missing {missing}, overlapping/out of range {extra})"
    return None


def _grounded(name: str, ref: Dict, field: str) -> bool:
    from rapidfuzz import fuzz

    hay = " ".join([*ref.get(field, []), ref.get("title", ""), ref.get("summary", "")]).lower()
    n = name.lower().strip()
    return bool(n) and (n in hay or fuzz.partial_ratio(n, hay) >= 85)


def verify_plan(plan: ResearchPlan, ctx: Dict, weeks: int) -> List[str]:
    """Problems found (empty = OK). Foreign entities are reported as warnings by the caller."""
    refs = {r["rid"]: r for r in ctx["references"]}
    issues: List[str] = []
    for kind, field, items in (("baseline", "methods", plan.baselines), ("dataset", "datasets", plan.datasets)):
        for ch in items:
            if ch.ref not in refs:
                issues.append(f"{kind} '{ch.name}' cites unknown reference {ch.ref}")
            elif not _grounded(ch.name, refs[ch.ref], field):
                issues.append(f"{kind} '{ch.name}' is not listed in {ch.ref}")
    bad = [r for r in plan.references_used if r not in refs]
    if bad:
        issues.append(f"unknown references used {bad}")
    ms = _milestone_issue(plan, weeks)
    if ms:
        issues.append(ms)
    return issues


def entity_warnings(plan: ResearchPlan, ctx: Dict) -> List[str]:
    from ..gaps.verify import foreign_entities

    vocab = " ".join(f"{r['title']} {r.get('summary', '')} {' '.join(r.get('methods', []))} "
                     f"{' '.join(r.get('datasets', []))} {' '.join(r.get('metrics', []))}"
                     for r in ctx["references"]).lower()
    foreign = foreign_entities(f"{plan.research_question} {plan.method_sketch}", vocab)
    return [f"Check these names, which are not in the references: {', '.join(foreign)}"] if foreign else []


def _even_milestones(goals: List[Tuple[str, str]], weeks: int) -> List[Milestone]:
    n = max(1, min(len(goals), weeks))
    bounds = [round(i * weeks / n) for i in range(n + 1)]
    out = []
    for i in range(n):
        a, b = bounds[i] + 1, bounds[i + 1]
        goal, deliverable = goals[i]
        out.append(Milestone(weeks=f"{a}-{b}" if b > a else str(a), goal=goal, tasks=[goal], deliverable=deliverable))
    return out


def repair_plan(plan: ResearchPlan, ctx: Dict, issues: List[str], weeks: int) -> ResearchPlan:
    refs = {r["rid"]: r for r in ctx["references"]}
    baselines = [c for c in plan.baselines if c.ref in refs and _grounded(c.name, refs[c.ref], "methods")]
    datasets = [c for c in plan.datasets if c.ref in refs and _grounded(c.name, refs[c.ref], "datasets")]
    risks = list(plan.risks)
    if len(datasets) < len(plan.datasets) or not datasets:
        risks.append(Risk(risk=DATASET_RISK, mitigation="Check the referenced papers' data sections and ask a "
                                                        "supervisor which public dataset fits."))
    milestones = plan.milestones
    if _milestone_issue(plan, weeks):
        goals = [(m.goal, m.deliverable) for m in plan.milestones] or [("Work on the project", "Progress notes")]
        milestones = _even_milestones(goals, weeks)
    return plan.model_copy(update={"baselines": baselines, "datasets": datasets, "risks": risks,
                                   "milestones": milestones,
                                   "references_used": [r for r in plan.references_used if r in refs]})


def template_plan(card: Any, ctx: Dict, weeks: int) -> ResearchPlan:
    refs = ctx["references"]
    comp = ctx["computed"]

    def ref_with(field: str, value: str) -> str:
        return next((r["rid"] for r in refs if value in r.get(field, [])), refs[0]["rid"] if refs else "R1")

    with_code = [r for r in refs if r.get("code_url") and r.get("methods")]
    base_src = with_code or [r for r in refs if r.get("methods")]
    baselines = [Choice(name=r["methods"][0], why="Proposed in a referenced paper" + (" with public code" if
                                                                                      r.get("code_url") else ""),
                        ref=r["rid"]) for r in base_src[:2]]
    datasets = [Choice(name=k, why=f"Used by {v} of the referenced papers", ref=ref_with("datasets", k))
                for k, v in comp.get("dataset_counts", [])[:2]]
    metrics = [k for k, _ in comp.get("metric_counts", [])[:2]]
    question = card.statement.rstrip(".")
    goals = [("Read the references and reproduce the closest baseline", "A working baseline and notes"),
             ("Run the baseline on the chosen data", "A first results table"),
             ("Design and implement one extension that targets the gap", "The extension, tested on a small run"),
             ("Run the main experiments and ablations", "A results table with error bars"),
             ("Write up the findings", "A short report or paper draft")]
    risks = [Risk(risk=w, mitigation="Check recent venues and ask a supervisor before investing heavily.")
             for w in card.why_it_might_not_be_a_gap]
    if not datasets:
        risks.append(Risk(risk=DATASET_RISK, mitigation="Ask a supervisor which public dataset fits."))
    return ResearchPlan(
        title=f"Starter project: {card.title}"[:140],
        research_question=f"In the setting described by the evidence, can we address this: {question}?",
        sub_questions=["How well does the closest existing method do on this problem?",
                       "Which single change targets the gap most directly?"],
        hypotheses=["A targeted extension of the closest baseline improves on the problem described in the gap."],
        why_it_matters=card.why_it_matters or "The authors of several papers describe this as unsolved.",
        contribution_type="empirical_study",
        method_sketch="Reproduce the closest baseline from the references, then add one change aimed at the gap "
                      "and measure it against the baseline on the same data.",
        baselines=baselines, datasets=datasets, metrics=metrics,
        experiments=["Baseline vs. extension on the same data and metric", "An ablation of the extension"],
        minimum_viable_result="A reproduced baseline plus one honest comparison, even if the result is negative.",
        stretch_goals=["Test a second dataset", "Release code and results"],
        milestones=_even_milestones(goals, weeks), risks=risks,
        skills_to_learn=list(card.skills) or ["Reading papers critically", "Running reproducible experiments"],
        compute_estimate=card.compute if card.compute != "unknown" else "Unknown; start small on a laptop or one GPU.",
        questions_for_supervisor=["Is this gap still open, as far as you know?",
                                  "Is the scope realistic for the time I have?",
                                  "Which venue or workshop would suit a first result?"],
        references_used=[r["rid"] for r in refs[:5]])


def generate_plan(card: Any, bundle: Any, cfg: Config, llm: Any, embedder: Any, *, weeks: int, background: str,
                  compute_access: str, goal: str, workspace: Any = None, attempt: int = 0) -> PlanDocument:
    from ..gaps.model import GapCard
    from ..llm import LLMError

    if isinstance(card, dict):
        card = GapCard.model_validate(card)
    ctx = plan_context(card, bundle, cfg, embedder, workspace)
    bg = str(background or "")[: int(cfg["plan"]["max_background_chars"])]
    inputs: Dict[str, Any] = {"weeks": int(weeks), "background": bg, "compute_access": compute_access, "goal": goal}
    warnings: List[str] = []
    try:
        if llm is None or not llm.available():
            raise LLMError("no LLM available")
        res = llm.structured("plan", "research_plan", ResearchPlan, weeks=weeks, goal=goal, compute=compute_access,
                             background=(bg or "not given").replace("</user_background>", "</ user_background>"),
                             gap=_gap_text(card), references=_refs_text(ctx["references"]),
                             computed=_computed_text(ctx["computed"]),
                             attempt=f"(attempt {attempt})" if attempt else "")
        plan, mode, model = res.value, "llm", res.route
        issues = verify_plan(plan, ctx, int(weeks))
        if issues:
            warnings += [f"Repaired: {i}" for i in issues]
            plan = repair_plan(plan, ctx, issues, int(weeks))
        warnings += entity_warnings(plan, ctx)
    except LLMError as exc:
        warnings.append(f"Plan written from a template ({type(exc).__name__}).")
        plan, mode, model = template_plan(card, ctx, int(weeks)), "template", "none"
    key = json.dumps({"gap": card.id, **inputs, "attempt": attempt}, sort_keys=True)
    return PlanDocument(id="p" + hashlib.sha1(key.encode()).hexdigest()[:12], gap_id=card.id,
                        gap_snapshot=card.model_dump(mode="json"), plan=plan, references=ctx["references"],
                        computed=ctx["computed"], inputs=inputs, warnings=warnings,
                        generated_by={"model": model, "mode": mode,
                                      "prompt_version": PROMPTS["research_plan"].version})
