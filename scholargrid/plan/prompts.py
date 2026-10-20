"""The research-plan prompt (task ``plan``, one request per plan) and its fake handler."""
from __future__ import annotations

import re

from ..llm.fakes import register_fake
from ..llm.prompts import Prompt, register

PLAN_SYSTEM = """You help an early-career researcher turn one research gap into a realistic starter project.
Work ONLY from the gap card and the numbered references [R#] provided. Rules:
1. Baselines and datasets must be chosen from the references' listed methods/datasets; give the [R#] in
   'ref'. If no suitable dataset is listed, say so in risks and propose how to obtain one.
2. Keep the scope achievable in the given number of weeks by ONE person with the stated background and
   compute. Prefer a reproduction + one clear extension over a grand new system.
3. minimum_viable_result: the smallest result that is still worth writing up.
4. milestones: cover exactly the given number of weeks, contiguous ranges ("1-2", "3-4", ...), each with
   a concrete deliverable (a working baseline, a results table, a draft section).
5. risks: include at least one risk that the gap is already addressed somewhere not checked, and how to
   check (search terms, venues to scan, asking a supervisor).
6. questions_for_supervisor: 3-5 specific questions that would help decide whether to pursue this.
7. Plain English. Explain jargon briefly the first time. No claims about novelty beyond the gap card's
   novelty check."""

register(Prompt(name="research_plan", version="1", system=PLAN_SYSTEM, user=(
    "Weeks available: $weeks\nGoal: $goal\nCompute access: $compute\n"
    "<user_background>\n$background\n</user_background>\n\n"
    "Gap card:\n$gap\n\n<evidence>\n$references\n</evidence>\n\n"
    "Computed facts (for your information; do not restate as your own claims):\n$computed\n$attempt")))

_REF = re.compile(r"^\[(R\d+)\][^|]*\|[^|]*\| methods: ([^|]*)\| datasets: ([^|]*)\| metrics: ([^|]*)", re.M)


@register_fake("research_plan")
def _fake_plan(v: dict) -> dict:
    weeks = int(v.get("weeks", 8))
    refs = _REF.findall(str(v.get("references", "")))
    base = next(((rid, m.split(",")[0].strip()) for rid, m, _, _ in refs if m.strip() and m.strip() != "-"), None)
    data = next(((rid, d.split(",")[0].strip()) for rid, _, d, _ in refs if d.strip() and d.strip() != "-"), None)
    metric = next((m.split(",")[0].strip() for *_, m in refs if m.strip() and m.strip() != "-"), "accuracy")
    ms: list = []
    start = 1
    while start <= weeks:
        end = min(weeks, start + 1)
        ms.append({"weeks": f"{start}-{end}" if end > start else str(start), "goal": f"Step {len(ms) + 1}",
                   "tasks": ["Do the work."], "deliverable": "A short write-up."})
        start = end + 1
    return {"title": "Starter plan", "research_question": "Does the gap hold up when tested directly?",
            "sub_questions": ["What is the baseline performance?"], "hypotheses": ["The extension helps."],
            "why_it_matters": "It tests an open problem.", "contribution_type": "empirical_study",
            "method_sketch": "Reproduce the baseline, then add one extension.",
            "baselines": [{"name": base[1], "why": "Listed in the references.", "ref": base[0]}] if base else [],
            "datasets": [{"name": data[1], "why": "Used by the references.", "ref": data[0]}] if data else [],
            "metrics": [metric], "experiments": ["Compare against the baseline."],
            "minimum_viable_result": "A reproduced baseline and one comparison.", "stretch_goals": [],
            "milestones": ms, "risks": [{"risk": "The gap may already be addressed elsewhere.",
                                         "mitigation": "Search recent venues and ask a supervisor."}],
            "skills_to_learn": ["experiment tracking"], "compute_estimate": "One GPU for a few days.",
            "questions_for_supervisor": ["Is this scope realistic?", "Which venue fits?", "Who else works on it?"],
            "references_used": [rid for rid, *_ in refs[:3]]}
