"""Gap Card -> starter research plan: computed context ([R#] references, baselines, datasets, venues,
people), one grounded LLM request (verified against the references), a template fallback, and exports.
It answers the beginner's next question: "so what do I actually do on Monday?"."""
from __future__ import annotations

from .context import plan_context
from .export import plan_to_bibtex, plan_to_json, plan_to_markdown, supervisor_email
from .generate import generate_plan, repair_plan, template_plan, verify_plan
from .schemas import Choice, Milestone, PlanDocument, ResearchPlan, Risk

__all__ = ["ResearchPlan", "PlanDocument", "Choice", "Milestone", "Risk", "generate_plan", "plan_context",
           "verify_plan", "repair_plan", "template_plan", "plan_to_markdown", "plan_to_json", "plan_to_bibtex",
           "supervisor_email"]
