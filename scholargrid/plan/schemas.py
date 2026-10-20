"""ResearchPlan (LLM output) and PlanDocument (what is shown, saved and exported)."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator


def _clip(v: Any, n: int) -> str:
    s = " ".join(str(v or "").split())
    return s[:n]


class Choice(BaseModel):
    name: str
    why: str
    ref: str               # "R3" - must be one of the provided reference ids

    @field_validator("name", "why", mode="before")
    @classmethod
    def _c(cls, v: Any) -> str:
        return _clip(v, 240)

    @field_validator("ref", mode="before")
    @classmethod
    def _r(cls, v: Any) -> str:
        return _clip(v, 20).strip("[]")


class Milestone(BaseModel):
    weeks: str             # e.g. "1-2"
    goal: str
    tasks: List[str]
    deliverable: str

    @field_validator("weeks", "goal", "deliverable", mode="before")
    @classmethod
    def _c(cls, v: Any) -> str:
        return _clip(v, 300)

    @field_validator("tasks", mode="before")
    @classmethod
    def _t(cls, v: Any) -> List[str]:
        return [_clip(x, 300) for x in (v or [])][:8]


class Risk(BaseModel):
    risk: str
    mitigation: str

    @field_validator("risk", "mitigation", mode="before")
    @classmethod
    def _c(cls, v: Any) -> str:
        return _clip(v, 400)


class ResearchPlan(BaseModel):
    title: str
    research_question: str
    sub_questions: List[str]
    hypotheses: List[str]
    why_it_matters: str
    contribution_type: Literal["new_method", "empirical_study", "benchmark_or_dataset", "reproduction",
                               "system_or_tool", "analysis"]
    method_sketch: str
    baselines: List[Choice]
    datasets: List[Choice]
    metrics: List[str]
    experiments: List[str]
    minimum_viable_result: str
    stretch_goals: List[str]
    milestones: List[Milestone]
    risks: List[Risk]
    skills_to_learn: List[str]
    compute_estimate: str
    questions_for_supervisor: List[str]
    references_used: List[str]

    @field_validator("title", mode="before")
    @classmethod
    def _title(cls, v: Any) -> str:
        return _clip(v, 140)

    @field_validator("research_question", "why_it_matters", "minimum_viable_result", "compute_estimate",
                     mode="before")
    @classmethod
    def _short(cls, v: Any) -> str:
        return _clip(v, 600)

    @field_validator("method_sketch", mode="before")
    @classmethod
    def _sketch(cls, v: Any) -> str:
        return _clip(v, 1500)

    @field_validator("sub_questions", "hypotheses", "metrics", "experiments", "stretch_goals",
                     "skills_to_learn", "questions_for_supervisor", mode="before")
    @classmethod
    def _lists(cls, v: Any) -> List[str]:
        return [_clip(x, 300) for x in (v or [])][:8]

    @field_validator("references_used", mode="before")
    @classmethod
    def _refs(cls, v: Any) -> List[str]:
        return [str(x).strip().strip("[]") for x in (v or []) if str(x).strip()][:40]

    @field_validator("milestones", "baselines", "datasets", "risks", mode="before")
    @classmethod
    def _ms(cls, v: Any) -> list:
        return list(v or [])[:12]


class PlanDocument(BaseModel):
    id: str
    gap_id: str
    gap_snapshot: Dict                     # the GapCard as it was (model_dump)
    plan: ResearchPlan
    references: List[Dict]                 # [{rid, paper_id, title, year, url, role, ...}]
    computed: Dict                         # venues, people, code_links, dataset_counts, metric_counts, reading
    inputs: Dict                           # weeks, background, compute_access, goal
    warnings: List[str] = Field(default_factory=list)
    generated_by: Dict = Field(default_factory=dict)   # {"model", "prompt_version", "mode": "llm"|"template"}
    created_at: float = Field(default_factory=time.time)
