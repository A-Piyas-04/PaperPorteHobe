"""PaperExtraction: what the LLM (or the heuristic) returns for one paper.

LLM-output models follow the plan 01 rule: no length/number constraints in ``Field``; validators clip.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Modality = Literal["text", "image", "video", "audio", "speech", "code", "tabular", "graph",
                   "time_series", "3d", "multimodal", "other"]
ContributionType = Literal["new_method", "benchmark_or_dataset", "empirical_study", "survey", "theory",
                           "system_or_tool", "position", "reproduction", "application"]
ProblemType = Literal["limitation_of_this_work", "future_work", "open_challenge_in_field",
                      "negative_result", "unexplored_setting"]
Direction = Literal["improves", "worsens", "no_effect", "mixed"]
Compute = Literal["unknown", "laptop", "single_gpu", "multi_gpu", "large_cluster"]
Role = Literal["proposed", "baseline", "used"]


def _clip(v: object, n: int) -> str:
    s = " ".join(str(v or "").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _clean_list(values: Any, n_items: int, n_chars: int = 80) -> List[str]:
    out, seen = [], set()
    for v in values or []:
        s = _clip(v, n_chars).strip(" .;,")
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out[:n_items]


class MethodMention(BaseModel):
    name: str = Field(description="Canonical method/model name as commonly written, e.g. 'LoRA', "
                                  "'speculative decoding', 'BERT'. Not a description.")
    role: Role = Field(description="proposed = introduced or extended by this paper; baseline = compared "
                                   "against; used = applied as a tool/component")

    @field_validator("name", mode="before")
    @classmethod
    def _n(cls, v: Any) -> str:
        return _clip(v, 80)


class StatedProblem(BaseModel):
    statement: str = Field(description="Self-contained paraphrase (<= 40 words) of an open problem the "
                                       "AUTHORS state. Name the method/task explicitly; never write "
                                       "'our method'.")
    quote: str = Field(description="Exact sentence(s) copied character-for-character from the paper that "
                                   "state this problem (<= 300 characters).")
    section: str = Field(description="The [S#] marker of the section the quote comes from, e.g. 'S5'.")
    problem_type: ProblemType
    specificity: int = Field(description="1 = vague ('more work is needed'), 2 = names the issue, "
                                         "3 = concrete and testable (setting, data or metric named)")

    @field_validator("statement", mode="before")
    @classmethod
    def _s(cls, v: Any) -> str:
        return _clip(v, 320)

    @field_validator("quote", mode="before")
    @classmethod
    def _q(cls, v: Any) -> str:
        return _clip(v, 400)

    @field_validator("section", mode="before")
    @classmethod
    def _sec(cls, v: Any) -> str:
        return str(v or "").strip().strip("[]")

    @field_validator("specificity", mode="before")
    @classmethod
    def _sp(cls, v: Any) -> int:
        try:
            return min(3, max(1, int(v)))
        except (TypeError, ValueError):
            return 1


class Finding(BaseModel):
    claim: str = Field(description="One empirical result in <= 30 words")
    task: str
    dataset: str = Field(description="Dataset/benchmark name or '' if not stated")
    metric: str = Field(description="Metric name or '' if not stated")
    direction: Direction = Field(description="Effect of the paper's method/intervention vs the comparison")
    compared_to: str = Field(description="What it is compared against, or ''")

    @field_validator("claim", "task", "dataset", "metric", "compared_to", mode="before")
    @classmethod
    def _c(cls, v: Any) -> str:
        return _clip(v, 200)


class EvalProfile(BaseModel):
    real_world: Optional[bool] = Field(description="Evaluated on real deployment / in-the-wild data")
    human_eval: Optional[bool] = Field(description="Includes human evaluation or a user study")
    robustness: Optional[bool] = Field(description="Tests robustness / OOD / adversarial conditions")
    efficiency_reported: Optional[bool] = Field(description="Reports latency, memory or compute cost")
    low_resource: Optional[bool] = Field(description="Evaluates low-resource data, languages or settings")
    multilingual: Optional[bool] = Field(description="Evaluates more than one natural language")


class PaperExtraction(BaseModel):
    one_line_summary: str = Field(description="What the paper does, <= 30 words, plain English")
    tasks: List[str] = Field(description="Research tasks/problems addressed, short noun phrases, e.g. "
                                         "'machine translation', 'code generation'")
    methods: List[MethodMention]
    datasets: List[str] = Field(description="Named datasets/benchmarks actually used")
    metrics: List[str]
    domains: List[str] = Field(description="Application domains, e.g. 'healthcare', 'legal'; ['general'] if none")
    languages: List[str] = Field(description="Natural languages of the data, e.g. 'English', 'Bengali'; [] if n/a")
    modalities: List[Modality]
    contribution_types: List[ContributionType]
    evaluation: EvalProfile
    open_problems: List[StatedProblem] = Field(description="Problems the authors say remain open, "
                                                           "their own limitations, or future work. [] if none.")
    findings: List[Finding]
    code_url: Optional[str] = Field(description="Code repository URL if stated, else null")
    data_released: Optional[bool]
    compute: Compute

    @field_validator("one_line_summary", mode="before")
    @classmethod
    def _sum(cls, v: Any) -> str:
        return _clip(v, 240)

    @field_validator("tasks", "datasets", "metrics", "domains", "languages", mode="before")
    @classmethod
    def _lists(cls, v: Any) -> List[str]:
        return _clean_list(v, 12)

    @field_validator("methods", mode="before")
    @classmethod
    def _methods(cls, v: Any) -> list:
        return list(v or [])[:15]

    @field_validator("open_problems", mode="before")
    @classmethod
    def _probs(cls, v: Any) -> list:
        return list(v or [])[:8]

    @field_validator("findings", mode="before")
    @classmethod
    def _finds(cls, v: Any) -> list:
        return list(v or [])[:6]

    @field_validator("code_url", mode="before")
    @classmethod
    def _url(cls, v: Any) -> Optional[str]:
        s = str(v or "").strip()
        return s if s.startswith(("http://", "https://")) else None
