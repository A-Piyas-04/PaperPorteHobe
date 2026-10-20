"""Gap candidates (internal) and Gap Cards (shown to users, stored in bundles/caches/projects)."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

EvidenceRole = Literal["states_problem", "related_attempt", "method_use", "task_use", "bridge", "narrow_eval",
                       "conflicting_result", "side_a", "side_b"]
NoveltyStatus = Literal["not_checked", "novel", "partially_explored", "already_addressed"]
Level = Literal["low", "medium", "high"]

TYPE_LABELS = {
    "stated_open_problem": "Open problem stated by authors",
    "long_standing_problem": "Long-standing open problem",
    "unexplored_combination": "Unexplored combination",
    "evaluation_gap": "Evaluation gap",
    "unsettled_question": "Unsettled question",
    "missing_bridge": "Missing connection between areas",
}
TYPE_HELP = {
    "stated_open_problem": "Several papers' authors say this is still unsolved.",
    "long_standing_problem": "Authors have called this open for two years or more, and still do.",
    "unexplored_combination": "A method and a problem that fit together, but no paper here combines them.",
    "evaluation_gap": "A popular method that has only been tested in narrow conditions.",
    "unsettled_question": "Papers report conflicting results on the same question.",
    "missing_bridge": "Two closely related lines of work that rarely cite each other.",
}
ROLE_LABELS = {
    "states_problem": "states the problem", "related_attempt": "possibly related later work",
    "method_use": "uses the method", "task_use": "works on the problem", "bridge": "bridging example",
    "narrow_eval": "evaluation setting", "conflicting_result": "reported result", "side_a": "one side",
    "side_b": "other side",
}


class Evidence(BaseModel):
    eid: str                         # "E1", "E2", ... unique within a card
    paper_id: str
    title: str
    year: Optional[int] = None
    url: str = ""
    role: EvidenceRole
    quote: str = ""                  # ONLY verified quotes from problems.parquet; else ""
    section: str = ""                # e.g. "Limitations"
    anchor_url: str = ""             # https://arxiv.org/html/<id>#<anchor> when known
    note: str = ""                   # e.g. "evaluated on: SQuAD, NQ" or a paraphrased finding (labelled)


class NoveltyHit(BaseModel):
    paper_id: str
    title: str
    url: str
    date: str = ""
    verdict: Literal["addresses", "partially", "unrelated", "unjudged"]
    reason: str = ""
    quote: str = ""                  # verified against the abstract, else ""
    score: float = 0.0


class NoveltyCheck(BaseModel):
    status: NoveltyStatus = "not_checked"
    scope: Literal["none", "collection", "collection+live"] = "none"
    checked_at: Optional[float] = None
    queries: List[str] = Field(default_factory=list)
    n_candidates: int = 0
    hits: List[NoveltyHit] = Field(default_factory=list)
    method: Literal["llm", "heuristic", "none"] = "none"


class Scores(BaseModel):
    evidence: float = 0.0
    novelty: float = 0.0
    feasibility: float = 0.0
    specificity: float = 0.0
    recency: float = 0.0
    total: float = 0.0
    explain: Dict[str, str] = Field(default_factory=dict)


class GapCard(BaseModel):
    id: str
    type: str
    title: str
    statement: str
    why_it_matters: str = ""
    why_it_might_not_be_a_gap: List[str]
    first_steps: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    data_needs: str = ""
    compute: str = "unknown"
    feasibility: Level = "medium"
    beginner_friendly: bool = False
    evidence: List[Evidence]
    novelty: NoveltyCheck = Field(default_factory=NoveltyCheck)
    scores: Scores = Field(default_factory=Scores)
    confidence: Level = "low"
    key_terms: Dict[str, str] = Field(default_factory=dict)
    stats: Dict = Field(default_factory=dict)
    scope: Dict = Field(default_factory=dict)          # {"kind": "area"|"workspace"|..., "id": ..., "label": ...}
    generated_by: Dict = Field(default_factory=dict)   # {"model", "prompt_version", "synthesized": bool}
    novelty_queries: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    version: int = 1


@dataclass
class GapCandidate:
    type: str
    scope: Dict
    key_terms: Dict[str, str]
    seed_statement: str
    evidence: List[Evidence]
    stats: Dict = field(default_factory=dict)
    specificity: float = 0.5           # 0..1
    strength: float = 0.0              # detector-specific pre-rank score (0..1)
    problem_ids: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        key = f"{self.type}|{sorted(self.key_terms.items())}|{self.scope.get('id')}|{self.seed_statement[:80]}"
        return "g" + hashlib.sha1(key.encode()).hexdigest()[:12]
