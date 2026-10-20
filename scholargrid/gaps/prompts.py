"""LLM judges (packed, task ``judge``) and the Gap Card synthesizer (task ``synthesize_gap``)."""
from __future__ import annotations

from typing import Any, List, Literal

from pydantic import BaseModel, field_validator

from ..llm.fakes import register_fake
from ..llm.prompts import Prompt, register


def _clip(v: Any, n: int) -> str:
    s = " ".join(str(v or "").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _clip_list(v: Any, n_items: int = 5, n_chars: int = 200) -> List[str]:
    return [_clip(x, n_chars) for x in (v or []) if str(x).strip()][:n_items]


class ComboJudgement(BaseModel):
    meaningful: bool          # would applying A to B be a sensible research question?
    trivial: bool             # obvious/uninteresting, or a plain engineering exercise
    inapplicable_reason: str  # "" if applicable
    reason: str               # one sentence

    @field_validator("inapplicable_reason", "reason", mode="before")
    @classmethod
    def _c(cls, v: Any) -> str:
        return _clip(v, 300)


class ConflictJudgement(BaseModel):
    is_conflict: bool
    explanation: str
    possible_reasons: List[str]   # e.g. different splits, model sizes, metrics

    @field_validator("explanation", mode="before")
    @classmethod
    def _e(cls, v: Any) -> str:
        return _clip(v, 400)

    @field_validator("possible_reasons", mode="before")
    @classmethod
    def _r(cls, v: Any) -> List[str]:
        return _clip_list(v)


class AddressedJudgement(BaseModel):
    verdict: Literal["addresses", "partially", "unrelated"]
    reason: str
    quote: str                    # exact supporting sentence from the abstract, or ""

    @field_validator("reason", mode="before")
    @classmethod
    def _r(cls, v: Any) -> str:
        return _clip(v, 300)

    @field_validator("quote", mode="before")
    @classmethod
    def _q(cls, v: Any) -> str:
        return _clip(v, 400)


class GapDraft(BaseModel):
    reject: bool
    reject_reason: str
    title: str
    statement: str
    why_it_matters: str
    why_it_might_not_be_a_gap: List[str]
    first_steps: List[str]
    skills: List[str]
    data_needs: str
    compute: Literal["laptop", "single_gpu", "multi_gpu", "large_cluster", "unknown"]
    feasibility: Literal["low", "medium", "high"]
    beginner_friendly: bool
    evidence_used: List[str]
    novelty_queries: List[str]

    @field_validator("title", mode="before")
    @classmethod
    def _t(cls, v: Any) -> str:
        return _clip(v, 90)

    @field_validator("statement", mode="before")
    @classmethod
    def _s(cls, v: Any) -> str:
        return _clip(v, 400)

    @field_validator("why_it_matters", "reject_reason", "data_needs", mode="before")
    @classmethod
    def _w(cls, v: Any) -> str:
        return _clip(v, 300)

    @field_validator("why_it_might_not_be_a_gap", "first_steps", "skills", mode="before")
    @classmethod
    def _lists(cls, v: Any) -> List[str]:
        return _clip_list(v)

    @field_validator("evidence_used", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> List[str]:
        return [str(x).strip().strip("[]") for x in (v or []) if str(x).strip()][:12]

    @field_validator("novelty_queries", mode="before")
    @classmethod
    def _q(cls, v: Any) -> List[str]:
        return _clip_list(v, 4, 120)


register(Prompt(
    name="judge_combination", version="1",
    system=("You assess whether combining a method and a research problem is a sensible, non-trivial research "
            "question. Use general CS knowledge only to judge applicability, not to claim whether it has been "
            "done."),
    user="Method/value A: $a ($fa)\nProblem/value B: $b ($fb)\nBridge: $bridge\nEvidence:\n$evidence",
    packable=True,
))
register(Prompt(
    name="judge_conflict", version="1",
    system=("Decide if these reported results genuinely conflict on the same question (same task, data and "
            "metric), and list plausible reasons for the disagreement (different splits, model sizes, metrics, "
            "baselines). The results are paraphrases written by another system."),
    user="Question: $question\n$findings",
    packable=True,
))
register(Prompt(
    name="judge_addressed", version="1",
    system=("Decide whether the candidate paper already addresses the stated research gap. 'addresses' = it "
            "directly tackles the gap; 'partially' = covers part of it or a close variant; 'unrelated' "
            "otherwise. Quote one exact sentence from the abstract that supports your verdict, or leave quote "
            "empty."),
    user="<evidence>\nGap: $statement\n</evidence>\n<abstract>\nTitle: $title\n$abstract\n</abstract>",
    packable=True,
))

SYNTH_SYSTEM = """You write one "Gap Card" for a beginner researcher from a gap CANDIDATE found by an automatic detector.
Use ONLY the candidate's evidence and statistics. Rules:
1. If the candidate is not a real, meaningful research gap (trivial, nonsensical, already a standard
   practice, or not supported by the evidence), set reject=true and explain in reject_reason.
2. statement: 1-2 sentences, specific and testable. Say "in the papers analysed here", never
   "no one has ever". Name the method/task/setting explicitly.
3. Mention only methods, datasets, tasks and names that appear in the evidence or key terms.
   Do not introduce new named entities.
4. why_it_might_not_be_a_gap: 2-4 concrete, honest reasons (e.g. work may exist outside arXiv,
   in industry, under different names; the combination may be infeasible because ...).
5. first_steps: 3 concrete actions a beginner can start this month, referencing evidence ids
   (e.g. "Reproduce E2's baseline on ...").
6. evidence_used: the evidence ids you relied on. novelty_queries: 2-4 short search queries
   another system will use to look for papers that already did this.
7. feasibility/compute: judge from the evidence (datasets public? code available? model scale?)."""

register(Prompt(
    name="synthesize_gap", version="1", system=SYNTH_SYSTEM,
    user=("Gap type: $type_label - $type_help\nScope: $scope_label\nKey terms: $key_terms\n"
          "Detector statistics:\n$stats\n<evidence>\n$evidence\n</evidence>\n"
          "Detector's draft statement (may be clumsy): $seed"),
))


@register_fake("judge_combination")
def _fake_combo(v: dict) -> dict:
    return {"meaningful": True, "trivial": False, "inapplicable_reason": "", "reason": "Plausible."}


@register_fake("judge_conflict")
def _fake_conflict(v: dict) -> dict:
    return {"is_conflict": True, "explanation": "The reported directions differ.",
            "possible_reasons": ["different data splits"]}


@register_fake("judge_addressed")
def _fake_addressed(v: dict) -> dict:
    hit = "ADDRESSES_GAP" in str(v.get("abstract", ""))
    return {"verdict": "addresses" if hit else "unrelated", "reason": "Planted test hit." if hit else "Different focus.",
            "quote": ""}


@register_fake("synthesize_gap")
def _fake_synth(v: dict) -> dict:
    import re

    terms = str(v.get("key_terms", ""))
    ids = sorted(set(re.findall(r"\bE\d+\b", str(v.get("evidence", "")))), key=lambda s: int(s[1:]))
    seed = str(v.get("seed", "")).split("\nPrevious draft")[0]
    return {"reject": False, "reject_reason": "", "title": (terms[:80] or "Research gap"),
            "statement": seed + " (in the papers analysed here)",
            "why_it_matters": "Addressing it would make the methods more useful in practice.",
            "why_it_might_not_be_a_gap": ["Work on this may exist outside the papers analysed here.",
                                          "It may be studied under a different name."],
            "first_steps": ["Read the evidence papers.", "Reproduce one baseline from E1.",
                            "Write down one testable hypothesis."],
            "skills": ["Python", "experiment design"], "data_needs": "Public datasets named in the evidence.",
            "compute": "single_gpu", "feasibility": "medium", "beginner_friendly": True,
            "evidence_used": ids, "novelty_queries": [terms[:100] or seed[:100]]}
