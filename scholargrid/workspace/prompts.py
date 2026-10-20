"""Prompts for sub-topic names/summaries and the topic overview (task ``label``)."""
from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, field_validator

from ..llm.fakes import register_fake
from ..llm.prompts import Prompt, register


class Concept(BaseModel):
    term: str
    meaning: str          # one short sentence a newcomer understands


class SubtopicLabel(BaseModel):
    name: str             # <= 6 words, specific, no buzzwords
    summary: str          # 2 sentences: what problem this group works on and how
    key_concepts: List[Concept]
    beginner_entry: str   # one sentence: what to read or try first, based only on the listed papers

    @field_validator("name", mode="before")
    @classmethod
    def _n(cls, v: Any) -> str:
        return " ".join(str(v or "").split()[:8])

    @field_validator("summary", "beginner_entry", mode="before")
    @classmethod
    def _s(cls, v: Any) -> str:
        return " ".join(str(v or "").split())[:600]

    @field_validator("key_concepts", mode="before")
    @classmethod
    def _k(cls, v: Any) -> list:
        return list(v or [])[:6]


class TopicOverview(BaseModel):
    summary: str                  # 3-4 sentences about the whole topic as reflected by these papers
    how_they_relate: str          # 2 sentences on how the sub-topics relate
    prerequisites: List[str]      # 3-6 things to know first (concepts/skills)

    @field_validator("prerequisites", mode="before")
    @classmethod
    def _p(cls, v: Any) -> List[str]:
        return [" ".join(str(x).split())[:80] for x in (v or []) if str(x).strip()][:6]


register(Prompt(
    name="subtopic_label", version="1",
    system=("You name and explain one group of related research papers for a beginner researcher. "
            "Use only the evidence given (keywords, frequent methods/datasets, paper titles and one-line "
            "summaries). Be specific: prefer 'Low-rank adapters for LLM fine-tuning' over 'Efficient methods'. "
            "Explain jargon in key_concepts. Do not claim trends or quality."),
    user=("User's topic: $query\nKeywords: $keywords\nFrequent facets: $facets\n"
          "$papers"),
    packable=True,
))
register(Prompt(
    name="topic_overview", version="1",
    system=("You orient a beginner researcher in a topic using only the sub-topic summaries provided. "
            "Plain English. No hype, no claims about what is best."),
    user="User's topic: $query\n$subtopics",
))


@register_fake("subtopic_label")
def _fake_label(v: dict) -> dict:
    kw = [k for k in str(v["keywords"]).split(", ") if k]
    return {"name": " ".join(kw[:3]).title() or "Group", "summary": f"Papers about {', '.join(kw[:3])}.",
            "key_concepts": [{"term": k, "meaning": f"A concept related to {k}."} for k in kw[:3]],
            "beginner_entry": "Start with the most cited paper listed."}


@register_fake("topic_overview")
def _fake_overview(v: dict) -> dict:
    return {"summary": f"An overview of {v['query']}.", "how_they_relate": "The groups share methods.",
            "prerequisites": ["linear algebra", "machine learning basics"]}
