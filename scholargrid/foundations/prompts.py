"""Optional plain-language reasons for reading-path items (task ``reason``, packable).

Template reasons are honest but dry; when an LLM is available, surveys and foundations get a "why read"
and "what you'll learn" based on the abstract only. On quota exhaustion the templates stay.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, field_validator

from ..llm.fakes import register_fake
from ..llm.prompts import Prompt, register


class ReadingReason(BaseModel):
    why_read: str             # <= 25 words, for a newcomer
    what_you_learn: str       # <= 25 words
    level: Literal["intro", "intermediate", "advanced"]

    @field_validator("why_read", "what_you_learn", mode="before")
    @classmethod
    def _c(cls, v: Any) -> str:
        s = " ".join(str(v or "").split())
        return s[:220]


register(Prompt(
    name="reading_reason", version="1",
    system=("You help a beginner researcher decide what to read. Given one paper's title and abstract and "
            "the topic it belongs to, write why a newcomer to the topic should read it and what they will "
            "learn. Base everything on the abstract only. Plain English, no hype, no citations counts."),
    user="Topic: $topic\nReading step: $step\n<abstract>\nTitle: $title\n$abstract\n</abstract>",
    packable=True,
))


@register_fake("reading_reason")
def _fake(v: dict) -> dict:
    return {"why_read": f"Introduces ideas central to {v['topic']}.",
            "what_you_learn": f"The core concepts behind {str(v['title'])[:60]}.", "level": "intro"}
