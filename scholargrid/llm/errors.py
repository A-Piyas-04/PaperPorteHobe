"""Exceptions. Callers catch LLMError and fall back to heuristics."""
from __future__ import annotations


class LLMError(RuntimeError):
    """Any failure to obtain a valid structured answer."""


class LLMUnavailable(LLMError):
    """Mode 'none', no enabled provider, or missing/invalid credentials."""


class LLMRefusal(LLMError):
    """The provider blocked the prompt or the answer (safety/recitation)."""

    def __init__(self, reason: str | None):
        super().__init__(f"blocked by provider ({reason})")
        self.reason = reason


class LLMTruncated(LLMError):
    """Output hit max_output_tokens (thinking counts too): raise llm.tasks.<task>.max_output_tokens."""


class LLMInvalidOutput(LLMError):
    """Output was not valid JSON for the schema, even after one repair attempt."""


class QuotaExhausted(LLMError):
    """Every route entry is out of quota (daily limit reached or per-minute wait too long)."""

    def __init__(self, message: str, resets_at: float | None = None):
        super().__init__(message)
        self.resets_at = resets_at


class BudgetExceeded(LLMError):
    """A configured request budget (offline run, online daily/session) is used up."""
