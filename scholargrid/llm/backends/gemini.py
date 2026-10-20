"""Google Gemini via google-genai (`client.models.generate_content`).

Docs (Oct 2026): structured output with response_mime_type="application/json" + response_schema=<pydantic
model>; thinking depth via ThinkingConfig(thinking_level=...) on Gemini 3.x; usage in usage_metadata.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from ..errors import LLMRefusal, LLMTruncated, LLMUnavailable
from . import AuthError, RawResult, classify_http_error

# Thinking levels each model accepts (from the Gemini thinking docs, Oct 2026). Unknown models -> omit.
THINKING_LEVELS = {
    "gemini-3.8-flash": ["low", "medium", "high"],
    "gemini-3.6-flash": ["minimal", "low", "medium", "high"],
    "gemini-3.5-flash-lite": ["minimal", "low", "medium", "high"],
}
_ORDER = ["minimal", "low", "medium", "high"]
_BLOCK = ("SAFETY", "PROHIBITED", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY")


def thinking_level_for(model: str, wanted: str) -> Optional[str]:
    if wanted == "default" or model not in THINKING_LEVELS:
        return None
    allowed = THINKING_LEVELS[model]
    if wanted in allowed:
        return wanted
    higher = [lvl for lvl in _ORDER[_ORDER.index(wanted):] if lvl in allowed]
    return higher[0] if higher else allowed[-1]


def _name(x: Any) -> str:
    return str(getattr(x, "name", x) or "")


class GeminiBackend:
    def __init__(self, provider_cfg: dict, timeout_s: float, api_key: Optional[str] = None, client: Any = None):
        if client is None:
            key = api_key or os.environ.get(provider_cfg["api_key_env"] or "", "")
            if not key:
                raise LLMUnavailable(f"set {provider_cfg['api_key_env']} (free key from Google AI Studio)")
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise LLMUnavailable("google-genai is not installed (pip install -e '.[llm]')") from exc

            client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self.client = client

    def generate(self, model: str, system: str, user: str, schema: type, *, thinking: str,
                 max_output_tokens: int, temperature: Optional[float]) -> RawResult:
        from google.genai import types

        level = thinking_level_for(model, thinking)
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_level=level) if level else None,  # type: ignore[arg-type]
        )
        try:
            resp = self.client.models.generate_content(model=model, contents=user, config=config)
        except Exception as exc:              # SDK errors carry .code; network errors don't
            err = classify_http_error(exc)
            if isinstance(err, AuthError):
                raise LLMUnavailable(f"Gemini rejected the key: {err}") from exc
            raise err from exc
        fb = getattr(resp, "prompt_feedback", None)
        if fb is not None and getattr(fb, "block_reason", None):
            raise LLMRefusal(_name(fb.block_reason))
        cands = getattr(resp, "candidates", None) or []
        reason = _name(getattr(cands[0], "finish_reason", "")) if cands else ""
        if any(b in reason for b in _BLOCK):
            raise LLMRefusal(reason)
        if "MAX_TOKENS" in reason:
            raise LLMTruncated(f"{model}: hit max_output_tokens={max_output_tokens} (thinking counts)")
        um = getattr(resp, "usage_metadata", None)
        usage = {"input_tokens": int(getattr(um, "prompt_token_count", 0) or 0),
                 "output_tokens": int(getattr(um, "candidates_token_count", 0) or 0),
                 "thinking_tokens": int(getattr(um, "thoughts_token_count", 0) or 0)}
        return RawResult(text=resp.text or "", usage=usage, model=model)
