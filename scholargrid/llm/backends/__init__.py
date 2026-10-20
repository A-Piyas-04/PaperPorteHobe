"""Provider backends return RawResult or raise one of the BackendError classes; the router decides."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RawResult:
    text: str
    usage: Dict[str, int] = field(default_factory=dict)   # input_tokens, output_tokens, thinking_tokens
    model: str = ""


class BackendError(Exception):
    pass


class RateLimited(BackendError):
    def __init__(self, daily: bool, retry_after: Optional[float]):
        super().__init__(f"rate limited (daily={daily}, retry_after={retry_after})")
        self.daily, self.retry_after = daily, retry_after


class Transient(BackendError):
    """5xx, timeouts, connection errors: retry with backoff."""


class AuthError(BackendError):
    pass


class BadRequest(BackendError):
    pass


_RETRY = re.compile(r"retry[_ ]?delay\W+(\d+(?:\.\d+)?)s", re.I)
_DAILY = re.compile(r"per\s*day|PerDay|RequestsPerDay|daily", re.I)


def classify_http_error(exc: Exception) -> BackendError:
    """Duck-typed: works for google-genai errors (.code, .details/.message) and openai errors
    (.status_code, .response.headers)."""
    raw_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    code: Optional[int]
    try:
        code = int(raw_code) if raw_code is not None else None
    except (TypeError, ValueError):
        code = None
    text = " ".join(str(getattr(exc, a, "") or "") for a in ("message", "details", "body")) + " " + str(exc)
    if code == 429:
        m = _RETRY.search(text)
        retry = float(m.group(1)) if m else None
        headers = getattr(getattr(exc, "response", None), "headers", None) or {}
        if retry is None and headers.get("retry-after"):
            try:
                retry = float(headers["retry-after"])
            except ValueError:
                retry = None
        return RateLimited(daily=bool(_DAILY.search(text)), retry_after=retry)
    if code in (401, 403):
        return AuthError(text[:300])
    if code is not None and 400 <= code < 500:
        return BadRequest(text[:500])
    return Transient(text[:300])
