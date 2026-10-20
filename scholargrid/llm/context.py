"""Per-request context for the online quota guard: which browser session is asking.

The app sets ``current_session`` at the top of every script run; the router copies the context into
worker threads in ``map_structured``, and ``structured`` / ``structured_many`` run in the caller's thread.
"""
from __future__ import annotations

import contextvars

current_session: contextvars.ContextVar[str | None] = contextvars.ContextVar("sg_session", default=None)
