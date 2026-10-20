"""Per-session rate limits and a process-wide cap on concurrent expensive builds.

Cached results bypass both. Sessions using their own AI key get 3x the limits: they spend their own quota.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator

import streamlit as st

LIMITS = {"workspace_build": (10, 3600), "gaps_generate": (4, 3600), "plan_generate": (4, 3600),
          "novelty_check": (15, 3600), "project_create": (10, 3600)}
BYOK_FACTOR = 3
_SEMAPHORES: dict = {}
_SEM_LOCK = threading.Lock()


class Busy(RuntimeError):
    """All heavy-work slots are taken."""


def _limit(action: str) -> tuple[int, int]:
    try:
        import ui

        configured = ui.get_config()["app"]["limits"].get(action)
        if configured:
            return int(configured[0]), int(configured[1])
    except Exception:
        pass
    return LIMITS.get(action, (10, 3600))


def allow(action: str, record: bool = True) -> bool:
    """Sliding-window limit per browser session (timestamps in st.session_state)."""
    n, window = _limit(action)
    if st.session_state.get("_byok_llm") is not None:
        n *= BYOK_FACTOR
    now = time.time()
    key = f"_rl_{action}"
    recent = [t for t in st.session_state.get(key, []) if now - t < window]
    if len(recent) >= n:
        st.session_state[key] = recent
        return False
    if record:
        recent.append(now)
    st.session_state[key] = recent
    return True


def limit_message(action: str) -> str:
    n, window = _limit(action)
    return (f"You've reached the limit of {n} of these per {window // 60} minutes for this session. "
            "Please wait a little, or add your own free AI key in Settings for higher limits.")


def _semaphore(n: int) -> threading.BoundedSemaphore:
    with _SEM_LOCK:
        if n not in _SEMAPHORES:
            _SEMAPHORES[n] = threading.BoundedSemaphore(n)
        return _SEMAPHORES[n]


@contextmanager
def heavy_slot(timeout: float = 30.0, slots: int | None = None) -> Iterator[None]:
    """Waits for a free slot; on timeout raises Busy so the page can say 'Many users right now, try again shortly'."""
    if slots is None:
        try:
            import ui

            slots = int(ui.get_config()["app"]["heavy_slots"])
        except Exception:
            slots = 2
    sem = _semaphore(int(slots))
    if not sem.acquire(timeout=timeout):
        raise Busy("Many people are using ScholarGrid right now. Please try again in a moment.")
    try:
        yield
    finally:
        sem.release()
