"""Deterministic fake handlers keyed by prompt name. Feature plans register theirs here."""
from __future__ import annotations

from typing import Any, Callable, Dict

FakeHandler = Callable[[Dict[str, Any]], Dict[str, Any]]

FAKE_HANDLERS: Dict[str, FakeHandler] = {
    "check": lambda v: {"ok": True, "echo": str(v.get("word", ""))},
}


def register_fake(prompt: str) -> Callable[[FakeHandler], FakeHandler]:
    def deco(fn: FakeHandler) -> FakeHandler:
        FAKE_HANDLERS[prompt] = fn
        return fn
    return deco
