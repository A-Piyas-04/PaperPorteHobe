"""Online free-quota guard for the SERVER key: product limits that hold across processes and instances.

The app may use at most ``online_daily_requests`` per day (the rest of the project's free quota is left to
the pipeline) and ``online_session_requests`` per browser session (one visitor can't use up everyone's
allowance). Counts live in the database (``llm_usage_daily``), with the day in the provider's reset
timezone. Never installed on bring-your-own-key routers: those spend the visitor's own quota.
"""
from __future__ import annotations

from typing import Any

from .context import current_session
from .errors import BudgetExceeded


class OnlineGuard:
    def __init__(self, cfg: Any, repo: Any):
        self.daily = int(cfg["llm"]["budget"]["online_daily_requests"])
        self.session = int(cfg["llm"]["budget"]["online_session_requests"])
        self.repo = repo

    def check(self, task: str) -> None:
        sid = current_session.get()
        if self.repo.requests_today("global") >= self.daily:
            raise BudgetExceeded("daily shared AI allowance used up")
        if sid and self.repo.requests_today(f"session:{sid}") >= self.session:
            raise BudgetExceeded("session AI allowance used up")

    def record(self, task: str, route: str) -> None:
        sid = current_session.get()
        self.repo.add_request(["global"] + ([f"session:{sid}"] if sid else []))

    def state(self) -> dict:
        sid = current_session.get()
        used = self.repo.requests_today("global")
        mine = self.repo.requests_today(f"session:{sid}") if sid else 0
        return {"daily_used": used, "daily_limit": self.daily, "session_used": mine, "session_limit": self.session,
                "exhausted": used >= self.daily or (bool(sid) and mine >= self.session)}


def install_guard(llm: Any, cfg: Any, repo: Any) -> OnlineGuard | None:
    """Attach the guard to a live router's ledger (no-op for fake/none clients)."""
    if repo is None or getattr(llm, "provider", "") != "live" or getattr(llm, "ledger", None) is None:
        return None
    guard = OnlineGuard(cfg, repo)
    llm.ledger.max_requests = None              # online: no per-process run cap; the guard applies
    llm.ledger.extra_guard = guard.check
    llm.ledger.on_record = guard.record
    return guard
