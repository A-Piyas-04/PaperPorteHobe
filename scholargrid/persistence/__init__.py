"""Durable projects (papers, open problems, plans, notes) in SQLite (dev) or Postgres (prod).
Starting research takes weeks; session-only storage loses everything on reload."""
from __future__ import annotations

from typing import Any, Optional

from ..config import Config

__all__ = ["get_repo", "ProjectRepo", "Access", "AccessDenied", "upgrade_to_head", "PAPER_STATUSES",
           "GAP_STATUSES"]


def __getattr__(name: str) -> Any:            # lazy: sqlalchemy is an optional extra
    if name in ("ProjectRepo", "Access", "AccessDenied", "PAPER_STATUSES", "GAP_STATUSES"):
        from . import repo

        return getattr(repo, name)
    if name == "upgrade_to_head":
        from .migrate import upgrade_to_head

        return upgrade_to_head
    raise AttributeError(name)


def get_repo(cfg: Config, migrate: bool = True) -> Optional[Any]:
    """A ProjectRepo (migrated to head), or ``None`` when persistence is disabled."""
    if not cfg["persistence"]["enabled"]:
        return None
    from .migrate import upgrade_to_head
    from .repo import ProjectRepo

    if migrate:
        upgrade_to_head(cfg)
    return ProjectRepo(cfg)
