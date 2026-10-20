"""Programmatic ``alembic upgrade head`` (app start, tests, CLI). Idempotent."""
from __future__ import annotations

import os

from ..config import REPO_ROOT, Config
from .db import resolve_url

MIGRATIONS = os.path.join(REPO_ROOT, "scholargrid", "persistence", "migrations")


def alembic_config(cfg: Config):  # -> alembic.config.Config
    from alembic.config import Config as AlembicConfig

    ac = AlembicConfig(os.path.join(REPO_ROOT, "alembic.ini"))
    ac.set_main_option("script_location", MIGRATIONS)
    ac.set_main_option("sqlalchemy.url", resolve_url(cfg).replace("%", "%%"))
    ac.attributes["configure_logger"] = False
    return ac


def upgrade_to_head(cfg: Config) -> None:
    from alembic import command

    command.upgrade(alembic_config(cfg), "head")
