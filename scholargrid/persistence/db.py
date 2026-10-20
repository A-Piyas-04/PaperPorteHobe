"""One engine per process; SQLite (dev) or Postgres (prod) from SCHOLARGRID_DB_URL."""
from __future__ import annotations

import os
import threading
from typing import Any, Dict

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from ..config import REPO_ROOT, Config

_ENGINES: Dict[str, Engine] = {}
_LOCK = threading.Lock()


def resolve_url(cfg: Config) -> str:
    p = cfg["persistence"]
    url = os.environ.get(p["db_url_env"]) or p["default_url"]
    if url.startswith("postgres://"):                 # Heroku/Supabase style -> SQLAlchemy driver URL
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        rest = url[len("sqlite:///"):]
        path = rest if os.path.isabs(rest) else os.path.join(REPO_ROOT, rest)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        url = "sqlite:///" + path.replace("\\", "/")
    return url


def get_engine(cfg: Config) -> Engine:
    url = resolve_url(cfg)
    with _LOCK:
        if url in _ENGINES:
            return _ENGINES[url]
        if url.startswith("sqlite"):
            engine = create_engine(url, connect_args={"check_same_thread": False}, pool_pre_ping=True)

            @event.listens_for(engine, "connect")
            def _sqlite_pragmas(dbapi_conn: Any, _: Any) -> None:
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        else:
            engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5, pool_recycle=1800)
        _ENGINES[url] = engine
        return engine


def dispose_all() -> None:
    """Close every engine (tests; Windows can't delete an open SQLite file)."""
    with _LOCK:
        for e in _ENGINES.values():
            e.dispose()
        _ENGINES.clear()


def session_factory(cfg: Config) -> sessionmaker:
    return sessionmaker(get_engine(cfg), expire_on_commit=False)
