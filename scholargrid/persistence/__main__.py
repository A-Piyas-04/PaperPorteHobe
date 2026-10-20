"""CLI for the projects database.

    python -m scholargrid.persistence upgrade               # alembic upgrade head (uses SCHOLARGRID_DB_URL)
    python -m scholargrid.persistence stats                 # projects (owned/anonymous), items, last 7 days
    python -m scholargrid.persistence purge [--days 365]    # delete anonymous projects not updated since
    python -m scholargrid.persistence purge-events [--days 180]   # delete old telemetry events
    python -m scholargrid.persistence export <project_id> --out p.json   # admin export (no token needed)
"""
from __future__ import annotations

import argparse
import json
from typing import List, Optional

from ..config import load_config


def _redacted(url: str) -> str:
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"
    return url


def main(argv: Optional[List[str]] = None) -> None:
    from .db import resolve_url
    from .migrate import upgrade_to_head
    from .repo import ProjectRepo

    p = argparse.ArgumentParser(prog="python -m scholargrid.persistence", description="Projects database tools")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("upgrade", "stats"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=argparse.SUPPRESS)
    pg = sub.add_parser("purge")
    pg.add_argument("--days", type=int, default=None)
    pg.add_argument("--config", default=argparse.SUPPRESS)
    pe = sub.add_parser("purge-events", help="delete telemetry events older than telemetry.retention_days")
    pe.add_argument("--days", type=int, default=None)
    pe.add_argument("--config", default=argparse.SUPPRESS)
    ex = sub.add_parser("export")
    ex.add_argument("project_id")
    ex.add_argument("--out", required=True)
    ex.add_argument("--config", default=argparse.SUPPRESS)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    print(f"database: {_redacted(resolve_url(cfg))}")
    upgrade_to_head(cfg)
    repo = ProjectRepo(cfg)
    if args.cmd == "upgrade":
        print("schema is at head")
    elif args.cmd == "stats":
        for k, v in repo.stats().items():
            print(f"  {k:22s} {v}")
    elif args.cmd == "purge":
        days = args.days or int(cfg["persistence"]["anonymous_retention_days"])
        print(f"purged {repo.purge_anonymous(days)} anonymous projects not updated in {days} days")
    elif args.cmd == "purge-events":
        days = args.days or int(cfg["telemetry"]["retention_days"])
        print(f"purged {repo.purge_events(days)} telemetry events older than {days} days")
    elif args.cmd == "export":
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(repo.admin_export(args.project_id), fh, indent=2, ensure_ascii=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
