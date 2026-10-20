"""SQLite answer cache (reruns are free; online pages are instant on repeat)."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Optional

from ..utils import ensure_dir


def cache_key(**parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, directory: str):
        ensure_dir(directory)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(os.path.join(directory, "llm_cache.sqlite"), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, task TEXT, route TEXT, "
                         "created REAL, value TEXT, usage TEXT)")
        self._db.commit()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._db.execute("SELECT route, value, usage FROM cache WHERE key=?", (key,)).fetchone()
        return None if not row else {"route": row[0], "value": json.loads(row[1]), "usage": json.loads(row[2] or "{}")}

    def put(self, key: str, task: str, route: str, value: Any, usage: Dict[str, int]) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?)",
                             (key, task, route, time.time(), json.dumps(value), json.dumps(usage)))
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()
