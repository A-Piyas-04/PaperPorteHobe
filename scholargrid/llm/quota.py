"""Per-route quota: requests/minute and tokens/minute (in-memory sliding window) plus requests/day
(persistent SQLite counter, day boundary in the provider's timezone). A 429 that signals a daily limit
marks the route exhausted until the next reset; a per-minute 429 just waits.

The per-minute windows are per process. The daily counters are shared by every process using the
same ``usage_dir``, so pipeline and app share the count on one machine.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Deque, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from ..utils import ensure_dir, get_logger
from .errors import QuotaExhausted

log = get_logger("llm.quota")


class QuotaManager:
    def __init__(self, usage_dir: str, limits: Dict[str, Dict], tz_of: Dict[str, str]):
        ensure_dir(usage_dir)
        self.limits = limits                      # "provider:model" -> {rpm, rpd, tpm}
        self.tz_of = tz_of                        # provider -> timezone name
        self._lock = threading.Lock()
        self._req: Dict[str, Deque[float]] = defaultdict(deque)
        self._tok: Dict[str, Deque[Tuple[float, int]]] = defaultdict(deque)
        self._db = sqlite3.connect(os.path.join(usage_dir, "quota.sqlite"), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS daily (day TEXT, route TEXT, requests INTEGER, tokens INTEGER,"
                         " PRIMARY KEY (day, route))")
        self._db.execute("CREATE TABLE IF NOT EXISTS exhausted (route TEXT PRIMARY KEY, until REAL)")
        self._db.commit()

    # -- day boundaries -----------------------------------------------------------------
    def _tz(self, route: str) -> ZoneInfo:
        return ZoneInfo(self.tz_of.get(route.split(":", 1)[0], "UTC"))

    def _now(self, route: str) -> datetime:
        return datetime.now(self._tz(route))

    def day(self, route: str) -> str:
        return self._now(route).strftime("%Y-%m-%d")

    def next_reset(self, route: str) -> float:
        now = self._now(route)
        return (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).timestamp()

    # -- reads --------------------------------------------------------------------------
    def used_today(self, route: str) -> int:
        with self._lock:
            row = self._db.execute("SELECT requests FROM daily WHERE day=? AND route=?",
                                   (self.day(route), route)).fetchone()
        return int(row[0]) if row else 0

    def remaining_today(self, route: str) -> Optional[int]:
        rpd = (self.limits.get(route) or {}).get("rpd")
        return None if rpd is None else max(0, int(rpd) - self.used_today(route))

    def exhausted_until(self, route: str) -> Optional[float]:
        with self._lock:
            row = self._db.execute("SELECT until FROM exhausted WHERE route=?", (route,)).fetchone()
        return float(row[0]) if row and float(row[0]) > time.time() else None

    def routes_seen(self) -> list[str]:
        with self._lock:
            rows = self._db.execute("SELECT DISTINCT route FROM daily UNION SELECT route FROM exhausted").fetchall()
        return sorted({r[0] for r in rows} | set(self.limits))

    # -- acquire / record -----------------------------------------------------------------
    def acquire(self, route: str, est_tokens: int, max_wait_s: float) -> None:
        """Block until a request may be sent on ``route``; raise QuotaExhausted if the daily quota is
        gone or the per-minute wait would exceed ``max_wait_s``."""
        until = self.exhausted_until(route)
        if until:
            raise QuotaExhausted(f"{route} exhausted until reset", until)
        rem = self.remaining_today(route)
        if rem is not None and rem <= 0:
            raise QuotaExhausted(f"{route} daily limit reached", self.next_reset(route))
        lim = self.limits.get(route) or {}
        rpm, tpm = lim.get("rpm"), lim.get("tpm")
        deadline = time.time() + max_wait_s
        while True:
            with self._lock:
                now = time.time()
                q, t = self._req[route], self._tok[route]
                while q and now - q[0] >= 60:
                    q.popleft()
                while t and now - t[0][0] >= 60:
                    t.popleft()
                ok_r = rpm is None or len(q) < int(rpm)
                ok_t = tpm is None or sum(n for _, n in t) + est_tokens <= int(tpm) or not t
                if ok_r and ok_t:
                    q.append(now)
                    t.append((now, est_tokens))
                    return
                oldest = min(q[0] if q else now, t[0][0] if t else now)
                wait = max(0.5, 60 - (now - oldest))
            if time.time() + wait > deadline:
                raise QuotaExhausted(f"{route}: per-minute limit; next slot in {wait:.0f}s")
            time.sleep(min(wait, 5.0))

    def record(self, route: str, tokens: int) -> None:
        with self._lock:
            self._db.execute("INSERT INTO daily VALUES (?,?,1,?) ON CONFLICT(day, route) DO UPDATE SET "
                             "requests = requests + 1, tokens = tokens + excluded.tokens",
                             (self.day(route), route, int(tokens)))
            self._db.commit()

    def mark_exhausted(self, route: str, until: Optional[float] = None) -> None:
        until = until or self.next_reset(route)
        log.warning("%s out of daily quota until %s.", route, datetime.fromtimestamp(until).isoformat())
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO exhausted VALUES (?, ?)", (route, until))
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()
