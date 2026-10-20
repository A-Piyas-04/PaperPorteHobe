"""Usage ledger: one JSON line per call (route, task, tokens, cost if a price is configured)."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import UTC, datetime
from typing import Any, Callable, Dict, List, Optional

from ..utils import ensure_dir
from .errors import BudgetExceeded


class UsageLedger:
    def __init__(self, usage_dir: str, prices: Dict[str, Dict], max_requests: Optional[int]):
        self.dir = ensure_dir(usage_dir)
        self.prices = prices
        self.max_requests = max_requests          # per process run (offline); None = unlimited
        self.requests = 0
        self.cost_usd = 0.0
        self._lock = threading.Lock()
        self.extra_guard: Optional[Callable[[str], None]] = None   # plan 11: may raise BudgetExceeded
        self.on_record: Optional[Callable[[str, str], None]] = None  # plan 11: (task, route)

    def check(self, task: str) -> None:
        with self._lock:
            if self.max_requests is not None and self.requests >= self.max_requests:
                raise BudgetExceeded(f"{task}: llm.budget.offline_max_requests={self.max_requests} reached")
        if self.extra_guard is not None:
            self.extra_guard(task)

    def record(self, task: str, route: str, usage: Dict[str, int], context: Optional[Dict] = None) -> float:
        p = self.prices.get(route) or {}
        cost = (usage.get("input_tokens", 0) * float(p.get("input", 0))
                + (usage.get("output_tokens", 0) + usage.get("thinking_tokens", 0)) * float(p.get("output", 0))) / 1e6
        with self._lock:
            self.requests += 1
            self.cost_usd += cost
            line = {"ts": time.time(), "task": task, "route": route, "usage": usage,
                    "cost_usd": round(cost, 6), **(context or {})}
            with open(os.path.join(self.dir, datetime.now(UTC).strftime("%Y-%m") + ".jsonl"), "a",
                      encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
        if self.on_record is not None:
            self.on_record(task, route)
        return cost


def read_usage(usage_dir: str, month: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ledger lines for ``month`` (``YYYY-MM``; default: current UTC month)."""
    month = month or datetime.now(UTC).strftime("%Y-%m")
    path = os.path.join(usage_dir, f"{month}.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out
