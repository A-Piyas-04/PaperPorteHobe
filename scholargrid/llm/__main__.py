"""CLI for the free-tier LLM layer.

    python -m scholargrid.llm check      # one tiny call per enabled provider/route of task 'check'
    python -m scholargrid.llm quota      # per route: used today / rpd, exhausted-until, next reset
    python -m scholargrid.llm usage [--month 2026-10]   # requests and tokens by task/route
    python -m scholargrid.llm estimate --task extract --n 12000   # requests and DAYS needed
"""
from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel

from ..config import Config, load_config

PRIVACY_NOTE = "Note: free-tier content may be used by the provider to improve its products."


class _Check(BaseModel):
    ok: bool
    echo: str


def _check(cfg: Config) -> None:
    from . import build_router
    from .errors import LLMError, LLMUnavailable, QuotaExhausted
    from .prompts import render

    print(PRIVACY_NOTE)
    router = build_router(cfg)
    lc = cfg["llm"]
    for name, p in lc["providers"].items():
        if not p["enabled"]:
            print(f"{name:12s} disabled")
        elif name not in router.backends:
            print(f"{name:12s} unavailable (set {p['api_key_env'] or 'an API key'})")
    routes = list(lc["tasks"]["check"]["route"])
    for r in routes:
        provider = r.split(":", 1)[0]
        if provider not in router.backends:
            print(f"{r:45s} unavailable")
            continue
        t0 = time.time()
        try:
            system, user = render("check", word="scholargrid")
            res = router._run("check", "check", _Check, system, user, only_route=r)
            status = "OK" + (" (cached)" if res.cached else "") if res.value.ok else "unexpected answer"
        except QuotaExhausted as exc:
            status = f"quota: {exc}"
        except LLMUnavailable as exc:
            status = f"unavailable: {exc}"
        except LLMError as exc:
            status = f"error: {exc}"
        print(f"{r:45s} {status}  ({time.time() - t0:.1f}s)")


def _fmt_ts(ts: Optional[float]) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "-"


def _quota(cfg: Config) -> None:
    from .quota import QuotaManager

    lc = cfg["llm"]
    q = QuotaManager(cfg.abspath(lc["usage_dir"]), lc["limits"],
                     {n: p["quota_tz"] for n, p in lc["providers"].items()})
    routes = sorted({r for t in lc["tasks"].values() for r in t["route"]} | set(q.routes_seen()))
    print(f"{'route':45s} {'used':>6s} {'rpd':>6s} {'left':>6s}  {'exhausted until':16s}  next reset (local)")
    for r in routes:
        rpd = (lc["limits"].get(r) or {}).get("rpd")
        left = q.remaining_today(r)
        print(f"{r:45s} {q.used_today(r):6d} {rpd if rpd else '-':>6} {left if left is not None else '-':>6}  "
              f"{_fmt_ts(q.exhausted_until(r)):16s}  {_fmt_ts(q.next_reset(r))}")


def _usage(cfg: Config, month: Optional[str]) -> None:
    from .usage import read_usage

    rows = read_usage(cfg.abspath(cfg["llm"]["usage_dir"]), month)
    if not rows:
        print("No LLM usage recorded for this month.")
        return
    agg: Dict[tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        a = agg[(row.get("task", "?"), row.get("route", "?"))]
        a["requests"] += 1
        for k, v in (row.get("usage") or {}).items():
            a[k] += float(v or 0)
        a["cost_usd"] += float(row.get("cost_usd") or 0)
    print(f"{'task':16s} {'route':45s} {'req':>6s} {'in tok':>10s} {'out tok':>10s} {'think':>8s} {'cost $':>8s}")
    for (task, route), a in sorted(agg.items()):
        print(f"{task:16s} {route:45s} {int(a['requests']):6d} {int(a['input_tokens']):10d} "
              f"{int(a['output_tokens']):10d} {int(a['thinking_tokens']):8d} {a['cost_usd']:8.4f}")


def estimate_lines(cfg: Config, task: str, n: int) -> List[str]:
    lc = cfg["llm"]
    spec = lc["tasks"][task]
    pack = int(spec["pack_size"])
    requests = math.ceil(n / pack)
    lines = [f"{task}: {n} items, pack_size={pack} -> {requests} requests"]
    total_rpd = 0
    for r in spec["route"]:
        rpd = (lc["limits"].get(r) or {}).get("rpd")
        if rpd:
            total_rpd += int(rpd)
            lines.append(f"  route {r}  rpd={rpd}  -> {requests / int(rpd):.1f} days")
        else:
            lines.append(f"  route {r}  rpd unknown (copy it into llm.limits)")
    if len(spec["route"]) > 1 and total_rpd:
        lines.append(f"  all routes combined: rpd={total_rpd} -> {requests / total_rpd:.1f} days")
    extra = [r for r in lc["limits"] if r not in spec["route"] and r.split(":", 1)[0] != "gemini"]
    for r in extra[:1]:
        rpd = lc["limits"][r].get("rpd")
        if rpd and total_rpd:
            lines.append(f"  (add {r} rpd={rpd} to the route -> ~{requests / (total_rpd + int(rpd)):.1f} days combined)")
    lines.append("Tip: narrow `scope` or raise pack_size (watch output truncation) to finish sooner.")
    return lines


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="python -m scholargrid.llm", description="Free-tier LLM layer tools")
    p.add_argument("--config", default=None, help="path to a config YAML")
    sub = p.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("check", help="one tiny call per route of task 'check'")
    sc.add_argument("--config", default=argparse.SUPPRESS)
    sq = sub.add_parser("quota", help="quota used today per route")
    sq.add_argument("--config", default=argparse.SUPPRESS)
    su = sub.add_parser("usage", help="requests and tokens by task and route")
    su.add_argument("--month", default=None, help="YYYY-MM (default: this month)")
    su.add_argument("--config", default=argparse.SUPPRESS)
    se = sub.add_parser("estimate", help="requests and days a job needs with the current limits")
    se.add_argument("--task", required=True)
    se.add_argument("--n", type=int, required=True)
    se.add_argument("--config", default=argparse.SUPPRESS)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "check":
        _check(cfg)
    elif args.cmd == "quota":
        _quota(cfg)
    elif args.cmd == "usage":
        _usage(cfg, args.month)
    elif args.cmd == "estimate":
        if args.task not in cfg["llm"]["tasks"]:
            p.error(f"unknown task {args.task!r}; known: {sorted(cfg['llm']['tasks'])}")
        print("\n".join(estimate_lines(cfg, args.task, args.n)))


if __name__ == "__main__":
    main()
