"""CLI for evaluation rounds and reports.

    python -m scholargrid.evaluation round gaps --name gaps-2026-11 --topics "rag hallucination; code llms" --n 60
    python -m scholargrid.evaluation round extraction --name ext-2026-11 --n 25
    python -m scholargrid.evaluation round search --name search-2026-11 --queries 15
    python -m scholargrid.evaluation round reading --name read-2026-11 --n 10
    python -m scholargrid.evaluation round plans --name plans-2026-11 --from-gaps gaps-2026-11 --n 5
    python -m scholargrid.evaluation report --batch gaps-2026-11
    python -m scholargrid.evaluation compare-extract --routes gemini:gemini-3.8-flash,gemini:gemini-3.5-flash-lite --n 20
"""
from __future__ import annotations

import argparse
from typing import List, Optional

from ..config import Config, load_config


def _gap_round(cfg: Config, name: str, topics: List[str], n: int) -> None:
    from ..artifacts import load_bundle
    from ..gaps import gaps_for_workspace
    from ..llm import get_llm
    from ..workspace import build_workspace
    from .rounds import save_round
    from .sampling import make_gap_round

    b = load_bundle(cfg)
    llm = get_llm(cfg)
    version = str(b.meta.get("generated_at", "v1"))
    cards: list = []
    if topics:
        for t in topics:
            ws = build_workspace(t, b, None, cfg, bundle_version=version, llm=llm)
            data = gaps_for_workspace(ws, b, None, cfg, llm)
            for c in data.get("cards", []) + data.get("area_cards", []):
                cards.append({**c, "_topic": t})
            print(f"{t}: {len(data.get('cards', []))} cards (+{len(data.get('area_cards', []))} from areas)")
    else:
        cards = [c for cs in (b.gaps or {}).get("areas", {}).values() for c in cs]
    rnd = make_gap_round(name, cards, n, float(cfg["evaluation"]["overlap_share"]), int(cfg["evaluation"]["sample_seed"]),
                         context={"bundle_version": version, "topics": topics})
    print(f"round {name}: {len(rnd['items'])} items -> {save_round(cfg, rnd)}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="python -m scholargrid.evaluation", description="Evaluation rounds and reports")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("round")
    r.add_argument("kind", choices=["gaps", "extraction", "search", "reading", "plans"])
    r.add_argument("--name", required=True)
    r.add_argument("--n", type=int, default=20)
    r.add_argument("--topics", default="", help="';'-separated topics (gaps); empty = offline area cards")
    r.add_argument("--queries", type=int, default=15)
    r.add_argument("--from-gaps", default=None, help="plans: generate plans for cards of this gaps round")
    r.add_argument("--config", default=argparse.SUPPRESS)
    rep = sub.add_parser("report")
    rep.add_argument("--batch", required=True)
    rep.add_argument("--config", default=argparse.SUPPRESS)
    ce = sub.add_parser("compare-extract")
    ce.add_argument("--routes", required=True, help="comma-separated provider:model routes; the first is the reference")
    ce.add_argument("--n", type=int, default=20)
    ce.add_argument("--config", default=argparse.SUPPRESS)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    from ..persistence import get_repo
    from .rounds import load_round, save_round

    if args.cmd == "round":
        overlap = float(cfg["evaluation"]["overlap_share"])
        seed = int(cfg["evaluation"]["sample_seed"])
        if args.kind == "gaps":
            _gap_round(cfg, args.name, [t.strip() for t in args.topics.split(";") if t.strip()], args.n)
            return
        from ..artifacts import load_bundle
        from . import sampling

        b = load_bundle(cfg)
        if args.kind == "extraction":
            rnd = sampling.make_extraction_round(args.name, b, args.n, seed, overlap)
        elif args.kind == "search":
            rnd = sampling.make_search_round(args.name, cfg, b, args.queries, overlap=overlap, seed=seed)
        elif args.kind == "reading":
            rnd = sampling.make_reading_round(args.name, b, args.n, seed, overlap)
        else:
            from ..llm import get_llm
            from ..plan import generate_plan

            if not args.from_gaps:
                p.error("round plans needs --from-gaps <gaps round>")
            src = load_round(cfg, args.from_gaps)
            plans = [generate_plan(it["payload"], b, cfg, get_llm(cfg), b.embedder, weeks=10, background="",
                                   compute_access="One GPU", goal="First paper").model_dump(mode="json")
                     for it in src["items"][: args.n]]
            rnd = sampling.make_plan_round(args.name, plans, seed, overlap)
        print(f"round {args.name}: {len(rnd['items'])} items -> {save_round(cfg, rnd)}")
    elif args.cmd == "report":
        from .metrics import report

        repo = get_repo(cfg)
        ratings = repo.ratings(batch=args.batch) if repo is not None else []
        m = report(cfg, load_round(cfg, args.batch), ratings)
        print({k: v for k, v in m.items() if not isinstance(v, dict)})
    elif args.cmd == "compare-extract":
        from .compare import compare_extract

        out = compare_extract(cfg, [r.strip() for r in args.routes.split(",") if r.strip()], args.n)
        for row in out["routes"]:
            print(row)
        print(f"blind rating round: {out['round']}")


if __name__ == "__main__":
    main()
