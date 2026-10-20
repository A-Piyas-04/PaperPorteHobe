"""CLI for the full-text store.

    python -m scholargrid.fulltext stats                    # status counts by source / kind coverage
    python -m scholargrid.fulltext show 2401.01234          # sections (kind, heading, chars) + cue sentences
    python -m scholargrid.fulltext fetch 2401.01234 ...     # fetch now (debug), respects rate limit
    python -m scholargrid.fulltext compact                  # merge part files
"""
from __future__ import annotations

import argparse
from typing import List, Optional

from ..config import Config, load_config
from .store import FulltextStore


def _stats(cfg: Config) -> None:
    store = FulltextStore(cfg)
    st = store.status
    total = len(st)
    print(f"store: {store.dir}")
    print(f"papers: {total}")
    if not total:
        return
    for status, n in st["status"].value_counts().items():
        print(f"  {status:12s} {n:6d}  ({n / total:.0%})")
    ok = st[st["status"] == "ok"]
    print("ok by source:")
    for src, n in ok["source"].value_counts().items():
        print(f"  {src:12s} {n:6d}")
    secs = store.sections(ok["arxiv_id"].astype(str))
    if len(ok) and not secs.empty:
        kinds_per_paper = secs.groupby("arxiv_id")["kind"].agg(set)
        for kind in ("limitations", "future_work", "conclusion"):
            share = kinds_per_paper.apply(lambda ks, k=kind: k in ks).mean()
            print(f"  ok papers with a {kind:12s} section: {share:.0%}")
        either = kinds_per_paper.apply(lambda ks: bool({"limitations", "future_work"} & ks)).mean()
        print(f"  ok papers with limitations OR future_work: {either:.0%}")
    cues = store.cues(ok["arxiv_id"].astype(str))
    print(f"cue sentences: {len(cues)}")


def _show(cfg: Config, arxiv_id: str) -> None:
    store = FulltextStore(cfg)
    st = store.status[store.status["arxiv_id"] == arxiv_id] if len(store.status) else store.status
    if st.empty:
        print(f"{arxiv_id}: not in the store")
        return
    row = st.iloc[-1]
    print(f"{arxiv_id}: {row['status']} via {row['source'] or '-'} {row['error'] or ''}".rstrip())
    for _, s in store.sections([arxiv_id]).sort_values("idx").iterrows():
        print(f"  [{int(s['idx']):2d}] {s['kind']:20s} {int(s['n_chars']):6d}  {s['heading']}")
    cues = store.cues([arxiv_id])
    if not cues.empty:
        print("cues:")
        for _, c in cues.iterrows():
            print(f"  ({c['cue']}, {c['kind']}) {c['sentence'][:200]}")


def _fetch(cfg: Config, ids: List[str]) -> None:
    from .fetch import Fetcher
    from .sections import find_cues

    store = FulltextStore(cfg)
    fetcher = Fetcher(cfg)
    try:
        for aid in ids:
            res = fetcher.fetch(aid)
            store.add(aid, res.status, res.source, res.sections, find_cues(res.sections), res.http_status, res.error)
            print(f"{aid}: {res.status} {res.source} {len(res.sections)} sections {res.error}".rstrip())
    finally:
        store.flush()


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="python -m scholargrid.fulltext", description="Full-text store tools")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("stats", "compact"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=argparse.SUPPRESS)
    sh = sub.add_parser("show")
    sh.add_argument("arxiv_id")
    sh.add_argument("--config", default=argparse.SUPPRESS)
    sf = sub.add_parser("fetch")
    sf.add_argument("arxiv_ids", nargs="+")
    sf.add_argument("--config", default=argparse.SUPPRESS)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "stats":
        _stats(cfg)
    elif args.cmd == "show":
        _show(cfg, args.arxiv_id)
    elif args.cmd == "fetch":
        _fetch(cfg, args.arxiv_ids)
    elif args.cmd == "compact":
        FulltextStore(cfg).compact()
        print("compacted")


if __name__ == "__main__":
    main()
