"""Drive the real Streamlit UI the way a person does, and judge what they would see.

Uses Streamlit's own AppTest harness, so these are the actual view functions, the actual
widgets and the actual button callbacks — not the library calls underneath them. The checks
are about comprehension, not HTTP status: does a card say what the gap IS, why it matters,
what to do first, and which papers back it up?

    python scripts/ui_e2e.py --topic "federated learning"
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app"))

from streamlit.testing.v1 import AppTest

GAPS_SCRIPT = '''
import sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {app!r})
import streamlit as st
st.session_state["topic_query"] = {query!r}
from views import gaps
gaps.render()
'''

PLAN_SCRIPT = '''
import sys, json
sys.path.insert(0, {root!r})
sys.path.insert(0, {app!r})
import streamlit as st
st.session_state["topic_query"] = {query!r}
st.session_state["plan_gap"] = json.loads({card!r})
from views import plan
plan.render()
'''


def texts(at) -> list:
    """Everything the page rendered as words, in order."""
    out = []
    for kind in ("markdown", "title", "header", "subheader", "caption", "text", "info",
                 "warning", "error", "success"):
        try:
            for el in getattr(at, kind):
                v = getattr(el, "value", None)
                if isinstance(v, str):
                    out.append(v)
        except Exception:
            pass
    return out


def show(label: str, at) -> list:
    body = texts(at)
    print(f"\n--- {label}: {len(body)} text blocks, {len(at.button)} buttons ---")
    for t in body[:8]:
        flat = " ".join(t.split())
        print("   |", flat[:150])
    if at.exception:
        print("   !! EXCEPTIONS:")
        for e in at.exception:
            print("     ", str(e.value)[:300])
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="federated learning")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    failures = []

    print(f"=== END-USER RUN: topic {args.topic!r} ===")

    # --- Step 1: the user opens "Open problems" for their topic -------------------
    at = AppTest.from_string(GAPS_SCRIPT.format(root=ROOT, app=os.path.join(ROOT, "app"),
                                                query=args.topic))
    at.timeout = args.timeout
    at.run(timeout=args.timeout)
    body = show("Open problems (first view)", at)
    if at.exception:
        failures.append("gaps page raised an exception on load")

    btn = [b for b in at.button if "find open problems" in (b.label or "").lower()]
    if not btn:
        print("   (no 'Find open problems' button — cards may already be cached)")
    else:
        print(f"\n>>> clicking {btn[0].label!r} (this is the 1-2 minute wait)")
        btn[0].click().run(timeout=args.timeout)
        body = show("Open problems (after clicking)", at)
        if at.exception:
            failures.append("clicking 'Find open problems' raised an exception")

    joined = " ".join(body).lower()
    empty_signals = ["few author-stated limitations", "no candidate gaps", "no extracted open problems"]
    hit = [s for s in empty_signals if s in joined]
    if hit:
        print(f"\n   USER SEES AN EMPTY STATE: {hit[0]!r}")
        failures.append(f"user got no gap cards: {hit[0]}")

    # --- Step 2: can the user actually understand a card? -------------------------
    from scholargrid.gaps.workspace import load_cached_gaps
    from scholargrid.config import load_config

    cfg = load_config()
    import ui as _ui  # noqa
    cards = []
    try:
        ws = _ui.get_workspace(args.topic)
        data = load_cached_gaps(ws.key, cfg) or {}
        cards = data.get("cards") or []
    except Exception as exc:
        print("   (could not read cached cards:", str(exc)[:120], ")")

    print(f"\n=== CARD COMPREHENSION: {len(cards)} card(s) ===")
    if not cards:
        failures.append("no gap cards for the user to read")
    for i, c in enumerate(cards[:3], 1):
        title = c.get("title", "")
        stmt = c.get("statement", "")
        matters = c.get("why_it_matters", "")
        steps = c.get("first_steps") or []
        ev = c.get("evidence") or []
        quotes = [e for e in ev if e.get("quote")]
        eids = [e.get("eid") for e in ev if e.get("eid")]
        print(f"\n  Card {i}: {title[:90]!r}")
        print(f"    statement      : {str(stmt)[:120]!r}")
        print(f"    why it matters : {str(matters)[:120]!r}")
        print(f"    first steps    : {len(steps)}")
        print(f"    evidence       : {len(ev)} ({len(quotes)} quoted, {len(eids)} labelled {eids[:4]})")
        gb = c.get("generated_by") or {}
        print(f"    written by     : {gb.get('model','?')} synthesized={gb.get('synthesized')}")
        if not title or not stmt:
            failures.append(f"card {i} has no title or statement")
        if not matters:
            failures.append(f"card {i} does not say why it matters")
        if not steps:
            failures.append(f"card {i} gives the user no first steps")
        if not ev:
            failures.append(f"card {i} shows no evidence")

    # --- Step 3: the user makes a plan from the best card -------------------------
    if cards:
        import json as _json
        print("\n=== PLAN FROM THE TOP CARD ===")
        at2 = AppTest.from_string(PLAN_SCRIPT.format(root=ROOT, app=os.path.join(ROOT, "app"),
                                                     query=args.topic,
                                                     card=_json.dumps(cards[0])))
        at2.timeout = args.timeout
        at2.run(timeout=args.timeout)
        show("Plan page", at2)
        subs = [b for b in at2.button if "starter plan" in (b.label or "").lower()]
        forms = list(at2.get("form_submit_button"))
        target = subs or forms
        if not target:
            print("   !! no 'Create my starter plan' control found")
            failures.append("user cannot reach the plan button")
        else:
            print(f"\n>>> clicking {target[0].label!r}")
            target[0].click().run(timeout=args.timeout)
            pbody = show("Plan (after clicking)", at2)
            if at2.exception:
                failures.append("generating a plan raised an exception")
            if not pbody:
                failures.append("plan page rendered nothing")

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} PROBLEM(S) A USER WOULD HIT:")
        for f in failures:
            print("  -", f)
        return 1
    print("END-USER FLOW OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
