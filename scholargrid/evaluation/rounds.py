"""Frozen evaluation rounds: created once, stored as JSON, rated in a per-rater random order.

The full item JSON is stored in the round file, so later code changes never change what was rated.
Each rater gets the overlap items (rated by everyone, for agreement) plus the non-overlap items nobody
else has rated yet.
"""
from __future__ import annotations

import glob
import hashlib
import os
import random
import time
from typing import Dict, List, Optional

from ..config import Config
from ..utils import ensure_dir, load_json, save_json


def round_dir(cfg: Config) -> str:
    return ensure_dir(cfg.abspath(cfg["evaluation"]["report_dir"]))


def round_path(cfg: Config, name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    return os.path.join(round_dir(cfg), f"{safe}.json")


def new_round(name: str, kind: str, items: List[Dict], meta: Optional[Dict] = None) -> Dict:
    return {"name": name, "kind": kind, "created_at": time.time(), "meta": dict(meta or {}), "items": items}


def save_round(cfg: Config, rnd: Dict) -> str:
    path = round_path(cfg, rnd["name"])
    save_json(rnd, path)
    return path


def load_round(cfg: Config, name: str) -> Dict:
    return load_json(round_path(cfg, name))


def list_rounds(cfg: Config) -> List[str]:
    out = []
    for p in sorted(glob.glob(os.path.join(round_dir(cfg), "*.json"))):
        try:
            d = load_json(p)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and "items" in d and "kind" in d:
            out.append(str(d["name"]))
    return out


def rater_queue(rnd: Dict, rater: str, rated_by: Dict[str, set]) -> List[Dict]:
    """Items left for ``rater``: overlap items + non-overlap items nobody else has rated, shuffled per rater.
    ``rated_by`` maps item_id -> set of raters who already rated it."""
    seed = int(hashlib.sha1(f"{rnd['name']}|{rater}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    mine = [it for it in rnd["items"] if rater not in rated_by.get(it["item_id"], set())]
    overlap = [it for it in mine if it.get("overlap")]
    rest = [it for it in mine if not it.get("overlap") and not rated_by.get(it["item_id"])]
    rng.shuffle(overlap)
    rng.shuffle(rest)
    return overlap + rest
