"""Shared helpers: logging, seeding, JSON IO, cosine math."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_ROOT = "scholargrid"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _root_logger() -> logging.Logger:
    root = logging.getLogger(_ROOT)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return root


def get_logger(name: str) -> logging.Logger:
    _root_logger()
    return logging.getLogger(f"{_ROOT}.{name}")


def configure_logging(json_logs: bool = False) -> None:
    root = _root_logger()
    fmt: logging.Formatter = (JsonFormatter() if json_logs
                              else logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
    for handler in root.handlers:
        handler.setFormatter(fmt)


@contextmanager
def stage_timer(name: str, timings: Dict[str, Dict], **fields: Any) -> Iterator[Dict]:
    """Time a pipeline stage and log a structured completion record."""
    log = get_logger("pipeline")
    info: Dict[str, Any] = dict(fields)
    t0 = time.time()
    log.info("stage %s: start", name, extra={"fields": {"stage": name, "event": "start"}})
    yield info
    info["seconds"] = round(time.time() - t0, 2)
    timings[name] = info
    log.info("stage %s: done in %.1fs %s", name, info["seconds"],
             {k: v for k, v in info.items() if k != "seconds"},
             extra={"fields": {"stage": name, "event": "done", **info}})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of the top-k rows in ``matrix`` by cosine
    similarity. Both inputs are assumed L2-normalised, so cosine == dot."""
    sims = np.asarray(matrix @ query_vec, dtype=np.float32)
    k = min(k, sims.shape[0])
    if k <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]
