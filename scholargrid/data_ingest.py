"""FR-01  Research dataset ingestion & cleaning.

Builds a reproducible corpus of recent arXiv computer-science papers. Three
sources are supported:

* ``arxiv_api``  - live harvest from export.arxiv.org (Atom API).
* ``kaggle``     - a locally downloaded arXiv metadata JSON-lines snapshot.
* ``synthetic``  - a deterministic, offline, topic-structured demo corpus.

Network sources degrade gracefully to ``synthetic`` so a run always completes.
Cleaning keeps cs.* papers inside the documented window, drops short/empty
abstracts and de-duplicates. The snapshot date and post-filter counts are
returned for the run metadata.
"""
from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Config
from .utils import get_logger

log = get_logger("ingest")

COLUMNS = ["arxiv_id", "title", "abstract", "categories", "primary_category", "date"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_corpus(cfg: Config) -> pd.DataFrame:
    d = cfg["data"]
    source = d["source"]
    log.info("Ingesting corpus (source=%s, target<=%s papers)", source, d["max_papers"])

    raw: pd.DataFrame
    if source == "arxiv_api":
        try:
            raw = _harvest_arxiv(cfg)
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("arXiv harvest failed (%s); falling back to synthetic.", exc)
            raw = _synthetic(cfg)
    elif source == "kaggle":
        raw = _load_kaggle(cfg)
    elif source == "synthetic":
        raw = _synthetic(cfg)
    else:
        raise ValueError(f"Unknown data.source: {source}")

    clean = _clean(raw, cfg)
    log.info("Corpus ready: %d papers after cleaning.", len(clean))
    return clean


# ---------------------------------------------------------------------------
# Cleaning (shared by all sources)
# ---------------------------------------------------------------------------
def _clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = cfg["data"]
    before = len(df)
    df = df.dropna(subset=["arxiv_id", "title", "abstract"]).copy()

    # Normalise whitespace.
    for col in ("title", "abstract"):
        df[col] = df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # Keep cs.* papers.
    prefix = d["categories_prefix"]
    df = df[df["categories"].astype(str).str.contains(prefix, na=False)]

    # Documented time window.
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["date"])
    start = pd.Timestamp(d["date_start"])
    end = pd.Timestamp(d["date_end"]) + pd.Timedelta(days=1)
    df = df[(df["date"] >= start) & (df["date"] < end)]

    # Drop short/empty abstracts.
    df = df[df["abstract"].str.len() >= int(d["min_abstract_chars"])]

    # De-duplicate.
    df["arxiv_id"] = df["arxiv_id"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["arxiv_id"])
    df = df.drop_duplicates(subset=["title"])

    df = df.sort_values("date").reset_index(drop=True)
    df["year_month"] = df["date"].dt.strftime("%Y-%m")
    log.info("Cleaning kept %d / %d records.", len(df), before)
    return df[COLUMNS + ["year_month"]]


# ---------------------------------------------------------------------------
# Source: arXiv API
# ---------------------------------------------------------------------------
_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _harvest_arxiv(cfg: Config) -> pd.DataFrame:
    import os

    d = cfg["data"]
    cache = os.path.join(cfg.raw_dir, "arxiv_harvest.csv")
    if d.get("use_cache", True) and os.path.exists(cache):
        df = pd.read_csv(cache, dtype=str)
        log.info("Loaded %d raw records from cache %s", len(df), cache)
        return df

    per_query = max(1, d["max_papers"] // max(1, len(d["arxiv_queries"])))
    rows: List[Dict] = []
    for query in d["arxiv_queries"]:
        rows.extend(_harvest_query(query, per_query))
    if not rows:
        raise RuntimeError("arXiv API returned no records")
    log.info("Harvested %d raw records from arXiv API.", len(rows))
    df = pd.DataFrame(rows)
    os.makedirs(cfg.raw_dir, exist_ok=True)
    df.to_csv(cache, index=False)
    log.info("Cached raw harvest to %s", cache)
    return df


def _harvest_query(query: str, want: int, page: int = 100) -> List[Dict]:
    out: List[Dict] = []
    start = 0
    while len(out) < want:
        params = {
            "search_query": query,
            "start": start,
            "max_results": min(page, want - len(out)),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as resp:
            feed = ET.fromstring(resp.read())
        entries = feed.findall("a:entry", _ATOM)
        if not entries:
            break
        for e in entries:
            out.append(_parse_entry(e))
        start += len(entries)
        time.sleep(3)  # arXiv API politeness
    log.info("  %-12s -> %d records", query, len(out))
    return out


def _parse_entry(e: ET.Element) -> Dict:
    def text(tag: str) -> str:
        node = e.find(tag, _ATOM)
        return (node.text or "").strip() if node is not None else ""

    raw_id = text("a:id")  # http://arxiv.org/abs/2501.01234v1
    arxiv_id = raw_id.rsplit("/", 1)[-1].split("v")[0]
    cats = [c.attrib.get("term", "") for c in e.findall("a:category", _ATOM)]
    primary = e.find("arxiv:primary_category", _ATOM)
    primary_cat = primary.attrib.get("term", cats[0] if cats else "") if primary is not None else (cats[0] if cats else "")
    return {
        "arxiv_id": arxiv_id,
        "title": text("a:title"),
        "abstract": text("a:summary"),
        "categories": " ".join([c for c in cats if c]),
        "primary_category": primary_cat,
        "date": text("a:published"),
    }


# ---------------------------------------------------------------------------
# Source: Kaggle snapshot (arxiv-metadata-oai-snapshot.json, JSON lines)
# ---------------------------------------------------------------------------
def _load_kaggle(cfg: Config) -> pd.DataFrame:
    import json

    d = cfg["data"]
    path = cfg.abspath(d["kaggle_json"])
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if len(rows) >= d["max_papers"] * 4:  # over-read; cleaning trims
                break
            rec = json.loads(line)
            cats = rec.get("categories", "")
            if cfg["data"]["categories_prefix"] not in cats:
                continue
            versions = rec.get("versions", [])
            date = versions[0].get("created") if versions else rec.get("update_date", "")
            rows.append({
                "arxiv_id": rec.get("id", ""),
                "title": rec.get("title", ""),
                "abstract": rec.get("abstract", ""),
                "categories": cats,
                "primary_category": cats.split(" ")[0] if cats else "",
                "date": date,
            })
    log.info("Loaded %d candidate records from Kaggle snapshot.", len(rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Source: synthetic (deterministic, offline)
# ---------------------------------------------------------------------------
_TOPICS = {
    "cs.CL": {
        "label": "language models",
        "terms": ["language model", "transformer", "attention", "tokenization", "prompt",
                  "instruction tuning", "machine translation", "text generation", "llm",
                  "sequence", "reasoning", "fine-tuning", "embedding", "dialogue"],
    },
    "cs.CV": {
        "label": "computer vision",
        "terms": ["image classification", "object detection", "segmentation", "diffusion model",
                  "convolutional", "vision transformer", "image generation", "visual",
                  "depth estimation", "optical flow", "pose", "video understanding"],
    },
    "cs.LG": {
        "label": "machine learning theory",
        "terms": ["gradient descent", "generalization", "optimization", "regularization",
                  "representation learning", "contrastive", "self-supervised", "overfitting",
                  "neural network", "convergence", "sample complexity", "loss landscape"],
    },
    "cs.CR": {
        "label": "security and privacy",
        "terms": ["adversarial example", "differential privacy", "encryption", "malware",
                  "vulnerability", "membership inference", "federated learning", "attack",
                  "defense", "robustness", "backdoor", "cryptographic protocol"],
    },
    "cs.RO": {
        "label": "robotics and control",
        "terms": ["reinforcement learning", "manipulation", "motion planning", "policy",
                  "sim-to-real", "locomotion", "control", "grasping", "navigation",
                  "imitation learning", "trajectory", "sensorimotor"],
    },
    "cs.DC": {
        "label": "distributed systems",
        "terms": ["consensus", "fault tolerance", "distributed training", "scheduling",
                  "parallel", "communication overhead", "scalability", "throughput",
                  "replication", "load balancing", "serverless", "cluster"],
    },
}


def _synthetic(cfg: Config) -> pd.DataFrame:
    """Create a deterministic corpus with planted topic structure and a mild
    temporal growth trend, so every pipeline stage has meaningful signal
    offline. Not scientific data - a reproducible stand-in for CI/demo."""
    d = cfg["data"]
    rng = np.random.default_rng(cfg["seed"])
    n = d["max_papers"]
    start = pd.Timestamp(d["date_start"])
    end = pd.Timestamp(d["date_end"])
    span_days = (end - start).days
    topics = list(_TOPICS.items())

    # Growth: later topics in the list get more recent-weighted dates.
    rows: List[Dict] = []
    for i in range(n):
        cat, spec = topics[i % len(topics)]
        topic_idx = i % len(topics)
        # Skew dates: higher-index topics skew later (simulated growth).
        skew = 0.5 + 0.9 * (topic_idx / max(1, len(topics) - 1))
        frac = rng.beta(2.0, 2.0 / skew)
        day = int(frac * span_days)
        date = start + pd.Timedelta(days=day)
        terms = list(rng.choice(spec["terms"], size=rng.integers(5, 9), replace=False))
        title = f"{terms[0].title()}: {terms[1]} for {spec['label']}"
        abstract = (
            f"We study {terms[0]} in the context of {spec['label']}. "
            f"Our approach leverages {terms[1]} and {terms[2]} to improve {terms[3]}. "
            f"Experiments on standard benchmarks demonstrate that {terms[0]} combined with "
            f"{terms[4 % len(terms)]} yields consistent gains. We further analyse "
            f"{', '.join(terms[1:4])} and discuss implications for {spec['label']} research. "
            f"This work connects {terms[0]} to broader questions of {terms[-1]}."
        )
        rows.append({
            "arxiv_id": f"synth.{i:05d}",
            "title": title,
            "abstract": abstract,
            "categories": f"{cat} cs.AI",
            "primary_category": cat,
            "date": date.isoformat(),
        })
    log.info("Generated %d synthetic records across %d topics.", n, len(topics))
    return pd.DataFrame(rows)


def snapshot_info(df: pd.DataFrame, cfg: Config) -> Dict:
    return {
        "source": cfg["data"]["source"],
        "n_papers": int(len(df)),
        "date_window": [cfg["data"]["date_start"], cfg["data"]["date_end"]],
        "latest_included_date": df["date"].max().strftime("%Y-%m-%d") if len(df) else None,
        "earliest_included_date": df["date"].min().strftime("%Y-%m-%d") if len(df) else None,
        "category_counts": df["primary_category"].value_counts().head(20).to_dict(),
        "harvested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
