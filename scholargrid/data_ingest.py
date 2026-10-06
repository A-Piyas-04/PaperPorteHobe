"""FR-01  Research dataset ingestion & cleaning.

Builds a reproducible corpus of arXiv computer-science papers. Sources:

* ``kaggle``     - bulk backfill from the arXiv metadata snapshot (JSON lines).
                   The whole file is streamed and filtered by category and date
                   while reading, so the configured window is actually covered.
* ``oai_pmh``    - official date-ranged bulk harvest; incremental and resumable.
* ``arxiv_api``  - the search API (newest papers only). Ad hoc demos.
* ``synthetic``  - deterministic, offline, topic-structured demo corpus.

Bulk sources write into the Parquet/DuckDB store (:mod:`scholargrid.storage`)
and the corpus is read back from it, so repeated refreshes merge instead of
starting over. A failed harvest raises unless ``data.allow_synthetic_fallback``
is true; production configs forbid that fallback.

Cleaning: LaTeX/unicode normalisation, per-token category matching, documented
date window on first submission, short-abstract and non-English filtering, and
de-duplication by base arXiv ID and normalised title (latest version wins).
Large corpora are sampled per (primary_category, year_month) stratum with
weights recorded for growth reweighting.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .text import arxiv_version, base_arxiv_id, has_category, is_english, normalize_text, title_key
from .utils import get_logger, load_json, save_json

log = get_logger("ingest")

COLUMNS = ["arxiv_id", "title", "abstract", "authors", "categories", "primary_category",
           "date", "first_submitted", "last_updated", "doi", "journal_ref", "license",
           "year_month", "sample_weight"]
_DEFAULT_CAP = 1500


class IngestError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_corpus(cfg: Config) -> pd.DataFrame:
    d = cfg["data"]
    source = d["source"]
    log.info("Ingesting corpus (source=%s, profile=%s, cap=%s, window=%s..%s)",
             source, d["profile"], cfg.max_papers, d["date_start"], d["date_end"])
    try:
        if source in ("kaggle", "oai_pmh"):
            clean = _from_store(cfg, source)
        elif source == "arxiv_api":
            clean = clean_records(_harvest_arxiv(cfg), cfg)
        elif source == "synthetic":
            clean = clean_records(_synthetic(cfg), cfg)
        else:  # pragma: no cover - rejected by the schema
            raise ValueError(f"Unknown data.source: {source}")
        if clean.empty:
            raise IngestError(f"Source '{source}' produced no papers inside the window.")
    except Exception as exc:
        if source == "synthetic" or not d["allow_synthetic_fallback"]:
            raise IngestError(f"Ingestion from '{source}' failed: {exc}") from exc
        log.warning("Ingestion from %s failed (%s); falling back to SYNTHETIC data "
                    "(data.allow_synthetic_fallback is true).", source, exc)
        clean = clean_records(_synthetic(cfg), cfg)
        clean.attrs["fallback_from"] = source

    available = len(clean)
    corpus = sample_corpus(clean, cfg)
    corpus.attrs["available"] = available
    corpus.attrs["fallback_from"] = clean.attrs.get("fallback_from")
    log.info("Corpus ready: %d papers (%d available after cleaning).", len(corpus), available)
    return corpus


# ---------------------------------------------------------------------------
# Cleaning (shared by all sources)
# ---------------------------------------------------------------------------
def clean_records(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = cfg["data"]
    before = len(df)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = df.copy()
    if "first_submitted" not in df.columns:
        df["first_submitted"] = df.get("date")
    if "last_updated" not in df.columns:
        df["last_updated"] = df["first_submitted"]
    for col in ("authors", "categories", "primary_category", "doi", "journal_ref", "license"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    df = df.dropna(subset=["arxiv_id", "title", "abstract"])

    df["version"] = df["arxiv_id"].map(arxiv_version)
    df["arxiv_id"] = df["arxiv_id"].map(base_arxiv_id)
    df["title"] = df["title"].map(normalize_text)
    df["abstract"] = df["abstract"].map(normalize_text)

    whitelist = cfg.category_list()
    df = df[df["categories"].map(lambda c: has_category(c, d["categories_prefix"], whitelist))]
    empty_primary = df["primary_category"].str.strip() == ""
    df.loc[empty_primary, "primary_category"] = df.loc[empty_primary, "categories"].str.split().str[0]

    for col in ("first_submitted", "last_updated"):
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
    df["last_updated"] = df["last_updated"].fillna(df["first_submitted"])
    df = df.dropna(subset=["first_submitted"])
    start = pd.Timestamp(d["date_start"])
    end = pd.Timestamp(d["date_end"]) + pd.Timedelta(days=1)
    df = df[(df["first_submitted"] >= start) & (df["first_submitted"] < end)]

    df = df[df["abstract"].str.len() >= int(d["min_abstract_chars"])]
    if d["language_filter"]:
        df = df[df["abstract"].map(is_english)]

    df = df.sort_values(["version", "last_updated"])
    df = df.drop_duplicates(subset=["arxiv_id"], keep="last")
    df = df[~df["title"].map(title_key).duplicated(keep="last")]

    df["date"] = df["first_submitted"]
    df["year_month"] = df["date"].dt.strftime("%Y-%m")
    if "sample_weight" not in df.columns:
        df["sample_weight"] = 1.0
    df = df.sort_values("date").reset_index(drop=True)
    log.info("Cleaning kept %d / %d records.", len(df), before)
    return df[COLUMNS]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def _water_fill(sizes: np.ndarray, budget: int) -> np.ndarray:
    """Per-stratum quota: every stratum gets min(size, q) with q chosen so the
    total is about ``budget``. Small strata are kept whole; big ones capped."""
    lo, hi = 0.0, float(sizes.max())
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.minimum(sizes, mid).sum() > budget:
            hi = mid
        else:
            lo = mid
    take = np.minimum(sizes, np.floor(lo)).astype(int)
    leftover = budget - int(take.sum())
    order = np.argsort(-(sizes - take))
    for i in order[:max(0, leftover)]:
        if take[i] < sizes[i]:
            take[i] += 1
    return take


def sample_corpus(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cap = cfg.max_papers
    mode = cfg["data"]["sampling"]
    out = df.copy()
    out["sample_weight"] = 1.0
    if cap is None or len(out) <= cap or mode == "none":
        return out.reset_index(drop=True)
    if mode == "newest":
        return out.sort_values("date").tail(cap).reset_index(drop=True)

    rng = np.random.default_rng(cfg["seed"])
    groups = out.groupby(["primary_category", "year_month"], sort=True)
    keys = list(groups.groups.keys())
    sizes = np.array([len(groups.groups[k]) for k in keys])
    take = _water_fill(sizes, cap)
    parts = []
    for key, size, n in zip(keys, sizes, take):
        if n <= 0:
            continue
        idx = groups.groups[key]
        chosen = rng.choice(np.asarray(idx), size=int(n), replace=False)
        part = out.loc[chosen].copy()
        part["sample_weight"] = float(size) / float(n)
        parts.append(part)
    sampled = pd.concat(parts).sort_values("date").reset_index(drop=True)
    log.info("Stratified sample: %d of %d papers across %d strata.", len(sampled), len(out), len(keys))
    return sampled


# ---------------------------------------------------------------------------
# Bulk sources via the store
# ---------------------------------------------------------------------------
def _from_store(cfg: Config, source: str) -> pd.DataFrame:
    from .storage import PaperStore

    store = PaperStore(cfg)
    if source == "kaggle":
        _ingest_kaggle(cfg, store)
    else:
        _ingest_oai(cfg, store)
    d = cfg["data"]
    stored = store.read(d["date_start"], d["date_end"])
    if stored.empty:
        return pd.DataFrame(columns=COLUMNS)
    # Stored rows were cleaned on the way in; re-cleaning applies the current
    # window/category settings and catches title duplicates across harvests.
    return clean_records(stored, cfg)


def _kaggle_state_matches(cfg: Config, path: str, state_path: str) -> bool:
    if not os.path.exists(state_path):
        return False
    st = os.stat(path)
    state = load_json(state_path)
    d = cfg["data"]
    return (state.get("size") == st.st_size and state.get("mtime") == int(st.st_mtime)
            and state.get("date_start") <= d["date_start"]
            and state.get("date_end") >= d["date_end"]
            and state.get("categories") == d["categories"])


def _ingest_kaggle(cfg: Config, store, chunk_rows: int = 50000) -> None:
    d = cfg["data"]
    path = cfg.abspath(d["kaggle_json"])
    if not os.path.exists(path):
        raise IngestError(f"Kaggle snapshot not found at {path}. Download "
                          "'arxiv-metadata-oai-snapshot.json' from kaggle.com/datasets/Cornell-University/arxiv.")
    state_path = os.path.join(cfg.raw_dir, "kaggle_state.json")
    if d["use_cache"] and _kaggle_state_matches(cfg, path, state_path):
        log.info("Kaggle snapshot unchanged since last ingest; using the stored corpus.")
        return
    total = 0
    for chunk in iter_kaggle(path, cfg, chunk_rows):
        total += store.upsert(clean_records(chunk, cfg).assign(source="kaggle"))
    st = os.stat(path)
    save_json({"size": st.st_size, "mtime": int(st.st_mtime), "date_start": d["date_start"],
               "date_end": d["date_end"], "categories": d["categories"],
               "ingested": total, "at": datetime.now(timezone.utc).isoformat()}, state_path)
    log.info("Kaggle ingest stored %d papers.", total)


def iter_kaggle(path: str, cfg: Config, chunk_rows: int = 50000) -> Iterator[pd.DataFrame]:
    """Stream the whole snapshot, keeping matching categories inside the window."""
    d = cfg["data"]
    prefix, whitelist = d["categories_prefix"], cfg.category_list()
    start, end = d["date_start"], d["date_end"]
    rows: List[Dict] = []
    scanned = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            scanned += 1
            if prefix not in line and whitelist is None:
                continue
            rec = json.loads(line)
            cats = rec.get("categories", "")
            if not has_category(cats, prefix, whitelist):
                continue
            versions = rec.get("versions") or []
            created = versions[0].get("created") if versions else None
            first = _parse_kaggle_date(created) or rec.get("update_date", "")
            if not first or not (start <= first[:10] <= end):
                continue
            last = _parse_kaggle_date(versions[-1].get("created")) if versions else None
            rows.append({
                "arxiv_id": f"{rec.get('id', '')}v{len(versions) or 1}",
                "title": rec.get("title", ""),
                "abstract": rec.get("abstract", ""),
                "authors": "; ".join(" ".join(p for p in (a[1], a[0]) if p).strip()
                                     for a in rec.get("authors_parsed", []) if a),
                "categories": cats,
                # arXiv lists the primary category first in snapshot/OAI metadata.
                "primary_category": cats.split()[0] if cats else "",
                "first_submitted": first,
                "last_updated": last or rec.get("update_date") or first,
                "doi": rec.get("doi") or "",
                "journal_ref": rec.get("journal-ref") or "",
                "license": rec.get("license") or "",
            })
            if len(rows) >= chunk_rows:
                log.info("Kaggle: scanned %d lines, emitting %d candidates.", scanned, len(rows))
                yield pd.DataFrame(rows)
                rows = []
    if rows:
        yield pd.DataFrame(rows)
    log.info("Kaggle: scanned %d lines in total.", scanned)


def _parse_kaggle_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%a, %d %b %Y %H:%M:%S %Z").strftime("%Y-%m-%d")
    except ValueError:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        return None if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def _ingest_oai(cfg: Config, store) -> None:
    from .oai_pmh import harvest

    d = cfg["data"]
    start = date.fromisoformat(d["date_start"])
    end = min(date.fromisoformat(d["date_end"]), date.today())

    def on_chunk(rows: List[Dict], lo: date, hi: date) -> None:
        if rows:
            store.upsert(clean_records(pd.DataFrame(rows), cfg).assign(source="oai_pmh"))

    n = harvest(start, end, d["oai_pmh"], os.path.join(cfg.raw_dir, "oai_state.json"),
                mailto=cfg["enrich"].get("mailto", ""), on_chunk=on_chunk)
    log.info("OAI-PMH harvested %d records this run.", n)


# ---------------------------------------------------------------------------
# Source: arXiv API (newest papers per query; demos only)
# ---------------------------------------------------------------------------
_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _harvest_arxiv(cfg: Config) -> pd.DataFrame:
    d = cfg["data"]
    cache = os.path.join(cfg.raw_dir, "arxiv_harvest.csv")
    if d["use_cache"] and os.path.exists(cache):
        df = pd.read_csv(cache, dtype=str)
        log.info("Loaded %d raw records from cache %s", len(df), cache)
        if "authors" not in df.columns:
            df = _backfill_authors(df)
            df.to_csv(cache, index=False)
        return df

    cap = cfg.max_papers or _DEFAULT_CAP
    per_query = max(1, cap // max(1, len(d["arxiv_queries"])))
    rows: List[Dict] = []
    for query in d["arxiv_queries"]:
        rows.extend(_harvest_query(query, per_query))
    if not rows:
        raise IngestError("arXiv API returned no records")
    df = pd.DataFrame(rows)
    os.makedirs(cfg.raw_dir, exist_ok=True)
    df.to_csv(cache, index=False)
    log.info("Harvested %d raw records from arXiv API; cached to %s", len(df), cache)
    return df


def _harvest_query(query: str, want: int, page: int = 100) -> List[Dict]:
    out: List[Dict] = []
    start = 0
    while len(out) < want:
        params = {"search_query": query, "start": start,
                  "max_results": min(page, want - len(out)),
                  "sortBy": "submittedDate", "sortOrder": "descending"}
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as resp:
            feed = ET.fromstring(resp.read())
        entries = feed.findall("a:entry", _ATOM)
        if not entries:
            break
        out.extend(parse_atom_entry(e) for e in entries)
        start += len(entries)
        time.sleep(3)
    log.info("  %-12s -> %d records", query, len(out))
    return out


def _backfill_authors(df: pd.DataFrame, batch: int = 100) -> pd.DataFrame:
    """Fetch authors for an older cache that predates the authors column."""
    ids = df["arxiv_id"].astype(str).tolist()
    found: Dict[str, str] = {}
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        params = {"id_list": ",".join(chunk), "max_results": len(chunk)}
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                feed = ET.fromstring(resp.read())
            for e in feed.findall("a:entry", _ATOM):
                rec = parse_atom_entry(e)
                found[base_arxiv_id(rec["arxiv_id"])] = rec["authors"]
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("Author backfill batch %d failed: %s", i // batch, exc)
        time.sleep(3)
    df = df.copy()
    df["authors"] = df["arxiv_id"].astype(str).map(base_arxiv_id).map(found).fillna("")
    return df


def parse_atom_entry(e: ET.Element) -> Dict:
    def text(tag: str) -> str:
        node = e.find(tag, _ATOM)
        return (node.text or "").strip() if node is not None and node.text else ""

    raw_id = text("a:id").rsplit("/abs/", 1)[-1]
    cats = [c.attrib.get("term", "") for c in e.findall("a:category", _ATOM)]
    primary = e.find("arxiv:primary_category", _ATOM)
    primary_cat = primary.attrib.get("term", "") if primary is not None else ""
    authors = [(a.findtext("a:name", default="", namespaces=_ATOM) or "").strip()
               for a in e.findall("a:author", _ATOM)]
    return {
        "arxiv_id": raw_id,
        "title": text("a:title"),
        "abstract": text("a:summary"),
        "authors": "; ".join(a for a in authors if a),
        "categories": " ".join(c for c in cats if c),
        "primary_category": primary_cat or (cats[0] if cats else ""),
        "date": text("a:published"),
        "first_submitted": text("a:published"),
        "last_updated": text("a:updated") or text("a:published"),
        "doi": text("arxiv:doi"),
        "journal_ref": text("arxiv:journal_ref"),
        "license": "",
    }


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

_FIRST = ["A.", "B.", "C.", "D.", "E.", "F.", "H.", "J.", "K.", "L.", "M.", "N.", "R.", "S.", "T."]
_LAST = ["Chen", "Rahman", "Garcia", "Kim", "Novak", "Okafor", "Silva", "Ito", "Patel",
         "Muller", "Haque", "Rossi", "Nguyen", "Kowalski", "Ahmed", "Larsen"]


def _synthetic(cfg: Config) -> pd.DataFrame:
    """Deterministic corpus with planted topic structure and a mild temporal
    growth trend, so every stage has meaningful signal offline. Not scientific
    data - a reproducible stand-in for CI/demo."""
    d = cfg["data"]
    rng = np.random.default_rng(cfg["seed"])
    n = cfg.max_papers or _DEFAULT_CAP
    start = pd.Timestamp(d["date_start"])
    end = pd.Timestamp(d["date_end"])
    span_days = max(1, (end - start).days)
    topics = list(_TOPICS.items())

    rows: List[Dict] = []
    for i in range(n):
        cat, spec = topics[i % len(topics)]
        topic_idx = i % len(topics)
        skew = 0.5 + 0.9 * (topic_idx / max(1, len(topics) - 1))
        frac = rng.beta(2.0, 2.0 / skew)
        when = start + pd.Timedelta(days=int(frac * span_days))
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
            "authors": "; ".join(f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
                                 for _ in range(int(rng.integers(1, 5)))),
            "categories": f"{cat} cs.AI",
            "primary_category": cat,
            "date": when.isoformat(),
        })
    log.info("Generated %d synthetic records across %d topics.", n, len(topics))
    return pd.DataFrame(rows)


def snapshot_info(df: pd.DataFrame, cfg: Config) -> Dict:
    from .quality import span_months

    d = cfg["data"]
    fallback = df.attrs.get("fallback_from")
    return {
        "source": "synthetic" if fallback else d["source"],
        "requested_source": d["source"],
        "fallback_from": fallback,
        "profile": d["profile"],
        "sampling": d["sampling"],
        "n_papers": int(len(df)),
        "n_available": int(df.attrs.get("available", len(df))),
        "date_window": [d["date_start"], d["date_end"]],
        "latest_included_date": df["date"].max().strftime("%Y-%m-%d") if len(df) else None,
        "earliest_included_date": df["date"].min().strftime("%Y-%m-%d") if len(df) else None,
        "span_months": span_months(df),
        "n_months": int(df["year_month"].nunique()) if len(df) else 0,
        "n_categories": int(df["primary_category"].nunique()) if len(df) else 0,
        "category_counts": df["primary_category"].value_counts().head(20).to_dict(),
        "harvested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
