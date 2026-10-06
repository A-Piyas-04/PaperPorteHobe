"""Configuration loading and capability detection.

A single YAML file drives every stage (NFR-03). A file may ``extends:`` another
one and override only what differs (``production.yaml`` extends
``config.yaml``). The merged result is validated against the pydantic schema in
:mod:`scholargrid.schema`, secrets are taken from environment variables, and
derived values (profile corpus size, ``date_end: auto``) are resolved once here.

Backends declared as ``auto`` resolve to the SRS-preferred library when
importable, else a reproducible scikit-learn fallback. The resolved choices are
recorded in the run metadata.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import yaml

from .schema import ConfigError, validate_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "config.yaml")

PROFILE_SIZES: Dict[str, Optional[int]] = {"dev": 5000, "standard": 50000, "full": None}

_ENV_OVERRIDES = {
    "OPENALEX_MAILTO": ("enrich", "mailto"),
    "SCHOLARGRID_ENV": ("environment",),
    "SCHOLARGRID_RELEASE": ("release", "pin"),
}


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def capabilities() -> Dict[str, bool]:
    """Which optional backends are installed in this environment."""
    return {
        "sentence_transformers": _have("sentence_transformers"),
        "umap": _have("umap"),
        "hdbscan": _have("hdbscan"),
        "faiss": _have("faiss"),
        "duckdb": _have("duckdb"),
    }


@dataclass
class Config:
    """Thin dot-and-dict accessor over the parsed YAML."""

    raw: Dict[str, Any] = field(default_factory=dict)
    path: str = DEFAULT_CONFIG

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # -- absolute paths -----------------------------------------------------
    def abspath(self, rel: str) -> str:
        return rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)

    def _path(self, key: str) -> str:
        return self.abspath(self.raw["paths"][key])

    @property
    def raw_dir(self) -> str:
        return self._path("raw_dir")

    @property
    def processed_dir(self) -> str:
        return self._path("processed_dir")

    @property
    def reports_dir(self) -> str:
        return self._path("reports_dir")

    @property
    def stages_dir(self) -> str:
        return self._path("stages_dir")

    @property
    def releases_dir(self) -> str:
        return self._path("releases_dir")

    # -- derived settings ---------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.raw.get("environment") == "production"

    @property
    def max_papers(self) -> Optional[int]:
        """Corpus cap for the active profile (``None`` = no cap)."""
        d = self.raw["data"]
        if d["profile"] == "custom":
            return d.get("max_papers")
        return PROFILE_SIZES[d["profile"]]

    def category_list(self) -> Optional[List[str]]:
        """Explicit category whitelist, or ``None`` for every ``cs.*`` category."""
        cats = self.raw["data"]["categories"]
        return None if cats == "all_cs" else list(cats)

    def config_hash(self) -> str:
        blob = json.dumps(self.raw, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    # -- resolved backend choices ------------------------------------------
    def resolve_embedding_backend(self) -> str:
        want = self.raw["embedding"]["backend"]
        caps = capabilities()
        if want == "auto":
            return "minilm" if caps["sentence_transformers"] else "tfidf"
        if want == "minilm" and not caps["sentence_transformers"]:
            return "tfidf"
        return want

    def resolve_reduce_backend(self, section: str = "reduce") -> str:
        want = self.raw[section]["backend"]
        if want == "auto":
            return "umap" if capabilities()["umap"] else "pca"
        if want == "umap" and not capabilities()["umap"]:
            return "pca"
        return want

    def resolve_cluster_backend(self) -> str:
        want = self.raw["cluster"]["backend"]
        if want == "auto":
            return "hdbscan" if capabilities()["hdbscan"] else "dbscan"
        if want == "hdbscan" and not capabilities()["hdbscan"]:
            return "dbscan"
        return want

    def resolve_index_backend(self, n_items: int) -> str:
        icfg = self.raw["index"]
        want = icfg["backend"]
        if want == "auto":
            if n_items < int(icfg["exact_threshold"]) or not capabilities()["faiss"]:
                return "exact"
            return "faiss_hnsw"
        if want == "faiss_hnsw" and not capabilities()["faiss"]:
            return "exact"
        return want

    def resolve_storage_backend(self) -> str:
        want = self.raw["storage"]["backend"]
        if want == "duckdb" and not capabilities()["duckdb"]:
            return "parquet"
        return want


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: str, seen: Optional[set] = None) -> Dict[str, Any]:
    seen = seen or set()
    real = os.path.realpath(path)
    if real in seen:
        raise ConfigError(f"Circular 'extends' in {path}")
    seen.add(real)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    parent = raw.pop("extends", None)
    if parent:
        parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(path), parent)
        raw = _deep_merge(_read_yaml(parent_path, seen), raw)
    return raw


def _apply_env(raw: Dict[str, Any]) -> None:
    for env, keys in _ENV_OVERRIDES.items():
        value = os.environ.get(env)
        if not value:
            continue
        node = raw
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value


def _check_production(raw: Dict[str, Any]) -> None:
    if raw.get("environment") != "production":
        return
    d = raw["data"]
    if d["source"] == "synthetic" or d["allow_synthetic_fallback"]:
        raise ConfigError("Production configs must not use or fall back to synthetic data "
                          "(data.source != synthetic, data.allow_synthetic_fallback: false).")


def load_config(path: str | None = None) -> Config:
    path = path or os.environ.get("SCHOLARGRID_CONFIG") or DEFAULT_CONFIG
    raw = _read_yaml(path)
    _apply_env(raw)
    raw = validate_config(raw)
    if raw["data"]["date_end"] == "auto":
        raw["data"]["date_end"] = date.today().isoformat()
    _check_production(raw)
    return Config(raw=raw, path=path)
