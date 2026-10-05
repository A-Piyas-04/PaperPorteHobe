"""Configuration loading and capability detection.

A single YAML file drives every stage (NFR-03). Backends declared as ``auto``
resolve to the SRS-preferred library when importable, else a reproducible
scikit-learn fallback. The resolved choices are recorded in the run metadata so
findings can be associated with the exact pipeline configuration.
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "config.yaml")


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def capabilities() -> Dict[str, bool]:
    """Which preferred analytical backends are installed in this environment."""
    return {
        "sentence_transformers": _have("sentence_transformers"),
        "umap": _have("umap"),
        "hdbscan": _have("hdbscan"),
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

    @property
    def raw_dir(self) -> str:
        return self.abspath(self.raw["paths"]["raw_dir"])

    @property
    def processed_dir(self) -> str:
        return self.abspath(self.raw["paths"]["processed_dir"])

    @property
    def reports_dir(self) -> str:
        return self.abspath(self.raw["paths"]["reports_dir"])

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


def load_config(path: str | None = None) -> Config:
    path = path or DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, path=path)
