import os

import pytest
import yaml

from scholargrid.config import REPO_ROOT, load_config
from scholargrid.schema import ConfigError


def _write(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def test_default_config_loads_and_resolves():
    cfg = load_config()
    assert cfg["environment"] == "development"
    assert cfg.max_papers == 1500
    assert cfg.category_list() is None


def test_extends_merges_and_overrides(tmp_path):
    child = _write(tmp_path / "child.yaml", {
        "extends": os.path.join(REPO_ROOT, "configs", "config.yaml"),
        "data": {"profile": "dev"},
    })
    cfg = load_config(child)
    assert cfg.max_papers == 5000
    assert cfg["data"]["source"] == "arxiv_api"  # inherited


def test_unknown_key_fails_fast(tmp_path):
    bad = _write(tmp_path / "bad.yaml", {"data": {"sourse": "kaggle"}})
    with pytest.raises(ConfigError, match="sourse"):
        load_config(bad)


def test_production_forbids_synthetic(tmp_path):
    bad = _write(tmp_path / "prod.yaml", {
        "environment": "production",
        "data": {"source": "kaggle", "allow_synthetic_fallback": True},
    })
    with pytest.raises(ConfigError, match="synthetic"):
        load_config(bad)


def test_production_config_is_valid():
    cfg = load_config(os.path.join(REPO_ROOT, "configs", "production.yaml"))
    assert cfg.is_production
    assert cfg["data"]["allow_synthetic_fallback"] is False
    assert cfg.max_papers == 50000
    assert cfg["validation"]["enforce_gates"] is True


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("OPENALEX_MAILTO", "ops@example.org")
    monkeypatch.setenv("SCHOLARGRID_RELEASE", "2026.01.01")
    cfg = load_config()
    assert cfg["enrich"]["mailto"] == "ops@example.org"
    assert cfg["release"]["pin"] == "2026.01.01"


def test_date_end_auto(tmp_path):
    p = _write(tmp_path / "c.yaml", {"data": {"date_end": "auto"}})
    import datetime as dt

    assert load_config(p)["data"]["date_end"] == dt.date.today().isoformat()
