"""Ops safety: env overrides (kill switch), secret redaction, BYOK isolation, rate limits, SDK import guard."""
from __future__ import annotations

import logging
import os
import pathlib
import re
import sys

import pytest
from conftest import ROOT, make_config

from scholargrid.redact import SecretFilter, scrub


def test_env_overrides(monkeypatch):
    from scholargrid.config import load_config

    monkeypatch.setenv("SCHOLARGRID_LLM_PROVIDER", "none")
    monkeypatch.setenv("SCHOLARGRID_DAILY_LLM_REQUESTS", "50")
    monkeypatch.setenv("SCHOLARGRID_OFFLINE_MAX_REQUESTS", "120")
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    assert cfg["llm"]["provider"] == "none"
    assert cfg["llm"]["budget"]["online_daily_requests"] == 50
    assert cfg["llm"]["budget"]["offline_max_requests"] == 120


@pytest.mark.parametrize("secret,kept", [
    ("key AIzaSyA1234567890abcdefghijklmnopqrstuv end", "key [redacted] end"),
    ("groq gsk_abcdefghijklmnopqrstuvwxyz0123", "groq [redacted]"),
    ("or sk-or-v1-abcdefghijklmnopqrstuvwxyz", "or [redacted]"),
    ("hf hf_abcdefghijklmnopqrstuvwxyz", "hf [redacted]"),
    ("https://x/project?p=abc&k=SECRETtoken_123", "https://x/project?p=abc&k=[redacted]"),
    ("https://api.example.org/v1?key=abc123", "https://api.example.org/v1?key=[redacted]"),
    ("postgresql://user:hunter2@db.neon.tech/db", "postgresql://user:[redacted]@db.neon.tech/db"),
])
def test_scrub(secret, kept):
    assert scrub(secret) == kept


def test_log_filter_scrubs_messages():
    rec = logging.LogRecord("scholargrid.x", logging.INFO, __file__, 1, "token %s in %s", (
        "AIzaSyA1234567890abcdefghijklmnopqrstuv", "?k=abcdef"), None)
    SecretFilter().filter(rec)
    assert "AIza" not in rec.getMessage() and "abcdef" not in rec.getMessage()


def test_byok_router_has_own_usage_dir_and_key_not_logged(tmp_path, caplog):
    from scholargrid.llm import build_router

    cfg = make_config(tmp_path)
    key = "AIzaSyB" + "q" * 33
    udir = str(tmp_path / "llm_usage" / "byok" / "abc123")
    with caplog.at_level(logging.INFO, logger="scholargrid"):
        router = build_router(cfg, api_keys={"gemini": key}, usage_dir=udir)
        logging.getLogger("scholargrid.test").info("using %s", key)
    assert router.ledger.dir == udir and router.quota is not None
    assert os.path.exists(os.path.join(udir, "quota.sqlite"))
    assert not os.path.exists(os.path.join(cfg.abspath(cfg["llm"]["usage_dir"]), "quota.sqlite"))
    for p in pathlib.Path(udir).rglob("*"):
        if p.is_file() and p.suffix in (".jsonl", ".json"):
            assert key not in p.read_text(encoding="utf-8")


def test_rate_limit_window_and_heavy_slot(monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "app"))
    import limits
    import streamlit as st

    monkeypatch.setattr(limits, "_limit", lambda action: (2, 3600))
    st.session_state.pop("_rl_test", None)
    st.session_state.pop("_byok_llm", None)
    assert limits.allow("test") and limits.allow("test") and not limits.allow("test")
    with limits.heavy_slot(timeout=0.1, slots=1), pytest.raises(limits.Busy):
        with limits.heavy_slot(timeout=0.1, slots=1):
            pass


def test_only_llm_package_imports_provider_sdks():
    rx = re.compile(r"^\s*(import|from)\s+(google\.genai|google\s+import\s+genai|openai)\b", re.M)
    for p in pathlib.Path(ROOT, "scholargrid").rglob("*.py"):
        if "scholargrid/llm" in p.as_posix():
            continue
        assert not rx.search(p.read_text(encoding="utf-8")), p
