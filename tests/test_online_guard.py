"""Online free-quota guard: global and per-session allowances, atomic counting, context propagation."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from conftest import fresh_db, make_config

from scholargrid.llm import BudgetExceeded
from scholargrid.llm.context import current_session
from scholargrid.llm.guard import OnlineGuard, install_guard
from scholargrid.persistence.repo import ProjectRepo


@pytest.fixture
def repo(tmp_path):
    from scholargrid.persistence.db import dispose_all

    cfg = make_config(tmp_path)
    fresh_db(cfg)
    yield cfg, ProjectRepo(cfg)
    dispose_all()


def test_global_and_session_allowances(repo):
    cfg, r = repo
    cfg.raw["llm"]["budget"].update(online_daily_requests=3, online_session_requests=2)
    g = OnlineGuard(cfg, r)
    tok = current_session.set("s1")
    try:
        g.check("label")
        g.record("label", "gemini:x")
        g.record("label", "gemini:x")
        with pytest.raises(BudgetExceeded, match="session"):
            g.check("label")
        current_session.set("s2")                      # another visitor still has their allowance
        g.check("label")
        g.record("label", "gemini:x")
        with pytest.raises(BudgetExceeded, match="daily"):
            g.check("label")
    finally:
        current_session.reset(tok)
    assert r.requests_today("global") == 3 and r.requests_today("session:s1") == 2


def test_add_request_is_atomic(repo):
    _, r = repo
    threads = [threading.Thread(target=r.add_request, args=(["global"],)) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert r.requests_today("global") == 20


def test_guard_not_on_byok_router(repo, monkeypatch):
    cfg, r = repo
    from scholargrid.llm import build_router

    router = build_router(cfg, api_keys={"gemini": "AIza" + "x" * 35}, usage_dir=str(cfg.abspath("byok_test")))
    assert router.ledger.extra_guard is None and router.ledger.on_record is None
    fake = SimpleNamespace(provider="fake", ledger=None)
    assert install_guard(fake, cfg, r) is None
    router.provider = "live"
    guard = install_guard(router, cfg, r)
    assert guard is not None and router.ledger.extra_guard == guard.check


def test_map_structured_propagates_session(tmp_path):
    from scholargrid.llm import FakeLLM
    from scholargrid.llm.router import LLMRouter

    cfg = make_config(tmp_path)
    seen = []

    class Probe(FakeLLM):
        def structured(self, task, prompt, schema, **v):
            seen.append(current_session.get())
            return super().structured(task, prompt, schema, **v)

    probe = Probe(cfg)
    from pydantic import BaseModel

    class Check(BaseModel):
        ok: bool
        echo: str

    tok = current_session.set("sess-42")
    try:
        LLMRouter.map_structured(probe, "check", "check", Check, [{"word": "a"}, {"word": "b"}])  # type: ignore[arg-type]
    finally:
        current_session.reset(tok)
    assert seen == ["sess-42", "sess-42"]
