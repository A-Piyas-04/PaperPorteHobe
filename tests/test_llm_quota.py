"""QuotaManager: per-minute windows, provider-timezone days, persistence, thread safety."""
from __future__ import annotations

import threading
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from scholargrid.llm import QuotaExhausted
from scholargrid.llm import quota as quota_mod
from scholargrid.llm.quota import QuotaManager

R = "gemini:gemini-3.5-flash-lite"


def test_rpm_window(tmp_path):
    q = QuotaManager(str(tmp_path), {R: {"rpm": 2}}, {"gemini": "America/Los_Angeles"})
    q.acquire(R, 10, 0)
    q.acquire(R, 10, 0)
    with pytest.raises(QuotaExhausted):
        q.acquire(R, 10, 0)


def test_tpm_window(tmp_path):
    q = QuotaManager(str(tmp_path), {R: {"tpm": 100}}, {})
    q.acquire(R, 80, 0)
    with pytest.raises(QuotaExhausted):
        q.acquire(R, 80, 0)


def test_day_uses_provider_timezone(tmp_path, monkeypatch):
    # 2026-10-21 05:30 UTC is still 2026-10-20 in Los Angeles (UTC-7 in October).
    fixed = datetime(2026, 10, 21, 5, 30, tzinfo=ZoneInfo("UTC"))

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz) if tz else fixed

    monkeypatch.setattr(quota_mod, "datetime", FixedDatetime)
    q = QuotaManager(str(tmp_path), {}, {"gemini": "America/Los_Angeles", "groq": "UTC"})
    assert q.day(R) == "2026-10-20"
    assert q.day("groq:openai/gpt-oss-120b") == "2026-10-21"
    reset = datetime.fromtimestamp(q.next_reset(R), ZoneInfo("America/Los_Angeles"))
    assert (reset.year, reset.month, reset.day, reset.hour) == (2026, 10, 21, 0)


def test_exhausted_persists(tmp_path):
    q = QuotaManager(str(tmp_path), {}, {"gemini": "America/Los_Angeles"})
    q.mark_exhausted(R)
    q.close()
    q2 = QuotaManager(str(tmp_path), {}, {"gemini": "America/Los_Angeles"})
    assert q2.exhausted_until(R) is not None
    with pytest.raises(QuotaExhausted):
        q2.acquire(R, 1, 0)


def test_rpd_and_concurrent_record(tmp_path):
    q = QuotaManager(str(tmp_path), {R: {"rpd": 10}}, {"gemini": "America/Los_Angeles"})
    threads = [threading.Thread(target=q.record, args=(R, 5)) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert q.used_today(R) == 10 and q.remaining_today(R) == 0
    with pytest.raises(QuotaExhausted):
        q.acquire(R, 1, 0)
