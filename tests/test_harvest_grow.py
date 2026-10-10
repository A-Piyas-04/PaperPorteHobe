"""Month-sliced arXiv harvest (mocked) and the background grow job."""
import datetime as dt
import os
import time

from scholargrid import data_ingest as di
from scholargrid import grow


def _fake_query(calls):
    def fake(query, want, page=100, window=None):
        calls.append((query, window))
        lo, _ = window
        return [{"arxiv_id": f"{lo:%y%m}.{len(calls):05d}", "title": f"{query} {lo}", "abstract": "x",
                 "first_submitted": lo.isoformat()}]
    return fake


def _harvest_cfg(cfg, end):
    cfg.raw["data"].update(source="arxiv_api", arxiv_slice="month", use_cache=True,
                           arxiv_queries=["cat:cs.SE", "cat:cs.PL"], date_start="2024-01-01",
                           date_end=end, profile="custom", max_papers=600)
    return cfg


def test_month_slices_cover_the_window():
    s = di.month_slices(dt.date(2023, 12, 10), dt.date(2024, 2, 5))
    assert s == [(dt.date(2023, 12, 10), dt.date(2023, 12, 31)),
                 (dt.date(2024, 1, 1), dt.date(2024, 1, 31)),
                 (dt.date(2024, 2, 1), dt.date(2024, 2, 5))]


def test_harvest_slices_are_cached_and_resumable(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(di, "_harvest_query", _fake_query(calls))
    _harvest_cfg(cfg, "2024-03-31")
    progress = []
    di.set_harvest_progress(lambda done, total, n: progress.append((done, total)))
    try:
        df = di._harvest_arxiv(cfg)
    finally:
        di.set_harvest_progress(None)
    assert len(calls) == 6 and len(df) == 6           # 2 queries x 3 months
    assert progress[-1] == (6, 6)
    assert {w[0].month for _, w in calls} == {1, 2, 3}
    di._harvest_arxiv(cfg)
    assert len(calls) == 6                            # every slice came from the cache

    cfg.raw["data"]["max_papers"] = 1200              # new per-slice quota -> new cache folder
    di._harvest_arxiv(cfg)
    assert len(calls) == 12


def test_unfinished_month_is_not_cached(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(di, "_harvest_query", _fake_query(calls))
    _harvest_cfg(cfg, dt.date.today().isoformat())
    di._harvest_arxiv(cfg)
    first = len(calls)
    di._harvest_arxiv(cfg)
    assert len(calls) - first == 2                    # only the current month, once per query


def test_failed_slices_are_retried_on_the_next_run(cfg, monkeypatch):
    calls = []
    ok = _fake_query(calls)

    def flaky(query, want, page=100, window=None):
        if query == "cat:cs.PL" and window[0].month == 2 and len(calls) < 10:
            calls.append((query, window))
            raise TimeoutError("arXiv timeout")
        return ok(query, want, page, window)

    monkeypatch.setattr(di, "_harvest_query", flaky)
    _harvest_cfg(cfg, "2024-03-31")
    assert len(di._harvest_arxiv(cfg)) == 5
    before = len(calls)
    di._harvest_arxiv(cfg)
    assert len(calls) - before == 1                   # only the failed slice is fetched again


def test_status_of_a_dead_job_is_reported_as_failed(cfg):
    path = grow.status_path(cfg)
    grow.write_status(path, state="running", pid=999999)
    assert grow.read_status(cfg)["state"] == "failed"
    assert not grow.is_running(cfg)


def test_background_job_runs_detached_and_reports_status(cfg, tmp_path):
    import yaml

    cfg_file = tmp_path / "grow.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg.raw), encoding="utf-8")
    cfg.path = str(cfg_file)
    t0 = time.monotonic()
    status = grow.start(cfg, ["--stage", "no-such-stage"])  # exits immediately with a usage error
    assert time.monotonic() - t0 < 5 and status["state"] == "running"
    deadline = time.monotonic() + 60
    while grow.read_status(cfg)["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.5)
    assert grow.read_status(cfg)["state"] == "failed"
    assert "invalid choice" in grow.log_tail(cfg)
    assert os.path.dirname(grow.status_path(cfg)) == os.path.dirname(cfg.live_dir)
