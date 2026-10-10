"""Streamlit smoke tests: every page renders against a freshly built bundle."""
import os
import sys
import textwrap

import pytest
import yaml
from conftest import ROOT, make_config

from scholargrid.pipeline import run

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP_DIR = os.path.join(ROOT, "app")
PAGES = ["home", "areas", "trends", "leads", "about", "explore", "library"]


@pytest.fixture(scope="module")
def app_config(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("app")
    cfg = make_config(tmp)
    run(cfg)
    path = tmp / "app.yaml"
    path.write_text(yaml.safe_dump(cfg.raw), encoding="utf-8")
    os.environ["SCHOLARGRID_CONFIG"] = str(path)
    yield str(path)
    os.environ.pop("SCHOLARGRID_CONFIG", None)


def _script(tmp_path, page: str) -> str:
    body = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {APP_DIR!r}); sys.path.insert(0, {ROOT!r})
        import ui
        ui.inject_css()
        ui.ensure_loaded()
        ui.global_notices(ui.get_bundle())
        from views import {page}
        {page}.render()
        ui.attribution_footer()
    """)
    path = tmp_path / f"page_{page}.py"
    path.write_text(body, encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app_config, tmp_path, page):
    at = AppTest.from_file(_script(tmp_path, page), default_timeout=120)
    at.run()
    assert not at.exception, at.exception


def test_search_results_render(app_config, tmp_path):
    at = AppTest.from_file(_script(tmp_path, "home"), default_timeout=120)
    at.session_state["q"] = "language models"
    at.run()
    assert not at.exception, at.exception
    assert any("papers" in m.value for m in at.markdown)


def test_full_app_shell(app_config):
    at = AppTest.from_file(os.path.join(APP_DIR, "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_tour_skip_restart_and_complete(app_config, tmp_path):
    at = AppTest.from_file(_script(tmp_path, "home"), default_timeout=120).run()
    _button(at, "Next").click().run()
    assert at.session_state["tour_step"] == 1
    _button(at, "Back").click().run()
    assert at.session_state["tour_step"] == 0
    _button(at, "Skip tour").click().run()
    assert at.session_state["tour_done"]
    _button(at, "Take the quick tour").click().run()
    for _ in range(3):
        _button(at, "Next").click().run()
    _button(at, "Start exploring").click().run()
    assert at.session_state["tour_done"]
    assert not at.exception


def test_search_submit_and_save_remove(app_config, tmp_path):
    at = AppTest.from_file(_script(tmp_path, "home"), default_timeout=120).run()
    at.text_input(key="q").input("language models")
    _button(at, "Find papers").click().run()
    assert not at.exception
    _button(at, "Save to reading list").click().run()
    assert len(at.session_state["saved_papers"]) == 1
    _button(at, "Remove from reading list").click().run()
    assert len(at.session_state["saved_papers"]) == 0
    assert not at.exception


def test_empty_area_filter(app_config, tmp_path):
    at = AppTest.from_file(_script(tmp_path, "areas"), default_timeout=120).run()
    at.text_input[0].input("nonexistent-area-xyz").run()
    assert any("No areas match" in info.value for info in at.info)
    assert not at.exception


def _import_explore():
    for p in (APP_DIR, ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    import ui
    from views import explore
    return ui, explore


def test_map_payload_has_overview_and_counts(app_config):
    """The map payload exposes aggregated areas, centroids and honest counts."""
    ui, explore = _import_explore()
    ui.get_bundle.clear()
    payload, pos = explore._payload(ui.bundle_version())
    n = len(ui.get_bundle().df)
    assert payload["corpusTotal"] == n
    assert payload["renderedTotal"] == len(pos)
    assert payload["maxAreaPapers"] >= 10
    assert isinstance(payload["areaEdges"], list)
    # Overview bubbles carry a rendered count and an embedding centroid.
    located = [a for a in payload["areas"] if a.get("rc")]
    assert located, "expected at least one area with rendered papers"
    assert all("x" in a and "y" in a for a in located)
    # Aggregated area links never connect an area to itself.
    assert all(a != b for a, b, _ in payload["areaEdges"])


def test_search_match_outside_sample_stays_accessible(app_config):
    """A required (search-matched) paper is added even when the sample is tiny."""
    ui, explore = _import_explore()
    cfg = ui.get_config()
    original = cfg["app"]["max_graph_nodes"]
    cfg["app"]["max_graph_nodes"] = 5  # force heavy sampling
    try:
        n = len(ui.get_bundle().df)
        target = n - 1  # most-recent-indexed paper, unlikely in a tiny sample
        payload, pos = explore._payload(ui.bundle_version(), required=(target,))
        assert target in pos, "a search match must remain reachable on the map"
        local = pos[target]
        assert any(p["i"] == local for p in payload["papers"])
    finally:
        cfg["app"]["max_graph_nodes"] = original
