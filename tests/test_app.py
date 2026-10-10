"""Streamlit smoke tests: every page renders against a freshly built bundle."""
import os
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
