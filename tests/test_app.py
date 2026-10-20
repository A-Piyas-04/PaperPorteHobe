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
PAGES = ["home", "areas", "trends", "leads", "about", "explore", "library", "start", "topic", "reading", "gaps",
         "plan", "project"]


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
    assert any("What do you want to research?" in m.value for m in at.markdown)   # Start is the default page


def test_topic_and_reading_flow(app_config, tmp_path):
    at = AppTest.from_file(_script(tmp_path, "topic"), default_timeout=180)
    at.session_state["topic_query"] = "language models"
    at.run()
    assert not at.exception, at.exception
    assert any("sub-topics" in m.value for m in at.markdown)
    assert at.session_state["j_topic"]
    at.button(key=next(b.key for b in at.button if str(b.key).startswith("ws_open_"))).click().run()
    assert not at.exception, at.exception
    assert any("Representative papers" in m.value for m in at.markdown)

    at2 = AppTest.from_file(_script(tmp_path, "reading"), default_timeout=180)
    at2.session_state["topic_query"] = "language models"
    at2.run()
    assert not at2.exception, at2.exception
    text = " ".join(m.value for m in at2.markdown)
    assert "Foundations" in text or "most-cited papers retrieved" in text


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _tour_script(tmp_path) -> str:
    path = tmp_path / "tour_home.py"
    path.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {APP_DIR!r}); sys.path.insert(0, {ROOT!r})
        import ui
        ui.inject_css()
        ui.ensure_loaded()
        from views import home, tour
        tour.header_control()
        tour.render_dock()
        home.render()
    """), encoding="utf-8")
    return str(path)


def test_tour_skip_restart_and_complete(app_config, tmp_path):
    _import_explore()
    from views import tour

    at = AppTest.from_file(_tour_script(tmp_path), default_timeout=120).run()
    at.button(key="tour_invite_start").click().run()
    assert at.session_state["tour_active"] and at.session_state["tour_step"] == 0
    at.button(key="tour_next").click().run()
    assert at.session_state["tour_step"] == 1
    at.button(key="tour_back").click().run()
    assert at.session_state["tour_step"] == 0
    at.button(key="tour_skip").click().run()
    assert at.session_state["tour_done"] and not at.session_state["tour_active"]
    at.button(key="tour_hdr_start").click().run()
    assert at.session_state["tour_active"]
    for _ in range(tour.TOTAL):
        at.button(key="tour_next").click().run()
    at.button(key="tour_finish").click().run()
    assert at.session_state["tour_done"] and not at.session_state["tour_active"]
    assert not at.exception


def _repo():
    ui, _ = _import_explore()
    from scholargrid.persistence import get_repo

    return get_repo(ui.get_config())


def test_search_submit_and_save_remove(app_config, tmp_path):
    from scholargrid.persistence.repo import Access

    at = AppTest.from_file(_script(tmp_path, "home"), default_timeout=120).run()
    at.text_input(key="q").input("language models")
    _button(at, "Find papers").click().run()
    assert not at.exception
    _button(at, "Save to project").click().run()
    assert not at.exception, at.exception
    pid, token = at.session_state["proj_id"], at.session_state["proj_token"]
    assert at.query_params["p"] == [pid] or at.query_params["p"] == pid
    acc = Access(project_id=pid, token=token)
    assert len(_repo().papers(pid, acc)) == 1
    _button(at, "Remove from project").click().run()
    assert len(_repo().papers(pid, acc)) == 0
    assert not at.exception


def test_project_page_lists_papers(app_config, tmp_path):
    from scholargrid.persistence.repo import Access

    repo = _repo()
    pid, token = repo.create_project(Access(), title="Test project")
    acc = Access(project_id=pid, token=token)
    for i, title in enumerate(["First saved paper", "Second saved paper"]):
        repo.add_paper(pid, acc, {"paper_id": f"x{i}", "title": title, "url": "https://example.org"})
    at = AppTest.from_file(_script(tmp_path, "project"), default_timeout=120)
    at.session_state["proj_id"], at.session_state["proj_token"] = pid, token
    at.run()
    assert not at.exception, at.exception
    assert set(at.dataframe[0].value["title"]) == {"First saved paper", "Second saved paper"}


def test_project_page_without_persistence(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, persistence={"enabled": False})
    run(cfg)
    path = tmp_path / "nopersist.yaml"
    path.write_text(yaml.safe_dump(cfg.raw), encoding="utf-8")
    monkeypatch.setenv("SCHOLARGRID_CONFIG", str(path))
    script = tmp_path / "proj_fallback.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {APP_DIR!r}); sys.path.insert(0, {ROOT!r})
        import streamlit as st
        st.cache_resource.clear(); st.cache_data.clear()
        import ui
        ui.ensure_loaded()
        from views import project
        project.render()
    """), encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=180).run()
    assert not at.exception, at.exception
    assert any("session only" in m.value for m in at.markdown)


def test_citations_ready_thresholds():
    from types import SimpleNamespace

    import pandas as pd

    ui, _ = _import_explore()
    df = pd.DataFrame({"cited_by_count": [1] * 10 + [0] * 90,
                       "citation_velocity": [0.5] + [0.0] * 99})
    bundle = SimpleNamespace(meta={"counts": {"papers": 100, "with_citations": 9}}, df=df)
    assert not ui.citations_ready(bundle) and not ui.velocity_ready(bundle)
    bundle.meta["counts"]["with_citations"] = 10
    assert ui.citations_ready(bundle) and ui.velocity_ready(bundle)
    df.loc[0, "citation_velocity"] = 0.0
    assert not ui.velocity_ready(bundle)


def test_citation_controls_hidden_without_citation_data(app_config, tmp_path):
    """The synthetic corpus has no citations, so citation sorts and filters must not be offered."""
    at = AppTest.from_file(_script(tmp_path, "home"), default_timeout=120)
    at.session_state["q"] = "language models"
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="f_sort").options == ["Relevance", "Newest"]
    assert not [t for t in at.toggle if t.key == "f_boost"]
    assert not [n for n in at.number_input if n.key == "f_cites"]


def test_live_paper_is_marked_unanalyzed(app_config, tmp_path):
    path = tmp_path / "live_card.py"
    path.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {APP_DIR!r}); sys.path.insert(0, {ROOT!r})
        import ui
        ui.paper_card("A live paper", "doi:10.1/x", authors="A. B", cites=0, new=True)
    """), encoding="utf-8")
    at = AppTest.from_file(str(path), default_timeout=120).run()
    html_out = " ".join(m.value for m in at.markdown)
    ui, _ = _import_explore()
    assert ui.NEW_CHIP in html_out and "Not part of the areas" in html_out


def test_gaps_page_with_cached_cards(app_config, tmp_path):
    ui, _ = _import_explore()
    from scholargrid.gaps import gaps_for_workspace
    from scholargrid.llm import FakeLLM
    from scholargrid.workspace import build_workspace

    cfg, bundle = ui.get_config(), ui.get_bundle()
    ws = build_workspace("language models", bundle, None, cfg, bundle_version=ui.bundle_version(), llm=FakeLLM(cfg))
    data = gaps_for_workspace(ws, bundle, None, cfg, FakeLLM(cfg))
    assert data["cards"]
    at = AppTest.from_file(_script(tmp_path, "gaps"), default_timeout=180)
    at.session_state["topic_query"] = "language models"
    at.run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Why this might not be a gap" in text and data["cards"][0]["title"][:30] in text
    assert at.session_state["j_gaps"]


def test_gaps_page_by_area_without_topic(app_config, tmp_path):
    at = AppTest.from_file(_script(tmp_path, "gaps"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("Why this might not be a gap" in m.value for m in at.markdown)


def test_plan_page_generates(app_config, tmp_path):
    ui, _ = _import_explore()
    card = next(c for cs in ui.get_bundle().gaps["areas"].values() for c in cs)
    at = AppTest.from_file(_script(tmp_path, "plan"), default_timeout=180)
    at.session_state["plan_gap"] = card
    at.run()
    assert not at.exception, at.exception
    at.button(key="FormSubmitter:plan-inputs-Create my starter plan").click().run()
    assert not at.exception, at.exception
    labels = [t.label for t in at.tabs]
    assert "Timeline" in labels and "Export" in labels
    assert at.session_state["j_plan"]


def test_evaluate_page_is_blind_and_saves(app_config, tmp_path, monkeypatch):
    ui, _ = _import_explore()
    from scholargrid.evaluation import make_gap_round, save_round

    cfg = ui.get_config()
    cards = [c for cs in ui.get_bundle().gaps["areas"].values() for c in cs]
    save_round(cfg, make_gap_round("blindtest", cards, 3, 0.0, seed=1))
    monkeypatch.setenv("SCHOLARGRID_EVAL_MODE", "1")
    at = AppTest.from_file(_script(tmp_path, "evaluate"), default_timeout=120)
    at.session_state["rater"] = "R-test"
    at.run()
    assert not at.exception, at.exception
    at.selectbox(key="eval_round").set_value("blindtest").run()
    text = " ".join(m.value for m in at.markdown if not m.value.lstrip().startswith("<style")).lower()
    assert "why it might not be a gap" in text
    for leak in ("confidence", "score", "novel"):
        assert leak not in text, leak
    for r in at.radio:
        r.set_value(r.options[0])
    at.button(key=next(b.key for b in at.button if "FormSubmitter" in str(b.key))).click().run()
    assert not at.exception, at.exception
    assert [r["rater"] for r in _repo().ratings(batch="blindtest")] == ["R-test"]


def test_evaluate_hidden_without_eval_mode(app_config, monkeypatch):
    monkeypatch.delenv("SCHOLARGRID_EVAL_MODE", raising=False)
    at = AppTest.from_file(os.path.join(APP_DIR, "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert "evaluate" not in at.session_state["_pages"]


def test_area_detail_shows_read_first(app_config, tmp_path):
    ui, _ = _import_explore()
    cid = int(next(iter(ui.get_bundle().clusters_meta)))
    at = AppTest.from_file(_script(tmp_path, "areas"), default_timeout=120)
    at.session_state["area_open"] = cid
    at.run()
    assert not at.exception, at.exception
    assert any("Read first" in m.value for m in at.markdown)


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
