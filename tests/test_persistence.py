"""Projects database: access control, items, export/import, claim, purge, migrations."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from conftest import make_config

from scholargrid.persistence.repo import Access, AccessDenied, ProjectRepo


@pytest.fixture
def repo(tmp_path):
    from conftest import fresh_db

    from scholargrid.persistence.db import dispose_all

    cfg = make_config(tmp_path)
    fresh_db(cfg)
    r = ProjectRepo(cfg)
    yield r
    dispose_all()


def _anon(repo):
    pid, token = repo.create_project(Access(), title="Thesis", topic_query="rag")
    return pid, token, Access(project_id=pid, token=token)


def test_create_and_access(repo):
    pid, token, acc = _anon(repo)
    assert len(token) >= 40
    assert repo.get_project(pid, acc)["title"] == "Thesis"
    with pytest.raises(AccessDenied):
        repo.get_project(pid, Access(project_id=pid, token="wrong" * 9))
    user = repo.upsert_user(email="A@Example.org", name="A")
    assert repo.upsert_user(email="a@example.org")["id"] == user["id"]
    owned, _ = repo.create_project(Access(user_id=user["id"]), title="Mine")
    assert repo.get_project(owned, Access(user_id=user["id"]))["owned"]
    with pytest.raises(AccessDenied):
        repo.get_project(owned, Access(user_id="someone-else"))


def test_papers(repo):
    pid, _, acc = _anon(repo)
    paper = {"paper_id": "2401.00001", "title": "A paper", "url": "https://arxiv.org/abs/2401.00001",
             "date": "2024-01-02", "authors": "A. B"}
    assert repo.add_paper(pid, acc, paper, source="search") is True
    assert repo.add_paper(pid, acc, paper) is False
    repo.update_paper(pid, acc, "2401.00001", status="reading", tags=["core", "core", "rag"], note="read §3")
    row = repo.papers(pid, acc)[0]
    assert (row["status"], row["tags"], row["note"], row["year"]) == ("reading", ["core", "rag"], "read §3", 2024)
    with pytest.raises(ValueError):
        repo.update_paper(pid, acc, "2401.00001", status="maybe")
    repo.remove_paper(pid, acc, "2401.00001")
    assert repo.papers(pid, acc) == []
    repo.p = {**repo.p, "max_items_per_project": 10}
    for i in range(10):
        repo.add_paper(pid, acc, {"paper_id": f"p{i}", "title": "t"})
    with pytest.raises(ValueError):
        repo.add_paper(pid, acc, {"paper_id": "p10", "title": "t"})


def test_gaps_plans_notes(repo):
    pid, _, acc = _anon(repo)
    card = {"id": "g123", "title": "Gap", "statement": "s", "evidence": [], "why_it_might_not_be_a_gap": ["x"]}
    assert repo.add_gap(pid, acc, card) and not repo.add_gap(pid, acc, card)
    repo.update_gap(pid, acc, "g123", status="shortlisted", note="promising")
    g = repo.gaps(pid, acc)[0]
    assert g["payload"] == card and g["status"] == "shortlisted"
    with pytest.raises(ValueError):
        repo.update_gap(pid, acc, "g123", status="done")
    plan_id = repo.add_plan(pid, acc, {"id": "p1", "gap_id": "g123", "plan": {"title": "Plan A"}})
    assert repo.plans(pid, acc)[0]["label"] == "Plan A"
    repo.remove_plan(pid, acc, plan_id)
    assert repo.plans(pid, acc) == []
    nid = repo.add_note(pid, acc, "first")
    repo.update_note(pid, acc, nid, "edited")
    assert repo.notes(pid, acc)[0]["body"] == "edited"
    with pytest.raises(ValueError):
        repo.add_note(pid, acc, "x" * 20000)


def test_export_import(repo):
    pid, _, acc = _anon(repo)
    repo.add_paper(pid, acc, {"paper_id": "p1", "title": "One"})
    repo.add_gap(pid, acc, {"id": "g1", "title": "G"})
    repo.add_note(pid, acc, "hello")
    data = repo.export_project(pid, acc)
    new_id, new_token = repo.import_project(data, Access())
    acc2 = Access(project_id=new_id, token=new_token)
    assert new_id != pid
    again = repo.export_project(new_id, acc2)
    assert {k: v for k, v in again.items() if k != "project"} == {k: v for k, v in data.items() if k != "project"}
    with pytest.raises(ValueError):
        repo.import_project({"schema": 99}, Access())
    with pytest.raises(ValueError):
        repo.import_project({"schema": 1, "project": {}, "papers": [{"title": "no id"}]}, Access())


def test_claim_and_rotate(repo):
    pid, token, acc = _anon(repo)
    user = repo.upsert_user(email="u@example.org")
    repo.claim(pid, token, user["id"])
    assert [p["id"] for p in repo.list_projects(user["id"])] == [pid]
    assert repo.get_project(pid, acc)["owned"]                     # old link still works
    repo.rotate_token(pid, Access(user_id=user["id"]))
    with pytest.raises(AccessDenied):
        repo.get_project(pid, acc)


def test_purge(repo):
    from scholargrid.persistence.models import Project

    old, _, _ = _anon(repo)
    user = repo.upsert_user(email="o@example.org")
    owned, _ = repo.create_project(Access(user_id=user["id"]), title="Owned")
    with repo.Session.begin() as s:
        for pid in (old, owned):
            s.get(Project, pid).updated_at = datetime.now(UTC) - timedelta(days=400)
    assert repo.purge_anonymous(365) == 1
    assert repo.get_project(owned, Access(user_id=user["id"]))


def test_migrations_idempotent(tmp_path):
    from sqlalchemy import inspect

    from scholargrid.persistence.db import dispose_all, get_engine
    from scholargrid.persistence.migrate import upgrade_to_head

    from conftest import fresh_db

    cfg = make_config(tmp_path)
    fresh_db(cfg)
    upgrade_to_head(cfg)
    tables = set(inspect(get_engine(cfg)).get_table_names())
    assert {"users", "projects", "project_papers", "project_gaps", "project_plans", "project_notes",
            "ratings", "events", "llm_usage_daily", "alembic_version"} <= tables
    dispose_all()
