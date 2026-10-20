# Plan 08: Durable projects (database, optional login, share links, My project page)

> **Prerequisites:** plans 01 and 05–07 are merged. `pip install -e ".[db]"` (and `".[db-postgres,auth]"` for production).
> **Produces:** `scholargrid/persistence/`: SQLAlchemy 2.x models and repository, Alembic migrations, SQLite for development and Postgres for production. Projects are identified by an owner (optional Streamlit OIDC login) **or** an unguessable capability link. Saved papers get status, tags and notes; saved gaps and plans get status and notes; there are journal notes, JSON export/import and a retention purge. The session-only **Reading list** is replaced by **My project**.
> **Why:** starting research takes weeks. Session-only storage (the current state) loses everything on reload, which makes the product unusable for its actual job.

---

## 1. Prompt to paste into Cursor

```text
Implement plan file 08_persistence_projects_auth.md. Use SQLAlchemy 2.0 typed ORM (Mapped/mapped_column),
Alembic for migrations, and keep the app working with persistence.enabled=false (session-only fallback).
Treat project access tokens as secrets (never log them; compare with secrets.compare_digest). Order:
§3 config → §4 models → §5 engine/session → §6 Alembic → §7 repository → §8 identity & access →
§9 app integration → §10 My project page → §11 CLI → §12 tests. Finish with §13.
```

---

## 2. Identity model (two ways in, one data model)

| Mode | How the user is recognised | What they get |
|---|---|---|
| **Anonymous (default)** | A **capability link** `…/project?p=<project_id>&k=<token>`. The token is 32 bytes from `secrets.token_urlsafe`. | One project per link. Works on free hosting with no setup. "Anyone with the link can open it." |
| **Signed in** (`persistence.auth: oidc`) | Streamlit's built-in `st.login()` (OIDC, e.g. Google), then `st.user.email` | Many projects, a project switcher, and the ability to claim anonymous projects into the account |

Both store rows in the same tables. An anonymous project simply has `owner_id = NULL`.

---

## 3. Config

```python
class Persistence(_Section):
    enabled: bool = True
    db_url_env: str = "SCHOLARGRID_DB_URL"
    default_url: str = "sqlite:///data/app.db"
    auth: Literal["none", "oidc"] = "none"
    oidc_provider: Optional[str] = None            # name of an [auth.<provider>] block in secrets.toml, or None for [auth]
    allow_anonymous_projects: bool = True
    max_projects_per_user: int = Field(default=50, ge=1)
    max_items_per_project: int = Field(default=1000, ge=10)
    max_note_chars: int = Field(default=10000, ge=100)
    anonymous_retention_days: int = Field(default=365, ge=7)
    create_rate_per_hour: int = Field(default=10, ge=1)   # per browser session
```

`Config.db_url` property: `os.environ.get(db_url_env) or default_url`. Resolve a relative SQLite path against `REPO_ROOT`, and create the `data/` directory if needed.

`tests/conftest.py::make_config`: set `raw["persistence"]["default_url"] = f"sqlite:///{tmp_path/'app.db'}"`.

---

## 4. Models: `scholargrid/persistence/models.py`

```python
"""ORM models. JSON payloads hold GapCard / PlanDocument / paper dicts as produced by plans 05-07."""
from __future__ import annotations

import secrets
import uuid
from datetime import UTC, datetime
from typing import List, Optional

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JsonType = JSON().with_variant(JSONB(), "postgresql")


def _now() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[Optional[str]] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    provider: Mapped[str] = mapped_column(String(50), default="oidc")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    owner_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    access_token: Mapped[str] = mapped_column(String(64), unique=True,
                                              default=lambda: secrets.token_urlsafe(32))
    title: Mapped[str] = mapped_column(String(200), default="My research project")
    topic_query: Mapped[str] = mapped_column(String(300), default="")
    scope_name: Mapped[str] = mapped_column(String(100), default="")
    bundle_version: Mapped[str] = mapped_column(String(64), default="")
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
    papers: Mapped[List[ProjectPaper]] = relationship(back_populates="project", cascade="all, delete-orphan")
    gaps: Mapped[List[ProjectGap]] = relationship(back_populates="project", cascade="all, delete-orphan")
    plans: Mapped[List[ProjectPlan]] = relationship(back_populates="project", cascade="all, delete-orphan")
    notes: Mapped[List[ProjectNote]] = relationship(back_populates="project", cascade="all, delete-orphan")


class ProjectPaper(Base):
    __tablename__ = "project_papers"
    __table_args__ = (UniqueConstraint("project_id", "paper_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    paper_id: Mapped[str] = mapped_column(String(200))
    title: Mapped[str] = mapped_column(Text, default="")
    url: Mapped[str] = mapped_column(Text, default="")
    year: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default="to_read")     # to_read|reading|read|key
    tags: Mapped[list] = mapped_column(JsonType, default=list)
    note: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(JsonType, default=dict)          # authors, venue, doi, abstract…
    source: Mapped[str] = mapped_column(String(30), default="")            # search|reading_path|gap_evidence|…
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
    project: Mapped[Project] = relationship(back_populates="papers")


class ProjectGap(Base):
    __tablename__ = "project_gaps"
    __table_args__ = (UniqueConstraint("project_id", "gap_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    gap_id: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20), default="considering")  # considering|shortlisted|chosen|rejected
    note: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(JsonType)                          # GapCard.model_dump()
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
    project: Mapped[Project] = relationship(back_populates="gaps")


class ProjectPlan(Base):
    __tablename__ = "project_plans"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    gap_id: Mapped[str] = mapped_column(String(40), index=True)
    label: Mapped[str] = mapped_column(String(200), default="")
    payload: Mapped[dict] = mapped_column(JsonType)                          # PlanDocument.model_dump()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    project: Mapped[Project] = relationship(back_populates="plans")


class ProjectNote(Base):
    __tablename__ = "project_notes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    body: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
    project: Mapped[Project] = relationship(back_populates="notes")
```

Plan 09 adds `Rating` and `Event`, and plan 11 adds `LlmUsageDaily`, each with its own Alembic revision.

---

## 5. Engine and sessions: `scholargrid/persistence/db.py`

```python
"""One engine per process; SQLite (dev) or Postgres (prod) from SCHOLARGRID_DB_URL."""
from __future__ import annotations

import os
import threading
from typing import Dict

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from ..config import REPO_ROOT, Config

_ENGINES: Dict[str, Engine] = {}
_LOCK = threading.Lock()


def resolve_url(cfg: Config) -> str:
    p = cfg["persistence"]
    url = os.environ.get(p["db_url_env"]) or p["default_url"]
    if url.startswith("postgres://"):                 # Heroku/Supabase style -> SQLAlchemy driver URL
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        path = os.path.join(REPO_ROOT, url[len("sqlite:///"):])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = "sqlite:///" + path
    return url


def get_engine(cfg: Config) -> Engine:
    url = resolve_url(cfg)
    with _LOCK:
        if url in _ENGINES:
            return _ENGINES[url]
        if url.startswith("sqlite"):
            engine = create_engine(url, connect_args={"check_same_thread": False}, pool_pre_ping=True)

            @event.listens_for(engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        else:
            engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5, pool_recycle=1800)
        _ENGINES[url] = engine
        return engine


def session_factory(cfg: Config) -> sessionmaker:
    return sessionmaker(get_engine(cfg), expire_on_commit=False)
```

---

## 6. Alembic migrations

```bash
pip install -e ".[db]"
alembic init scholargrid/persistence/migrations
```

- Move the generated `alembic.ini` to the repo root, with `script_location = scholargrid/persistence/migrations`.
- Edit `migrations/env.py`:
  ```python
  from scholargrid.config import load_config
  from scholargrid.persistence.db import resolve_url
  from scholargrid.persistence.models import Base

  config.set_main_option("sqlalchemy.url", resolve_url(load_config()))
  target_metadata = Base.metadata
  # SQLite needs batch mode for ALTER TABLE:
  context.configure(..., render_as_batch=True)
  ```
- Create the first revision: `alembic revision --autogenerate -m "projects initial"`. Review it by hand and commit it.
- Programmatic upgrade at app start and in tests (`scholargrid/persistence/migrate.py`):
  ```python
  def upgrade_to_head(cfg) -> None:
      from alembic import command
      from alembic.config import Config as AlembicConfig

      ac = AlembicConfig(os.path.join(REPO_ROOT, "alembic.ini"))
      ac.set_main_option("script_location", os.path.join(REPO_ROOT, "scholargrid", "persistence", "migrations"))
      ac.set_main_option("sqlalchemy.url", resolve_url(cfg))
      command.upgrade(ac, "head")
  ```
- The app calls `upgrade_to_head` once per process (`@st.cache_resource`). In production with Postgres you may instead run `alembic upgrade head` in the deploy step (plan 11). Both are idempotent.
- Add the migrations directory to the Dockerfile `COPY` lines (plan 11).

---

## 7. Repository: `scholargrid/persistence/repo.py`

All public methods take an **`Access`** (who is asking) and enforce authorization in one place.

```python
@dataclass(frozen=True)
class Access:
    user_id: Optional[str] = None       # signed-in user
    project_id: Optional[str] = None    # capability link
    token: Optional[str] = None


class AccessDenied(PermissionError):
    pass


class ProjectRepo:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.Session = session_factory(cfg)
        self.p = cfg["persistence"]

    # -- authorization ---------------------------------------------------------------------
    def _load(self, s, project_id: str, access: Access) -> Project:
        proj = s.get(Project, project_id)
        if proj is None or proj.archived:
            raise AccessDenied("not found")
        if access.user_id and proj.owner_id == access.user_id:
            return proj
        if access.token and access.project_id == project_id and secrets.compare_digest(proj.access_token, access.token):
            return proj
        raise AccessDenied("no access")

    # -- users ----------------------------------------------------------------------------
    def upsert_user(self, *, email: str, name: str = "", provider: str = "oidc") -> Dict:
        """Find by email (case-insensitive) or create; updates last_seen_at. Returns {id, email, name}."""

    # -- projects -------------------------------------------------------------------------
    def create_project(self, access: Access, *, title: str, topic_query: str = "", scope_name: str = "",
                       bundle_version: str = "") -> Tuple[str, str]:
        """Returns (project_id, access_token). Enforces max_projects_per_user for signed-in users."""

    def get_project(self, project_id: str, access: Access) -> Dict: ...          # summary dict incl. counts
    def list_projects(self, user_id: str) -> List[Dict]: ...
    def rename(self, project_id: str, access: Access, title: str) -> None: ...
    def delete_project(self, project_id: str, access: Access) -> None: ...     # hard delete (cascade)
    def rotate_token(self, project_id: str, access: Access) -> str: ...        # invalidates old share links
    def claim(self, project_id: str, token: str, user_id: str) -> None: ...    # anonymous -> owned

    # -- papers ---------------------------------------------------------------------------
    def add_paper(self, project_id: str, access: Access, paper: Dict, source: str = "") -> bool: ...  # upsert; False if exists
    def update_paper(self, project_id: str, access: Access, paper_id: str, *, status=None, tags=None, note=None) -> None: ...
    def remove_paper(self, project_id: str, access: Access, paper_id: str) -> None: ...
    def papers(self, project_id: str, access: Access) -> List[Dict]: ...

    # -- gaps / plans / notes -------------------------------------------------------------
    def add_gap(self, project_id: str, access: Access, card: Dict) -> bool: ...
    def update_gap(self, project_id: str, access: Access, gap_id: str, *, status=None, note=None) -> None: ...
    def remove_gap(self, project_id: str, access: Access, gap_id: str) -> None: ...
    def gaps(self, project_id: str, access: Access) -> List[Dict]: ...
    def add_plan(self, project_id: str, access: Access, plan_doc: Dict, label: str = "") -> int: ...
    def plans(self, project_id: str, access: Access) -> List[Dict]: ...
    def remove_plan(self, project_id: str, access: Access, plan_id: int) -> None: ...
    def add_note(self, project_id: str, access: Access, body: str) -> int: ...
    def update_note(self, project_id: str, access: Access, note_id: int, body: str) -> None: ...
    def notes(self, project_id: str, access: Access) -> List[Dict]: ...

    # -- portability ----------------------------------------------------------------------
    def export_project(self, project_id: str, access: Access) -> Dict: ...     # {"schema": 1, "project", "papers", "gaps", "plans", "notes"}
    def import_project(self, data: Dict, access: Access) -> Tuple[str, str]: ...  # new ids/token; validates schema & sizes

    # -- maintenance ----------------------------------------------------------------------
    def purge_anonymous(self, older_than_days: int) -> int: ...                # owner_id IS NULL and updated_at < cutoff
```

Implementation rules:
- Each method uses `with self.Session.begin() as s:` (one transaction).
- Every mutation touches `Project.updated_at` (retention and "recent projects" use it).
- Validate sizes: titles ≤ 200 chars, notes ≤ `max_note_chars`, tags ≤ 20 of ≤ 40 chars, items ≤ `max_items_per_project` (raise `ValueError` with a friendly message).
- `status` values come from fixed sets (`PAPER_STATUSES`, `GAP_STATUSES`). Reject others.
- Return plain dicts (not ORM objects) so Streamlit can cache and serialize them.
- Never include `access_token` in `get_project` output except in `create_project`/`rotate_token` return values and for the owner's "share link" display.

---

## 8. Identity and access in the app: `app/project_access.py` (new, imported by `ui.py`)

```python
"""Who is the current user and which project is active (session + URL)."""
from __future__ import annotations

import streamlit as st

from scholargrid.persistence.repo import Access


def current_user(cfg, repo) -> dict | None:
    if cfg["persistence"]["auth"] != "oidc":
        return None
    user = getattr(st, "user", None)
    if user is None or not user.is_logged_in:
        return None
    return repo.upsert_user(email=user.email, name=getattr(user, "name", "") or "")


def active_access(cfg, repo) -> Access:
    """Signed-in user, else a capability link from the URL (?p=&k=) or the session."""
    qp = st.query_params
    if "p" in qp and "k" in qp:
        st.session_state["proj_id"], st.session_state["proj_token"] = qp["p"], qp["k"]
    user = current_user(cfg, repo)
    return Access(user_id=user["id"] if user else None,
                  project_id=st.session_state.get("proj_id"), token=st.session_state.get("proj_token"))


def ensure_project(cfg, repo, *, title: str, topic: str) -> str:
    """Active project id; creates one on first save. Anonymous: puts ?p=&k= into the URL."""
    acc = active_access(cfg, repo)
    pid = st.session_state.get("proj_id")
    if pid:
        return pid
    _rate_limit_creation(cfg)
    pid, token = repo.create_project(acc, title=title or "My research project", topic_query=topic)
    st.session_state["proj_id"], st.session_state["proj_token"] = pid, token
    if not acc.user_id:
        st.query_params["p"], st.query_params["k"] = pid, token
        st.session_state["show_link_notice"] = True
    _migrate_session_items(repo, pid, Access(acc.user_id, pid, token))
    return pid
```

- `_rate_limit_creation`: at most `create_rate_per_hour` creations per session (timestamps in session state). Otherwise show an error.
- `_migrate_session_items`: move `saved_papers` (the old reading list) and `saved_gaps` from the session into the new project once.
- **Login button** (only when `auth: oidc`): in the top nav, `st.button("Sign in")` → `st.login(cfg["persistence"]["oidc_provider"])` if a provider is named, else `st.login()`. When signed in, show the user's name and `st.logout` in the More menu.
- **Link notice:** after an anonymous project is created, show a sticky `ui.note`: "Your project is saved. **Bookmark this page.** Anyone with this link can view and edit the project." Include a copy-able `st.code(full_url)`. Build `full_url` from `st.context.url` if available; otherwise show the relative `?p=…&k=…` part.

**OIDC setup** (document in `Docs/deployment.md`, done in plan 11). Streamlit's native auth needs `Authlib>=1.3.2` and a `.streamlit/secrets.toml` block such as:

```toml
[auth]
redirect_uri = "https://<your-app>/oauth2callback"
cookie_secret = "<long random string>"
client_id = "<from Google Cloud console>"
client_secret = "<...>"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

---

## 9. App integration

### 9.1 `ui.py`
- `@st.cache_resource def get_repo()`: returns `None` when `persistence.enabled` is false; otherwise runs `upgrade_to_head(cfg)` once, then returns `ProjectRepo(cfg)`.
- `ui.save_paper(record, source)`: if the repo exists → `ensure_project` + `repo.add_paper`; else the old session behaviour. Return the new state for button labels.
- `ui.save_gap(card)` and `ui.save_plan(doc)`: same pattern.
- `ui.is_saved_paper(paper_id)`: checks a cached set of project paper ids. Invalidate it after mutations: keep `st.session_state["_proj_rev"]` and bump it after each write.

### 9.2 Replace callers
- `views/library.py::toggle` → `ui.save_paper` / remove. Keep the function name so the existing callers in `home.py` and `explore.py` keep working.
- Gap card "Save to project" → `ui.save_gap`; Plan page "Save plan to project" → `ui.save_plan`.
- Reading-path items and evidence papers get "Save" buttons with `source="reading_path"` / `"gap_evidence"`.

### 9.3 Navigation
Replace the `library` page in the journey with `project` (title "My project", icon `:material/folder_open:`, url_path `project`). Keep `/reading-list` as an alias page that redirects (`st.switch_page`) to `project`, so old links keep working.

---

## 10. Page: `app/views/project.py` ("My project")

- No project yet: explain ("Save papers, open problems and plans; they'll appear here and stay after you close the tab") with buttons to Start and Search.
- **Header:** editable title (`st.text_input` + "Rename"), topic, created/updated dates, counts; signed in → a project switcher `st.selectbox` over `repo.list_projects(user_id)` and "New project".
- **Tabs:**
  1. **Papers:** a `st.data_editor` table with columns title (link), year, status (`SelectboxColumn` over to_read/reading/read/key), tags (`ListColumn`, or a text column with comma-separated tags), note (text). Filters by status/tag. "Apply changes" writes the diffs through `repo.update_paper`. Export BibTeX/CSV (`scholargrid.bibtex`). Remove selected.
  2. **Open problems:** saved gap cards rendered compactly with `ui.gap_card(compact=True)`, a status selector (considering/shortlisted/chosen/rejected), a note, and "Make a plan". A **Compare** view: pick 2–3 gaps → side-by-side columns with statement, evidence count, novelty, feasibility, scores and your notes.
  3. **Plans:** a list of saved plans (label, gap title, date), with view (re-uses the plan renderer from `views/plan.py`, factored into `ui.render_plan(doc)`), Markdown export and delete.
  4. **Notes:** a journal (newest first); add a note (`st.text_area` + "Add"); edit inline. Render notes as plain text (`st.text` or `st.markdown` of escaped text), **never** `unsafe_allow_html`.
  5. **Settings:** share link (anonymous or owner), rotate link, export JSON, import JSON (`st.file_uploader`, ≤ 2 MB, validated), claim into account (signed in + link present), delete project (type the title to confirm).
- Set `j_project = True` when the project has ≥ 1 gap or plan.

---

## 11. CLI: `python -m scholargrid.persistence`

```text
python -m scholargrid.persistence upgrade               # alembic upgrade head (uses SCHOLARGRID_DB_URL)
python -m scholargrid.persistence stats                 # projects (owned/anonymous), items, last 7 days activity
python -m scholargrid.persistence purge [--days 365]    # delete anonymous projects not updated since
python -m scholargrid.persistence export <project_id> --out p.json   # admin export (no token needed)
```

---

## 12. Tests: `tests/test_persistence.py`

Use a fresh SQLite file per test (`make_config(tmp_path)`) and call `upgrade_to_head(cfg)` in a fixture, so the migrations themselves are tested.

1. **Create and access:** an anonymous project gets a token of ≥ 40 chars. With the right token → access; with the wrong token → `AccessDenied`; with a signed-in owner → access without a token.
2. **Papers:** add (idempotent: a second add returns False), update status/tags/note (an invalid status raises `ValueError`), remove, and the item limit is enforced.
3. **Gaps/plans/notes:** round-trip payloads (GapCard and PlanDocument dicts from earlier plans' fixtures). Status updates work.
4. **Export/import:** export → import as a new project → same content, different id and token. A malformed import raises `ValueError`.
5. **Claim:** an anonymous project is claimed by a user and shows up in `list_projects`, and the old token still works until `rotate_token`.
6. **Purge:** set `updated_at` back 400 days → purged; owned projects are never purged.
7. **Migration sanity:** `alembic upgrade head` on an empty DB creates all tables, and a second run is a no-op.
8. **App:**
   - Add `"project"` to `PAGES`.
   - Set `at.session_state["proj_id"]` / `["proj_token"]` for a project created in the test DB with 2 papers → the page shows both titles.
   - `persistence.enabled=false` → the page renders the session fallback without errors.
   - Saving from the home results (`toggle`) creates a project and writes the paper (check via the repo).

Postgres is covered by the CI `postgres` service container on GitHub (plan 11 §10). Locally, tests use SQLite only: no Docker and no local Postgres (00 §4.7; deferred in `LATER_NOT_NOW.md`). To try the real hosted database by hand, point `SCHOLARGRID_DB_URL` at your free Neon database and run `pytest tests/test_persistence.py` once.

---

## 13. Acceptance checks

```bash
ruff check . && mypy && pytest -q
python -m scholargrid.persistence upgrade
streamlit run app/streamlit_app.py
#  Search → Save a paper → URL gains ?p=…&k=… and the "Bookmark this page" notice appears
#  Open problems → Save a card; Plan → Save plan
#  Close the browser, reopen the bookmarked URL → everything is in My project
#  My project → change a paper's status, add a note, export JSON, import it again
python -m scholargrid.persistence stats
```

## 14. Pitfalls

- **Ephemeral disks:** Hugging Face Spaces resets local disk on restart, so the SQLite file is lost there. Production must use an external **free** Postgres (Neon) via `SCHOLARGRID_DB_URL` (plan 11). SQLite is for local and dev use. Paid persistent storage on Spaces is deferred (`LATER_NOT_NOW.md`).
- Tokens are secrets: never log full URLs with `k=`; redact them in Sentry breadcrumbs (plan 11).
- Streamlit reruns: wrap repo reads in small cached functions keyed by `(project_id, _proj_rev)`, so pages don't hit the DB on every widget change. Invalidate by bumping `_proj_rev`.
- SQLite + threads: `check_same_thread=False` and short transactions are fine for one-instance dev use.
- Don't store abstracts in `payload` for every paper if you have thousands. A title, URL and minimal metadata are enough; the abstract can be refetched.
