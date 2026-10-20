"""ProjectRepo: every public method takes an ``Access`` (who is asking) and authorizes in one place.

Access is either a signed-in owner (``user_id``) or a capability link (``project_id`` + ``token``). Tokens
are secrets: compared with ``secrets.compare_digest``, never logged, and returned only on creation,
rotation, or to the owner / link holder for the share-link display. Methods return plain dicts.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import func, select

from ..config import Config
from .db import session_factory
from .models import Event, LlmUsageDaily, Project, ProjectGap, ProjectNote, ProjectPaper, ProjectPlan, Rating, User

PAPER_STATUSES = ("to_read", "reading", "read", "key")
GAP_STATUSES = ("considering", "shortlisted", "chosen", "rejected")
EXPORT_SCHEMA = 1
_MAX_TITLE = 200
_MAX_TAGS = 20
_MAX_TAG_CHARS = 40
_PAYLOAD_KEYS = ("arxiv_id", "doi", "authors", "venue", "date", "primary_category", "cited_by_count", "sources",
                 "url")


@dataclass(frozen=True)
class Access:
    user_id: Optional[str] = None       # signed-in user
    project_id: Optional[str] = None    # capability link
    token: Optional[str] = None


class AccessDenied(PermissionError):
    pass


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: Optional[datetime]) -> str:
    return dt.isoformat(timespec="seconds") if dt else ""


def _year(value: Any) -> Optional[int]:
    s = str(value or "")[:4]
    return int(s) if s.isdigit() else None


class ProjectRepo:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.Session = session_factory(cfg)
        self.p = cfg["persistence"]

    # -- validation ----------------------------------------------------------------------
    def _title(self, title: str) -> str:
        t = " ".join(str(title or "").split())
        if len(t) > _MAX_TITLE:
            raise ValueError(f"Titles can be at most {_MAX_TITLE} characters.")
        return t or "My research project"

    def _note(self, body: str) -> str:
        body = str(body or "")
        if len(body) > int(self.p["max_note_chars"]):
            raise ValueError(f"Notes can be at most {self.p['max_note_chars']:,} characters.")
        return body

    def _tags(self, tags: Iterable[str]) -> List[str]:
        out = [" ".join(str(t).split()) for t in tags if str(t).strip()]
        if len(out) > _MAX_TAGS or any(len(t) > _MAX_TAG_CHARS for t in out):
            raise ValueError(f"Use at most {_MAX_TAGS} tags of at most {_MAX_TAG_CHARS} characters each.")
        return list(dict.fromkeys(out))

    def _limit(self, s: Any, model: Any, project_id: str) -> None:
        n = s.scalar(select(func.count()).select_from(model).where(model.project_id == project_id))
        if int(n or 0) >= int(self.p["max_items_per_project"]):
            raise ValueError(f"This project already holds the maximum of {self.p['max_items_per_project']} items "
                             "of this kind.")

    # -- authorization ---------------------------------------------------------------------
    def _load(self, s: Any, project_id: Optional[str], access: Access) -> Project:
        proj = s.get(Project, project_id) if project_id else None
        if proj is None or proj.archived:
            raise AccessDenied("not found")
        if access.user_id and proj.owner_id == access.user_id:
            return proj
        if (access.token and access.project_id == project_id
                and secrets.compare_digest(str(proj.access_token), str(access.token))):
            return proj
        raise AccessDenied("no access")

    @staticmethod
    def _touch(proj: Project) -> None:
        proj.updated_at = _now()

    def _summary(self, s: Any, proj: Project) -> Dict:
        def count(model: Any) -> int:
            return int(s.scalar(select(func.count()).select_from(model).where(model.project_id == proj.id)) or 0)

        return {"id": proj.id, "title": proj.title, "topic_query": proj.topic_query, "scope_name": proj.scope_name,
                "bundle_version": proj.bundle_version, "owned": proj.owner_id is not None,
                "owner_id": proj.owner_id, "created_at": _iso(proj.created_at), "updated_at": _iso(proj.updated_at),
                "counts": {"papers": count(ProjectPaper), "gaps": count(ProjectGap), "plans": count(ProjectPlan),
                           "notes": count(ProjectNote)}}

    # -- users ----------------------------------------------------------------------------
    def upsert_user(self, *, email: str, name: str = "", provider: str = "oidc") -> Dict:
        """Find by email (case-insensitive) or create; updates last_seen_at. Returns {id, email, name}."""
        email = str(email or "").strip().lower()
        if not email:
            raise ValueError("email required")
        with self.Session.begin() as s:
            user = s.scalar(select(User).where(func.lower(User.email) == email))
            if user is None:
                user = User(email=email, name=str(name or "")[:200], provider=provider)
                s.add(user)
                s.flush()
            else:
                user.last_seen_at = _now()
                if name and not user.name:
                    user.name = str(name)[:200]
            return {"id": user.id, "email": user.email, "name": user.name}

    # -- projects -------------------------------------------------------------------------
    def create_project(self, access: Access, *, title: str, topic_query: str = "", scope_name: str = "",
                       bundle_version: str = "") -> Tuple[str, str]:
        """Returns (project_id, access_token). Enforces max_projects_per_user for signed-in users."""
        with self.Session.begin() as s:
            if access.user_id:
                n = s.scalar(select(func.count()).select_from(Project).where(Project.owner_id == access.user_id))
                if int(n or 0) >= int(self.p["max_projects_per_user"]):
                    raise ValueError(f"You already have {n} projects (the maximum). Delete one first.")
            elif not self.p["allow_anonymous_projects"]:
                raise AccessDenied("sign in to create a project")
            proj = Project(owner_id=access.user_id or None, title=self._title(title),
                           topic_query=str(topic_query or "")[:300], scope_name=str(scope_name or "")[:100],
                           bundle_version=str(bundle_version or "")[:64])
            s.add(proj)
            s.flush()
            return proj.id, proj.access_token

    def get_project(self, project_id: str, access: Access) -> Dict:
        with self.Session() as s:
            return self._summary(s, self._load(s, project_id, access))

    def share_token(self, project_id: str, access: Access) -> str:
        """The link token, for the owner's / link holder's share-link display only."""
        with self.Session() as s:
            return str(self._load(s, project_id, access).access_token)

    def list_projects(self, user_id: str) -> List[Dict]:
        with self.Session() as s:
            rows = s.scalars(select(Project).where(Project.owner_id == user_id, Project.archived.is_(False))
                             .order_by(Project.updated_at.desc())).all()
            return [self._summary(s, p) for p in rows]

    def rename(self, project_id: str, access: Access, title: str) -> None:
        with self.Session.begin() as s:
            proj = self._load(s, project_id, access)
            proj.title = self._title(title)
            self._touch(proj)

    def set_topic(self, project_id: str, access: Access, topic_query: str, bundle_version: str = "") -> None:
        with self.Session.begin() as s:
            proj = self._load(s, project_id, access)
            proj.topic_query = str(topic_query or "")[:300]
            if bundle_version:
                proj.bundle_version = str(bundle_version)[:64]
            self._touch(proj)

    def delete_project(self, project_id: str, access: Access) -> None:
        with self.Session.begin() as s:
            s.delete(self._load(s, project_id, access))

    def rotate_token(self, project_id: str, access: Access) -> str:
        with self.Session.begin() as s:
            proj = self._load(s, project_id, access)
            proj.access_token = secrets.token_urlsafe(32)
            self._touch(proj)
            return proj.access_token

    def claim(self, project_id: str, token: str, user_id: str) -> None:
        with self.Session.begin() as s:
            proj = self._load(s, project_id, Access(project_id=project_id, token=token))
            if proj.owner_id and proj.owner_id != user_id:
                raise AccessDenied("already owned")
            proj.owner_id = user_id
            self._touch(proj)

    # -- papers ---------------------------------------------------------------------------
    def add_paper(self, project_id: str, access: Access, paper: Dict, source: str = "") -> bool:
        pid = str(paper.get("paper_id") or paper.get("arxiv_id") or "").strip()
        if not pid:
            raise ValueError("paper_id required")
        with self.Session.begin() as s:
            proj = self._load(s, project_id, access)
            if s.scalar(select(ProjectPaper.id).where(ProjectPaper.project_id == project_id,
                                                      ProjectPaper.paper_id == pid)) is not None:
                return False
            self._limit(s, ProjectPaper, project_id)
            payload = {k: paper.get(k) for k in _PAYLOAD_KEYS if paper.get(k) not in (None, "")}
            s.add(ProjectPaper(project_id=project_id, paper_id=pid[:200], title=str(paper.get("title") or "")[:2000],
                               url=str(paper.get("url") or ""), year=_year(paper.get("year") or paper.get("date")),
                               payload=payload, source=str(source or "")[:30]))
            self._touch(proj)
            return True

    def update_paper(self, project_id: str, access: Access, paper_id: str, *, status: Optional[str] = None,
                     tags: Optional[Iterable[str]] = None, note: Optional[str] = None) -> None:
        if status is not None and status not in PAPER_STATUSES:
            raise ValueError(f"status must be one of {PAPER_STATUSES}")
        with self.Session.begin() as s:
            proj = self._load(s, project_id, access)
            row = s.scalar(select(ProjectPaper).where(ProjectPaper.project_id == project_id,
                                                      ProjectPaper.paper_id == paper_id))
            if row is None:
                raise ValueError("paper not in this project")
            if status is not None:
                row.status = status
            if tags is not None:
                row.tags = self._tags(tags)
            if note is not None:
                row.note = self._note(note)
            self._touch(proj)

    def remove_paper(self, project_id: str, access: Access, paper_id: str) -> None:
        with self.Session.begin() as s:
            proj = self._load(s, project_id, access)
            row = s.scalar(select(ProjectPaper).where(ProjectPaper.project_id == project_id,
                                                      ProjectPaper.paper_id == paper_id))
            if row is not None:
                s.delete(row)
            self._touch(proj)

    def papers(self, project_id: str, access: Access) -> List[Dict]:
        with self.Session() as s:
            self._load(s, project_id, access)
            rows = s.scalars(select(ProjectPaper).where(ProjectPaper.project_id == project_id)
                             .order_by(ProjectPaper.added_at, ProjectPaper.id)).all()
            return [{"paper_id": r.paper_id, "title": r.title, "url": r.url, "year": r.year, "status": r.status,
                     "tags": list(r.tags or []), "note": r.note, "payload": dict(r.payload or {}), "source": r.source,
                     "added_at": _iso(r.added_at)} for r in rows]

    # -- gaps -----------------------------------------------------------------------------
    def add_gap(self, project_id: str, access: Access, card: Dict) -> bool:
        gid = str(card.get("id") or "").strip()
        if not gid:
            raise ValueError("gap card needs an id")
        with self.Session.begin() as s:
            proj = self._load(s, project_id, access)
            if s.scalar(select(ProjectGap.id).where(ProjectGap.project_id == project_id,
                                                    ProjectGap.gap_id == gid)) is not None:
                return False
            self._limit(s, ProjectGap, project_id)
            s.add(ProjectGap(project_id=project_id, gap_id=gid[:40], payload=dict(card)))
            self._touch(proj)
            return True

    def update_gap(self, project_id: str, access: Access, gap_id: str, *, status: Optional[str] = None,
                   note: Optional[str] = None) -> None:
        if status is not None and status not in GAP_STATUSES:
            raise ValueError(f"status must be one of {GAP_STATUSES}")
        with self.Session.begin() as s:
            proj = self._load(s, project_id, access)
            row = s.scalar(select(ProjectGap).where(ProjectGap.project_id == project_id, ProjectGap.gap_id == gap_id))
            if row is None:
                raise ValueError("open problem not in this project")
            if status is not None:
                row.status = status
            if note is not None:
                row.note = self._note(note)
            self._touch(proj)

    def remove_gap(self, project_id: str, access: Access, gap_id: str) -> None:
        with self.Session.begin() as s:
            proj = self._load(s, project_id, access)
            row = s.scalar(select(ProjectGap).where(ProjectGap.project_id == project_id, ProjectGap.gap_id == gap_id))
            if row is not None:
                s.delete(row)
            self._touch(proj)

    def gaps(self, project_id: str, access: Access) -> List[Dict]:
        with self.Session() as s:
            self._load(s, project_id, access)
            rows = s.scalars(select(ProjectGap).where(ProjectGap.project_id == project_id)
                             .order_by(ProjectGap.added_at, ProjectGap.id)).all()
            return [{"gap_id": r.gap_id, "status": r.status, "note": r.note, "payload": dict(r.payload or {}),
                     "added_at": _iso(r.added_at)} for r in rows]

    # -- plans ----------------------------------------------------------------------------
    def add_plan(self, project_id: str, access: Access, plan_doc: Dict, label: str = "") -> int:
        with self.Session.begin() as s:
            proj = self._load(s, project_id, access)
            self._limit(s, ProjectPlan, project_id)
            label = label or str((plan_doc.get("plan") or {}).get("title") or "Plan")
            row = ProjectPlan(project_id=project_id, gap_id=str(plan_doc.get("gap_id") or "")[:40],
                              label=label[:200], payload=dict(plan_doc))
            s.add(row)
            s.flush()
            self._touch(proj)
            return int(row.id)

    def plans(self, project_id: str, access: Access) -> List[Dict]:
        with self.Session() as s:
            self._load(s, project_id, access)
            rows = s.scalars(select(ProjectPlan).where(ProjectPlan.project_id == project_id)
                             .order_by(ProjectPlan.created_at.desc(), ProjectPlan.id.desc())).all()
            return [{"id": r.id, "gap_id": r.gap_id, "label": r.label, "payload": dict(r.payload or {}),
                     "created_at": _iso(r.created_at)} for r in rows]

    def remove_plan(self, project_id: str, access: Access, plan_id: int) -> None:
        with self.Session.begin() as s:
            proj = self._load(s, project_id, access)
            row = s.get(ProjectPlan, int(plan_id))
            if row is not None and row.project_id == project_id:
                s.delete(row)
            self._touch(proj)

    # -- notes ----------------------------------------------------------------------------
    def add_note(self, project_id: str, access: Access, body: str) -> int:
        body = self._note(body)
        with self.Session.begin() as s:
            proj = self._load(s, project_id, access)
            self._limit(s, ProjectNote, project_id)
            row = ProjectNote(project_id=project_id, body=body)
            s.add(row)
            s.flush()
            self._touch(proj)
            return int(row.id)

    def update_note(self, project_id: str, access: Access, note_id: int, body: str) -> None:
        body = self._note(body)
        with self.Session.begin() as s:
            proj = self._load(s, project_id, access)
            row = s.get(ProjectNote, int(note_id))
            if row is None or row.project_id != project_id:
                raise ValueError("note not in this project")
            row.body = body
            self._touch(proj)

    def remove_note(self, project_id: str, access: Access, note_id: int) -> None:
        with self.Session.begin() as s:
            proj = self._load(s, project_id, access)
            row = s.get(ProjectNote, int(note_id))
            if row is not None and row.project_id == project_id:
                s.delete(row)
            self._touch(proj)

    def notes(self, project_id: str, access: Access) -> List[Dict]:
        with self.Session() as s:
            self._load(s, project_id, access)
            rows = s.scalars(select(ProjectNote).where(ProjectNote.project_id == project_id)
                             .order_by(ProjectNote.created_at.desc(), ProjectNote.id.desc())).all()
            return [{"id": r.id, "body": r.body, "created_at": _iso(r.created_at), "updated_at": _iso(r.updated_at)}
                    for r in rows]

    # -- portability ----------------------------------------------------------------------
    def _export(self, s: Any, proj: Project) -> Dict:
        def rows(model: Any, order: Any) -> List[Any]:
            return list(s.scalars(select(model).where(model.project_id == proj.id).order_by(order)).all())

        return {"schema": EXPORT_SCHEMA,
                "project": {"title": proj.title, "topic_query": proj.topic_query, "scope_name": proj.scope_name,
                            "bundle_version": proj.bundle_version, "created_at": _iso(proj.created_at)},
                "papers": [{"paper_id": r.paper_id, "title": r.title, "url": r.url, "year": r.year, "status": r.status,
                            "tags": list(r.tags or []), "note": r.note, "payload": dict(r.payload or {}),
                            "source": r.source} for r in rows(ProjectPaper, ProjectPaper.id)],
                "gaps": [{"gap_id": r.gap_id, "status": r.status, "note": r.note, "payload": dict(r.payload or {})}
                         for r in rows(ProjectGap, ProjectGap.id)],
                "plans": [{"gap_id": r.gap_id, "label": r.label, "payload": dict(r.payload or {})}
                          for r in rows(ProjectPlan, ProjectPlan.id)],
                "notes": [{"body": r.body} for r in rows(ProjectNote, ProjectNote.id)]}

    def export_project(self, project_id: str, access: Access) -> Dict:
        with self.Session() as s:
            return self._export(s, self._load(s, project_id, access))

    def admin_export(self, project_id: str) -> Dict:
        """CLI only: export without a token."""
        with self.Session() as s:
            proj = s.get(Project, project_id)
            if proj is None:
                raise AccessDenied("not found")
            return self._export(s, proj)

    def import_project(self, data: Dict, access: Access) -> Tuple[str, str]:
        if not isinstance(data, dict) or data.get("schema") != EXPORT_SCHEMA:
            raise ValueError("This file is not a ScholarGrid project export (schema 1).")
        proj_d = data.get("project")
        lists = {k: data.get(k, []) for k in ("papers", "gaps", "plans", "notes")}
        if not isinstance(proj_d, dict) or not all(isinstance(v, list) for v in lists.values()):
            raise ValueError("The project file is malformed.")
        limit = int(self.p["max_items_per_project"])
        if any(len(v) > limit for v in lists.values()):
            raise ValueError(f"The project file has more than {limit} items of one kind.")
        for p in lists["papers"]:
            if not isinstance(p, dict) or not str(p.get("paper_id") or "").strip():
                raise ValueError("Every paper needs a paper_id.")
            if p.get("status", "to_read") not in PAPER_STATUSES:
                raise ValueError(f"Unknown paper status {p.get('status')!r}.")
        for g in lists["gaps"]:
            if not isinstance(g, dict) or not str(g.get("gap_id") or "").strip() or not isinstance(g.get("payload"), dict):
                raise ValueError("Every open problem needs a gap_id and a payload.")
            if g.get("status", "considering") not in GAP_STATUSES:
                raise ValueError(f"Unknown open-problem status {g.get('status')!r}.")
        pid, token = self.create_project(access, title=str(proj_d.get("title") or "Imported project"),
                                         topic_query=str(proj_d.get("topic_query") or ""),
                                         scope_name=str(proj_d.get("scope_name") or ""),
                                         bundle_version=str(proj_d.get("bundle_version") or ""))
        with self.Session.begin() as s:
            for p in lists["papers"]:
                s.add(ProjectPaper(project_id=pid, paper_id=str(p["paper_id"])[:200], title=str(p.get("title") or ""),
                                   url=str(p.get("url") or ""), year=_year(p.get("year")),
                                   status=p.get("status", "to_read"), tags=self._tags(p.get("tags") or []),
                                   note=self._note(p.get("note") or ""), payload=dict(p.get("payload") or {}),
                                   source=str(p.get("source") or "")[:30]))
            for g in lists["gaps"]:
                s.add(ProjectGap(project_id=pid, gap_id=str(g["gap_id"])[:40], status=g.get("status", "considering"),
                                 note=self._note(g.get("note") or ""), payload=dict(g["payload"])))
            for pl in lists["plans"]:
                if isinstance(pl, dict) and isinstance(pl.get("payload"), dict):
                    s.add(ProjectPlan(project_id=pid, gap_id=str(pl.get("gap_id") or "")[:40],
                                      label=str(pl.get("label") or "")[:200], payload=dict(pl["payload"])))
            for n in lists["notes"]:
                if isinstance(n, dict):
                    s.add(ProjectNote(project_id=pid, body=self._note(n.get("body") or "")))
        return pid, token

    # -- evaluation ratings (plan 09) --------------------------------------------------------
    def add_rating(self, *, rater: str, item_type: str, item_id: str, batch: str = "", context: Optional[Dict] = None,
                   criteria: Optional[Dict] = None, comment: str = "") -> int:
        rater = " ".join(str(rater or "").split())[:80]
        if not rater:
            raise ValueError("rater code required")
        with self.Session.begin() as s:
            row = Rating(rater=rater, item_type=str(item_type)[:30], item_id=str(item_id)[:80], batch=str(batch)[:80],
                         context=dict(context or {}), criteria=dict(criteria or {}), comment=str(comment or "")[:4000])
            s.add(row)
            s.flush()
            return int(row.id)

    def ratings(self, batch: Optional[str] = None, item_type: Optional[str] = None) -> List[Dict]:
        with self.Session() as s:
            q = select(Rating)
            if batch is not None:
                q = q.where(Rating.batch == batch)
            if item_type is not None:
                q = q.where(Rating.item_type == item_type)
            rows = s.scalars(q.order_by(Rating.id)).all()
            return [{"id": r.id, "rater": r.rater, "item_type": r.item_type, "item_id": r.item_id, "batch": r.batch,
                     "context": dict(r.context or {}), "criteria": dict(r.criteria or {}), "comment": r.comment,
                     "created_at": _iso(r.created_at)} for r in rows]

    # -- telemetry (plan 09) --------------------------------------------------------------------
    def add_event(self, *, session: str, name: str, user_id: Optional[str] = None, project_id: Optional[str] = None,
                  props: Optional[Dict] = None) -> None:
        with self.Session.begin() as s:
            s.add(Event(session=str(session)[:32], name=str(name)[:50], user_id=(user_id or None),
                        project_id=(project_id or None), props=dict(props or {})))

    def events(self, since_days: int = 30, names: Optional[Iterable[str]] = None) -> List[Dict]:
        cutoff = _now() - timedelta(days=int(since_days))
        with self.Session() as s:
            q = select(Event).where(Event.ts >= cutoff)
            if names is not None:
                q = q.where(Event.name.in_(list(names)))
            rows = s.scalars(q.order_by(Event.ts)).all()
            return [{"ts": _iso(r.ts), "session": r.session, "user_id": r.user_id, "project_id": r.project_id,
                     "name": r.name, "props": dict(r.props or {})} for r in rows]

    def purge_events(self, older_than_days: int) -> int:
        cutoff = _now() - timedelta(days=int(older_than_days))
        with self.Session.begin() as s:
            rows = s.scalars(select(Event).where(Event.ts < cutoff)).all()
            for r in rows:
                s.delete(r)
            return len(rows)

    # -- online free-quota counters (plan 11) ---------------------------------------------------
    @staticmethod
    def quota_day() -> str:
        """Today in the Gemini reset timezone (midnight Pacific)."""
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")

    def requests_today(self, scope: str) -> int:
        with self.Session() as s:
            n = s.scalar(select(LlmUsageDaily.requests).where(LlmUsageDaily.day == self.quota_day(),
                                                              LlmUsageDaily.scope == scope))
            return int(n or 0)

    def add_request(self, scopes: List[str]) -> None:
        """Atomic per-(day, scope) increment (INSERT ... ON CONFLICT DO UPDATE on SQLite and Postgres)."""
        day = self.quota_day()
        with self.Session.begin() as s:
            dialect = s.get_bind().dialect.name
            if dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as _insert
            else:
                from sqlalchemy.dialects.sqlite import insert as _insert  # type: ignore[assignment]
            for scope in scopes:
                stmt = _insert(LlmUsageDaily).values(day=day, scope=str(scope)[:80], requests=1)
                stmt = stmt.on_conflict_do_update(index_elements=["day", "scope"],
                                                  set_={"requests": LlmUsageDaily.requests + 1})
                s.execute(stmt)

    # -- maintenance ----------------------------------------------------------------------
    def purge_anonymous(self, older_than_days: int) -> int:
        cutoff = _now() - timedelta(days=int(older_than_days))
        with self.Session.begin() as s:
            rows = s.scalars(select(Project).where(Project.owner_id.is_(None), Project.updated_at < cutoff)).all()
            for r in rows:
                s.delete(r)
            return len(rows)

    def stats(self) -> Dict:
        week = _now() - timedelta(days=7)
        with self.Session() as s:
            def n(q: Any) -> int:
                return int(s.scalar(q) or 0)

            return {"projects": n(select(func.count()).select_from(Project)),
                    "owned": n(select(func.count()).select_from(Project).where(Project.owner_id.is_not(None))),
                    "anonymous": n(select(func.count()).select_from(Project).where(Project.owner_id.is_(None))),
                    "users": n(select(func.count()).select_from(User)),
                    "papers": n(select(func.count()).select_from(ProjectPaper)),
                    "gaps": n(select(func.count()).select_from(ProjectGap)),
                    "plans": n(select(func.count()).select_from(ProjectPlan)),
                    "notes": n(select(func.count()).select_from(ProjectNote)),
                    "updated_last_7_days": n(select(func.count()).select_from(Project)
                                             .where(Project.updated_at >= week))}
