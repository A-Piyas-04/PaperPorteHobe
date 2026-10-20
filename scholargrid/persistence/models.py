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


def _token() -> str:
    return secrets.token_urlsafe(32)


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
    access_token: Mapped[str] = mapped_column(String(64), unique=True, default=_token)
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
    payload: Mapped[dict] = mapped_column(JsonType, default=dict)          # authors, venue, doi, ...
    source: Mapped[str] = mapped_column(String(30), default="")            # search|reading_path|gap_evidence|...
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


class Rating(Base):
    __tablename__ = "ratings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rater: Mapped[str] = mapped_column(String(80), index=True)            # rater code or user id
    item_type: Mapped[str] = mapped_column(String(30), index=True)        # gap_card|problem|facets|search|plan|reading
    item_id: Mapped[str] = mapped_column(String(80), index=True)
    batch: Mapped[str] = mapped_column(String(80), default="", index=True)  # evaluation round name
    context: Mapped[dict] = mapped_column(JsonType, default=dict)          # topic, scope, bundle_version, model
    criteria: Mapped[dict] = mapped_column(JsonType, default=dict)         # rubric answers
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    session: Mapped[str] = mapped_column(String(32), index=True)          # random per browser session
    user_id: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    project_id: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(50), index=True)
    props: Mapped[dict] = mapped_column(JsonType, default=dict)


class LlmUsageDaily(Base):
    __tablename__ = "llm_usage_daily"
    __table_args__ = (UniqueConstraint("day", "scope"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    day: Mapped[str] = mapped_column(String(10), index=True)        # YYYY-MM-DD in America/Los_Angeles (Gemini reset)
    scope: Mapped[str] = mapped_column(String(80), index=True)      # "global" or "session:<sid>"
    requests: Mapped[int] = mapped_column(Integer, default=0)


class ProjectNote(Base):
    __tablename__ = "project_notes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    body: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
    project: Mapped[Project] = relationship(back_populates="notes")
