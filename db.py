from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
	pass


class Resume(Base):
	__tablename__ = "resumes"
	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	filename: Mapped[str] = mapped_column(String(512))
	content: Mapped[str] = mapped_column(Text)
	parsed_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
	created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class JobDescription(Base):
	__tablename__ = "job_descriptions"
	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	title: Mapped[str] = mapped_column(String(256))
	content: Mapped[str] = mapped_column(Text)
	parsed_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
	created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MatchResult(Base):
	__tablename__ = "match_results"
	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	jd_id: Mapped[int] = mapped_column(Integer)
	resume_id: Mapped[int] = mapped_column(Integer)
	tfidf_similarity: Mapped[Optional[float]] = mapped_column(nullable=True)
	sbert_similarity: Mapped[Optional[float]] = mapped_column(nullable=True)
	skills_match: Mapped[Optional[float]] = mapped_column(nullable=True)
	experience_match: Mapped[Optional[float]] = mapped_column(nullable=True)
	education_match: Mapped[Optional[float]] = mapped_column(nullable=True)
	overall_match: Mapped[Optional[float]] = mapped_column(nullable=True)
	created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def get_engine(db_url: str = "sqlite:///app.db"):
	return create_engine(db_url, future=True)


def get_session(db_url: str = "sqlite:///app.db"):
	engine = get_engine(db_url)
	Base.metadata.create_all(engine)
	return sessionmaker(bind=engine, expire_on_commit=False)()


