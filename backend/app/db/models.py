"""Database tables. Times are stored as UTC without a zone and read back as UTC-aware datetimes."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, BigInteger, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    impl = DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect):
        # MySQL DATETIME drops microseconds by default, which would make history ordering ambiguous.
        return dialect.type_descriptor(mysql.DATETIME(fsp=6) if dialect.name == "mysql" else DateTime())

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is not None and value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        return value.replace(tzinfo=timezone.utc) if value is not None else None


class Base(DeclarativeBase):
    pass


class UserRecord(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str | None] = mapped_column(String(320))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class AnalysisRecord(Base):
    __tablename__ = "analyses"
    __table_args__ = (Index("ix_analyses_user_created", "user_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    status: Mapped[str] = mapped_column(String(16))
    image_sha256: Mapped[str] = mapped_column(String(64))
    image_format: Mapped[str] = mapped_column(String(8))
    findings: Mapped[list] = mapped_column(JSON)
    stages: Mapped[dict] = mapped_column(JSON)
    errors: Mapped[dict] = mapped_column(JSON)
    models: Mapped[list] = mapped_column(JSON)
    preprocess: Mapped[dict] = mapped_column(JSON)
    timings_ms: Mapped[dict] = mapped_column(JSON)
    heatmap_keys: Mapped[dict | None] = mapped_column(JSON)
    report: Mapped[str | None] = mapped_column(Text)
    report_source: Mapped[str | None] = mapped_column(String(16))


class AuditEventRecord(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True,
                                    autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(String(36), ForeignKey("analyses.id", ondelete="CASCADE"),
                                             index=True)
    step: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    error_code: Mapped[str | None] = mapped_column(String(64))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)