"""Create users, analyses and audit_events.

Revision ID: 0001
Revises:
Create Date: 2026-10-26
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from backend.app.db.models import UTCDateTime

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("email", sa.String(320), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
    )
    op.create_table(
        "analyses",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(64), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("image_sha256", sa.String(64), nullable=False),
        sa.Column("image_format", sa.String(8), nullable=False),
        sa.Column("findings", sa.JSON(), nullable=False),
        sa.Column("stages", sa.JSON(), nullable=False),
        sa.Column("errors", sa.JSON(), nullable=False),
        sa.Column("models", sa.JSON(), nullable=False),
        sa.Column("preprocess", sa.JSON(), nullable=False),
        sa.Column("timings_ms", sa.JSON(), nullable=False),
        sa.Column("heatmap_keys", sa.JSON(), nullable=True),
        sa.Column("report", sa.Text(), nullable=True),
        sa.Column("report_source", sa.String(16), nullable=True),
    )
    op.create_index("ix_analyses_user_created", "analyses", ["user_id", "created_at"])
    op.create_table(
        "audit_events",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("analysis_id", sa.String(36), sa.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("step", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
    )
    op.create_index("ix_audit_events_analysis_id", "audit_events", ["analysis_id"])


def downgrade() -> None:
    op.drop_index("ix_audit_events_analysis_id", table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_index("ix_analyses_user_created", table_name="analyses")
    op.drop_table("analyses")
    op.drop_table("users")