"""Record which model wrote each report and when.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-31
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from backend.app.db.models import UTCDateTime

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("analyses", sa.Column("report_model", sa.String(64), nullable=True))
    op.add_column("analyses", sa.Column("report_created_at", UTCDateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("analyses", "report_created_at")
    op.drop_column("analyses", "report_model")