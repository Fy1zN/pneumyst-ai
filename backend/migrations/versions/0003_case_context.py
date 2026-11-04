"""Optional de-identified case context entered with an upload.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-04
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("analyses", sa.Column("case_context", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("analyses", "case_context")