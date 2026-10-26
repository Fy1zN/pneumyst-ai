from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from backend.app.core.config import load_settings
from backend.app.db.models import Base

if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name, disable_existing_loggers=False)

URL = load_settings().database_url
if not URL:
    raise RuntimeError("DATABASE_URL is not set. Add it to .env before running migrations.")


def run_offline() -> None:
    context.configure(url=URL, target_metadata=Base.metadata, literal_binds=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    engine = create_engine(URL, poolclass=NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()