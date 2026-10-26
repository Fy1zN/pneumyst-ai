from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker


class Database:
    def __init__(self, url: str, **engine_kwargs):
        if url.startswith("mysql"):
            engine_kwargs.setdefault("pool_recycle", 3600)  # MySQL closes idle connections after 8 hours
            engine_kwargs.setdefault("connect_args", {"connect_timeout": 5})
        self.engine = create_engine(url, pool_pre_ping=True, **engine_kwargs)
        self.session = sessionmaker(self.engine, expire_on_commit=False)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False

    def dispose(self) -> None:
        self.engine.dispose()