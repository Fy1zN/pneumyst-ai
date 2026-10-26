from __future__ import annotations

from fastapi import Request

from backend.app.repositories.analyses import AnalysisStore
from backend.app.services.screening import ScreeningService


def get_screening(request: Request) -> ScreeningService:
    return request.app.state.screening


def get_store(request: Request) -> AnalysisStore | None:
    return request.app.state.store


def get_current_user(request: Request) -> str:
    # Replaced by Clerk token verification when authentication is added.
    return request.app.state.settings.dev_user_id