from __future__ import annotations

from fastapi import Request

from backend.app.services.screening import ScreeningService


def get_screening(request: Request) -> ScreeningService:
    return request.app.state.screening