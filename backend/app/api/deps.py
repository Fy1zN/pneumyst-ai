from __future__ import annotations

import logging

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app.auth.clerk import AuthUnavailable, TokenError
from backend.app.core.errors import ApiError
from backend.app.repositories.analyses import AnalysisStore
from backend.app.services.screening import ScreeningService

logger = logging.getLogger("pneumyst.auth")
bearer = HTTPBearer(auto_error=False, description="Clerk session token")


def get_screening(request: Request) -> ScreeningService:
    return request.app.state.screening


def get_store(request: Request) -> AnalysisStore | None:
    return request.app.state.store


def get_current_user(request: Request,
                     credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> str:
    settings = request.app.state.settings
    if settings.auth_mode == "dev":
        return settings.dev_user_id
    if credentials is None:
        raise ApiError(401, "UNAUTHENTICATED", "sign in required")
    try:
        claims = request.app.state.verifier.verify(credentials.credentials)
    except AuthUnavailable:
        logger.exception("Clerk signing keys could not be fetched")
        raise ApiError(503, "AUTH_UNAVAILABLE", "sign-in could not be checked, try again shortly")
    except TokenError as exc:
        logger.info("rejected token: %s", exc)
        raise ApiError(401, "UNAUTHENTICATED", "invalid or expired session")
    return claims["sub"]