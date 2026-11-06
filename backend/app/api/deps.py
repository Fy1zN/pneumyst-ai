from __future__ import annotations

import logging

from fastapi import Depends, Form, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app.agent.report_graph import ReportWriter
from backend.app.agent.assistant import Assistant
from backend.app.auth.clerk import AuthUnavailable, TokenError
from backend.app.core.errors import ApiError
from backend.app.repositories.analyses import AnalysisStore
from backend.app.schemas.analysis import (
    CASE_NUMBER_PATTERN,
    AgeBand,
    CaseContext,
    CoughDuration,
    ImageSource,
    Position,
    Reason,
    Sex,
    Smoking,
    View,
    YesNo,
    YesNoUnknown,
)
from backend.app.services.screening import ScreeningService
from backend.app.storage.files import LocalStorage

logger = logging.getLogger("pneumyst.auth")
bearer = HTTPBearer(auto_error=False, description="Clerk session token")


def get_screening(request: Request) -> ScreeningService:
    return request.app.state.screening


def get_store(request: Request) -> AnalysisStore | None:
    return request.app.state.store


def get_storage(request: Request) -> LocalStorage | None:
    return request.app.state.storage


def get_report_writer(request: Request) -> ReportWriter:
    return request.app.state.report_writer

def get_assistant(request: Request) -> Assistant | None:
    return request.app.state.assistant




def case_context_form(
        case_number: str | None = Form(None, pattern=CASE_NUMBER_PATTERN,
                                       description="Your own reference such as CASE-0042, never a name"),
        age_band: AgeBand | None = Form(None), sex: Sex | None = Form(None),
        reason: Reason | None = Form(None, description="Reason for the X-ray"),
        view: View | None = Form(None), position: Position | None = Form(None),
        image_source: ImageSource | None = Form(None),
        cough: YesNo | None = Form(None, description="Current cough"),
        cough_duration: CoughDuration | None = Form(None),
        fever: YesNo | None = Form(None), night_sweats: YesNo | None = Form(None),
        weight_loss: YesNo | None = Form(None), breathlessness: YesNo | None = Form(None),
        chest_pain: YesNo | None = Form(None), coughing_blood: YesNo | None = Form(None),
        tb_contact: YesNoUnknown | None = Form(None, description="Close contact with a TB patient"),
        previous_tb: YesNoUnknown | None = Form(None, description="Treated for TB before"),
        diabetes: YesNoUnknown | None = Form(None), smoking: Smoking | None = Form(None)) -> CaseContext:
    """Optional case context sent as form fields next to the upload; Swagger shows each as a dropdown."""
    return CaseContext(**{name: value for name, value in locals().items() if value is not None})


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