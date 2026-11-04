from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from backend.app.agent.facts import build_facts
from backend.app.agent.report_graph import DISCLAIMER, ReportWriter
from backend.app.api.deps import get_current_user, get_report_writer, get_storage, get_store
from backend.app.core.errors import ApiError
from backend.app.db.models import AnalysisRecord, utcnow
from backend.app.reporting.audit import build_audit
from backend.app.reporting.pdf import build_pdf
from backend.app.repositories.analyses import AnalysisStore
from backend.app.schemas.analysis import (
    AnalysisPage,
    AnalysisResponse,
    AnalysisSummary,
    CaseContext,
    ErrorResponse,
    FlaggedFinding,
    ReportResponse,
)
from backend.app.storage.files import LocalStorage
from backend.app.storage.images import image_links, remove_images

logger = logging.getLogger("pneumyst.history")
router = APIRouter(prefix="/analyses", tags=["history"])
ERRORS = {code: {"model": ErrorResponse} for code in (404, 503)}
DOWNLOAD = Query(False, description="Save as a file instead of opening in the browser")
BAND_ORDER = {"likely": 0, "possible": 1}


async def call_store(store: AnalysisStore | None, method: str, *args):
    if store is None:
        raise ApiError(503, "DATABASE_UNAVAILABLE", "the database is not configured")
    try:
        return await run_in_threadpool(getattr(store, method), *args)
    except SQLAlchemyError:
        raise ApiError(503, "DATABASE_UNAVAILABLE", "the database could not be reached")


def require_uuid(analysis_id: str) -> str:
    try:
        return str(uuid.UUID(analysis_id))
    except ValueError:
        raise ApiError(404, "ANALYSIS_NOT_FOUND", "no such analysis", analysis_id)


def summary(row: AnalysisRecord) -> AnalysisSummary:
    flagged = sorted((f for f in row.findings if f["band"] in BAND_ORDER), key=lambda f: BAND_ORDER[f["band"]])
    return AnalysisSummary(id=row.id, created_at=row.created_at, status=row.status, image_format=row.image_format,
                           flagged=[FlaggedFinding(finding=f["finding"], display_name=f["display_name"],
                                                   band=f["band"]) for f in flagged],
                           case_number=(row.case_context or {}).get("case_number"))


@router.get("", response_model=AnalysisPage, responses=ERRORS)
async def history(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                  store: AnalysisStore | None = Depends(get_store),
                  user_id: str = Depends(get_current_user)) -> AnalysisPage:
    rows, total = await call_store(store, "history", user_id, limit, offset)
    return AnalysisPage(items=[summary(r) for r in rows], total=total, limit=limit, offset=offset)


@router.get("/{analysis_id}", response_model=AnalysisResponse, responses=ERRORS)
async def get_analysis(analysis_id: str, request: Request, store: AnalysisStore | None = Depends(get_store),
                       storage: LocalStorage | None = Depends(get_storage),
                       user_id: str = Depends(get_current_user)) -> AnalysisResponse:
    row = await call_store(store, "get", user_id, require_uuid(analysis_id))
    if row is None:
        raise ApiError(404, "ANALYSIS_NOT_FOUND", "no such analysis", analysis_id)
    return AnalysisResponse(
        id=row.id, created_at=row.created_at, image_sha256=row.image_sha256, image_format=row.image_format,
        status=row.status, stages=row.stages, errors=row.errors, findings=row.findings, models=row.models,
        preprocess=row.preprocess, timings_ms=row.timings_ms, persistence="saved",
        images=image_links(storage, row.heatmap_keys, str(request.base_url)),
        case_context=CaseContext(**row.case_context) if row.case_context else None)


async def load_report(analysis_id: str, store: AnalysisStore | None, writer: ReportWriter,
                      user_id: str) -> tuple[AnalysisRecord, ReportResponse]:
    """The analysis and its summary, written on first request and stored so it never calls the LLM again."""
    analysis_id = require_uuid(analysis_id)
    row = await call_store(store, "get", user_id, analysis_id)
    if row is not None and row.report is None:
        result = await run_in_threadpool(writer.write, build_facts(row.findings, row.errors))
        row = await call_store(store, "save_report", user_id, analysis_id, result)
        if row is not None and row.report is None:
            return row, ReportResponse(analysis_id=row.id, summary=result.text, source=result.source, model=None,
                                       created_at=utcnow(), disclaimer=DISCLAIMER)
    if row is None:
        raise ApiError(404, "ANALYSIS_NOT_FOUND", "no such analysis", analysis_id)
    return row, ReportResponse(analysis_id=row.id, summary=row.report, source=row.report_source,
                               model=row.report_model, created_at=row.report_created_at, disclaimer=DISCLAIMER)


def stored_images(storage: LocalStorage | None, row: AnalysisRecord) -> dict[str, bytes] | None:
    keys = row.heatmap_keys
    if storage is None or not keys:
        return None
    try:
        return {"image": storage.read(keys["image"]),
                **{name: storage.read(key) for name, key in keys["heatmaps"].items()}}
    except (OSError, ValueError):
        logger.exception("images of analysis %s could not be read", row.id)
        return None


def disposition(row: AnalysisRecord, suffix: str, download: bool) -> dict[str, str]:
    name = (row.case_context or {}).get("case_number") or row.id[:8]
    return {"Content-Disposition": f'{"attachment" if download else "inline"}; filename="pneumyst-{name}{suffix}"'}


@router.get("/{analysis_id}/report", response_model=ReportResponse, responses=ERRORS)
async def get_report(analysis_id: str, store: AnalysisStore | None = Depends(get_store),
                     writer: ReportWriter = Depends(get_report_writer),
                     user_id: str = Depends(get_current_user)) -> ReportResponse:
    _, report = await load_report(analysis_id, store, writer, user_id)
    return report


@router.get("/{analysis_id}/report.pdf", response_class=Response,
            responses={**ERRORS, 200: {"content": {"application/pdf": {}}}})
async def report_pdf(analysis_id: str, request: Request, download: bool = DOWNLOAD,
                     store: AnalysisStore | None = Depends(get_store),
                     storage: LocalStorage | None = Depends(get_storage),
                     writer: ReportWriter = Depends(get_report_writer),
                     user_id: str = Depends(get_current_user)) -> Response:
    row, report = await load_report(analysis_id, store, writer, user_id)
    images = await run_in_threadpool(stored_images, storage, row)
    pdf = await run_in_threadpool(build_pdf, row, report.model_dump(), images, request.app.version, utcnow())
    return Response(pdf, media_type="application/pdf", headers=disposition(row, "-report.pdf", download))


@router.get("/{analysis_id}/audit.json", responses=ERRORS)
async def audit_json(analysis_id: str, request: Request, download: bool = DOWNLOAD,
                     store: AnalysisStore | None = Depends(get_store),
                     user_id: str = Depends(get_current_user)) -> JSONResponse:
    analysis_id = require_uuid(analysis_id)
    row = await call_store(store, "get", user_id, analysis_id)
    if row is None:
        raise ApiError(404, "ANALYSIS_NOT_FOUND", "no such analysis", analysis_id)
    events = await call_store(store, "audit_events", user_id, analysis_id)
    return JSONResponse(build_audit(row, events, request.app.version, utcnow()),
                        headers=disposition(row, "-audit.json", download))


@router.delete("/{analysis_id}", status_code=204, response_class=Response, responses=ERRORS)
async def delete_analysis(analysis_id: str, store: AnalysisStore | None = Depends(get_store),
                          storage: LocalStorage | None = Depends(get_storage),
                          user_id: str = Depends(get_current_user)) -> Response:
    analysis_id = require_uuid(analysis_id)
    if not await call_store(store, "delete", user_id, analysis_id):
        raise ApiError(404, "ANALYSIS_NOT_FOUND", "no such analysis", analysis_id)
    if storage is not None:
        # The row is gone, so a file that cannot be removed is only logged; the user sees the deletion succeed.
        await run_in_threadpool(remove_images, storage, analysis_id)
    return Response(status_code=204)