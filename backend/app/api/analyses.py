from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.exc import SQLAlchemyError

from backend.app.api.deps import get_current_user, get_storage, get_store
from backend.app.core.errors import ApiError
from backend.app.db.models import AnalysisRecord
from backend.app.repositories.analyses import AnalysisStore
from backend.app.schemas.analysis import AnalysisPage, AnalysisResponse, AnalysisSummary, ErrorResponse, FlaggedFinding
from backend.app.storage.files import LocalStorage
from backend.app.storage.images import image_links, remove_images

router = APIRouter(prefix="/analyses", tags=["history"])
ERRORS = {code: {"model": ErrorResponse} for code in (404, 503)}
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
                                                   band=f["band"]) for f in flagged])


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
        images=image_links(storage, row.heatmap_keys, str(request.base_url)))


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