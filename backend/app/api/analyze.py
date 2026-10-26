from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.exc import SQLAlchemyError

from backend.app.api.deps import get_current_user, get_screening, get_store
from backend.app.core.errors import ApiError
from backend.app.repositories.analyses import AnalysisStore
from backend.app.schemas.analysis import AnalysisResponse, ErrorResponse
from backend.app.services.screening import Analysis, ScreeningService

logger = logging.getLogger("pneumyst.analyze")
router = APIRouter(tags=["analysis"])
ERRORS = {code: {"model": ErrorResponse} for code in (413, 422, 500, 503)}


@router.post("/analyze", response_model=AnalysisResponse, responses=ERRORS)
async def analyze(file: UploadFile = File(..., description="Chest X-ray, PNG or JPEG"),
                  explain: bool = Query(True, description="Compute heatmaps for possible and likely findings"),
                  screening: ScreeningService = Depends(get_screening),
                  store: AnalysisStore | None = Depends(get_store),
                  user_id: str = Depends(get_current_user)) -> AnalysisResponse:
    analysis_id = str(uuid.uuid4())
    # Read one byte past the limit so an oversized upload is detected without loading all of it.
    data = await file.read(screening.settings.max_upload_bytes + 1)
    try:
        analysis = await screening.analyze(analysis_id, data, explain)
    except ApiError as exc:
        exc.analysis_id = exc.analysis_id or analysis_id
        logger.info("analysis %s rejected: %s %s", analysis_id, exc.error_code, exc.detail)
        raise
    except Exception:
        logger.exception("analysis %s failed", analysis_id)
        raise ApiError(500, "INTERNAL_ERROR", "unexpected server error", analysis_id)

    models = [{"name": m.name, "checkpoint_sha256": m.checkpoint_sha256} for m in screening.models]
    persistence_error = None
    if store is None:
        persistence_error = "DATABASE_NOT_CONFIGURED"
    else:
        try:
            await run_in_threadpool(store.save, user_id, analysis, models)
        except SQLAlchemyError:
            logger.exception("analysis %s could not be saved", analysis_id)
            persistence_error = "STORAGE_FAILED"
    logger.info("analysis %s %s in %.0f ms, %s", analysis_id, analysis.result.status,
                analysis.result.timings_ms.get("total", 0.0), persistence_error or "saved")
    return to_response(analysis, models, persistence_error)


def to_response(analysis: Analysis, models: list[dict], persistence_error: str | None) -> AnalysisResponse:
    result = analysis.result.to_dict()
    errors = dict(result["errors"])
    if persistence_error:
        errors["persistence"] = persistence_error
    return AnalysisResponse(
        id=analysis.id, created_at=analysis.created_at, image_sha256=analysis.image_sha256,
        image_format=analysis.image_format, status=result["status"], stages=result["stages"], errors=errors,
        findings=result["findings"], models=models, preprocess=result["preprocess"],
        timings_ms=result["timings_ms"], persistence="not_saved" if persistence_error else "saved")