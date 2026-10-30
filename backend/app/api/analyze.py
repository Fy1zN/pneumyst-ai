from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.exc import SQLAlchemyError

from backend.app.api.deps import get_current_user, get_screening, get_storage, get_store
from backend.app.core.errors import ApiError
from backend.app.repositories.analyses import AnalysisStore
from backend.app.schemas.analysis import AnalysisResponse, ErrorResponse
from backend.app.services.screening import Analysis, ScreeningService
from backend.app.storage.files import LocalStorage
from backend.app.storage.images import STAGE, image_links, remove_images, save_images

logger = logging.getLogger("pneumyst.analyze")
router = APIRouter(tags=["analysis"])
ERRORS = {code: {"model": ErrorResponse} for code in (413, 422, 500, 503)}


@router.post("/analyze", response_model=AnalysisResponse, responses=ERRORS)
async def analyze(request: Request, file: UploadFile = File(..., description="Chest X-ray, PNG or JPEG"),
                  explain: bool = Query(True, description="Compute heatmaps for possible and likely findings"),
                  screening: ScreeningService = Depends(get_screening),
                  store: AnalysisStore | None = Depends(get_store),
                  storage: LocalStorage | None = Depends(get_storage),
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
    persistence_error, keys = await persist(analysis, models, user_id, store, storage)
    logger.info("analysis %s %s in %.0f ms, %s", analysis_id, analysis.result.status,
                analysis.result.timings_ms.get("total", 0.0), persistence_error or "saved")
    return to_response(analysis, models, persistence_error, image_links(storage, keys, str(request.base_url)))


async def persist(analysis: Analysis, models: list[dict], user_id: str, store: AnalysisStore | None,
                  storage: LocalStorage | None) -> tuple[str | None, dict]:
    """Save images, then the database row. Images are only kept when the row that points to them is saved."""
    if store is None:
        return "DATABASE_NOT_CONFIGURED", {}
    res = analysis.result
    keys: dict = {}
    if storage is None:
        res.stages[STAGE], res.errors[STAGE] = "skipped", "STORAGE_NOT_CONFIGURED"
    else:
        keys = await run_in_threadpool(save_images, storage, analysis)
    try:
        await run_in_threadpool(store.save, user_id, analysis, models, keys)
    except SQLAlchemyError:
        logger.exception("analysis %s could not be saved", analysis.id)
        if storage is not None:
            await run_in_threadpool(remove_images, storage, analysis.id)
        for part in (res.stages, res.errors, res.timings_ms):
            part.pop(STAGE, None)
        return "STORAGE_FAILED", {}
    return None, keys


def to_response(analysis: Analysis, models: list[dict], persistence_error: str | None,
                images: dict | None) -> AnalysisResponse:
    result = analysis.result.to_dict()
    errors = dict(result["errors"])
    if persistence_error:
        errors["persistence"] = persistence_error
    return AnalysisResponse(
        id=analysis.id, created_at=analysis.created_at, image_sha256=analysis.image_sha256,
        image_format=analysis.image_format, status=result["status"], stages=result["stages"], errors=errors,
        findings=result["findings"], models=models, preprocess=result["preprocess"],
        timings_ms=result["timings_ms"], persistence="not_saved" if persistence_error else "saved", images=images)