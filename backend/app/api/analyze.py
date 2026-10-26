from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, File, Query, UploadFile

from backend.app.api.deps import get_screening
from backend.app.core.errors import ApiError
from backend.app.schemas.analysis import AnalysisResponse, ErrorResponse, ModelRef
from backend.app.services.screening import Analysis, ScreeningService

logger = logging.getLogger("pneumyst.analyze")
router = APIRouter(tags=["analysis"])
ERRORS = {code: {"model": ErrorResponse} for code in (413, 422, 500, 503)}


@router.post("/analyze", response_model=AnalysisResponse, responses=ERRORS)
async def analyze(file: UploadFile = File(..., description="Chest X-ray, PNG or JPEG"),
                  explain: bool = Query(True, description="Compute heatmaps for possible and likely findings"),
                  screening: ScreeningService = Depends(get_screening)) -> AnalysisResponse:
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
    logger.info("analysis %s %s in %.0f ms", analysis_id, analysis.result.status,
                analysis.result.timings_ms.get("total", 0.0))
    return to_response(analysis, screening)


def to_response(analysis: Analysis, screening: ScreeningService) -> AnalysisResponse:
    result = analysis.result.to_dict()
    return AnalysisResponse(
        id=analysis.id, created_at=analysis.created_at, image_sha256=analysis.image_sha256,
        image_format=analysis.image_format, status=result["status"], stages=result["stages"],
        errors=result["errors"], findings=result["findings"],
        models=[ModelRef(name=m.name, checkpoint_sha256=m.checkpoint_sha256) for m in screening.models],
        preprocess=result["preprocess"], timings_ms=result["timings_ms"])