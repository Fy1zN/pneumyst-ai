from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import get_screening
from backend.app.schemas.analysis import FindingConfig, HealthResponse, ModelStatus
from backend.app.services.screening import ScreeningService
from ml.datasets.targets import DISPLAY_NAMES

router = APIRouter(tags=["service"])


@router.get("/health", response_model=HealthResponse)
def health(screening: ScreeningService = Depends(get_screening)) -> HealthResponse:
    available = {m.name: m.available for m in screening.models}
    status = "ok" if all(available.values()) else ("degraded" if any(available.values()) else "down")
    return HealthResponse(status=status, device=screening.device, models=available)


@router.get("/models", response_model=list[ModelStatus])
def models(screening: ScreeningService = Depends(get_screening)) -> list[ModelStatus]:
    return [ModelStatus(
        name=m.name, available=m.available, error=m.error, checkpoint_sha256=m.checkpoint_sha256,
        findings=[FindingConfig(finding=f, display_name=DISPLAY_NAMES[f], band_status=m.calibration[f]["status"],
                                possible_threshold=m.calibration[f].get("possible_threshold"),
                                likely_threshold=m.calibration[f].get("likely_threshold"))
                  for f in m.findings])
        for m in screening.models]