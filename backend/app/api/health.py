from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from backend.app.api.deps import get_current_user, get_screening, get_store
from backend.app.repositories.analyses import AnalysisStore
from backend.app.schemas.analysis import FindingConfig, HealthResponse, ModelStatus
from backend.app.services.screening import ScreeningService
from ml.datasets.targets import DISPLAY_NAMES

router = APIRouter(tags=["service"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request, screening: ScreeningService = Depends(get_screening),
           store: AnalysisStore | None = Depends(get_store)) -> HealthResponse:
    available = {m.name: m.available for m in screening.models}
    database = "not_configured" if store is None else ("ok" if store.db.ping() else "unavailable")
    if not any(available.values()):
        status = "down"
    elif all(available.values()) and database == "ok":
        status = "ok"
    else:
        status = "degraded"
    state = request.app.state
    llm = state.assistant is not None
    # Configuration only, never a key or URL: enough to spot a missing LLM or dev auth left on after deploy.
    return HealthResponse(status=status, version=request.app.version, device=screening.device, models=available,
                          database=database, storage="configured" if state.storage else "not_configured",
                          llm="configured" if llm else "not_configured",
                          llm_provider=state.settings.llm_provider if llm else None,
                          auth_mode=state.settings.auth_mode)


@router.get("/models", response_model=list[ModelStatus], dependencies=[Depends(get_current_user)])
def models(screening: ScreeningService = Depends(get_screening)) -> list[ModelStatus]:
    return [ModelStatus(
        name=m.name, available=m.available, error=m.error, checkpoint_sha256=m.checkpoint_sha256,
        findings=[FindingConfig(finding=f, display_name=DISPLAY_NAMES[f], band_status=m.calibration[f]["status"],
                                possible_threshold=m.calibration[f].get("possible_threshold"),
                                likely_threshold=m.calibration[f].get("likely_threshold"))
                  for f in m.findings])
        for m in screening.models]