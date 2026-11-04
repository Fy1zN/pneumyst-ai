"""Response shapes of the public API."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Band = Literal["unlikely", "possible", "likely"]
AgeBand = Literal["0-17", "18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
Sex = Literal["female", "male", "other"]
Reason = Literal["cough or fever", "TB screening", "breathlessness", "chest pain", "follow-up", "pre-operative",
                 "routine check", "other"]
# Letters, digits, '-' and '_' only, so a full name with spaces cannot be typed in by accident.
CASE_NUMBER_PATTERN = r"^[A-Za-z0-9_-]{1,32}$"


class ErrorResponse(BaseModel):
    error_code: str
    detail: str
    analysis_id: str | None = None


class Finding(BaseModel):
    finding: str
    display_name: str
    group: str
    model: str
    status: Literal["success", "unavailable", "failed"]
    score: float | None
    band: Band | None
    band_status: Literal["banded", "insufficient_data"]
    possible_threshold: float | None
    likely_threshold: float | None
    heatmap: str
    heatmap_peak: float | None


class CaseContext(BaseModel):
    """De-identified details entered by the user. Shown on the report, never used by the models or the LLM."""

    case_number: str | None = Field(None, pattern=CASE_NUMBER_PATTERN)
    age_band: AgeBand | None = None
    sex: Sex | None = None
    reason: Reason | None = None


class ModelRef(BaseModel):
    name: str
    checkpoint_sha256: str | None


class ImageLinks(BaseModel):
    image: str
    heatmaps: dict[str, str]


class AnalysisResponse(BaseModel):
    id: str
    created_at: datetime
    image_sha256: str
    image_format: Literal["png", "jpeg"]
    status: Literal["success", "partial"]
    stages: dict[str, str]
    errors: dict[str, str]
    findings: list[Finding]
    models: list[ModelRef]
    preprocess: dict
    timings_ms: dict[str, float]
    persistence: Literal["saved", "not_saved"]
    images: ImageLinks | None = None
    case_context: CaseContext | None = None


class ReportResponse(BaseModel):
    analysis_id: str
    summary: str
    source: Literal["llm", "template"]
    model: str | None
    created_at: datetime
    disclaimer: str


class FlaggedFinding(BaseModel):
    finding: str
    display_name: str
    band: Literal["possible", "likely"]


class AnalysisSummary(BaseModel):
    id: str
    created_at: datetime
    status: Literal["success", "partial"]
    image_format: Literal["png", "jpeg"]
    flagged: list[FlaggedFinding]
    case_number: str | None = None


class AnalysisPage(BaseModel):
    items: list[AnalysisSummary]
    total: int
    limit: int
    offset: int


class FindingConfig(BaseModel):
    finding: str
    display_name: str
    band_status: Literal["banded", "insufficient_data"]
    possible_threshold: float | None
    likely_threshold: float | None


class ModelStatus(BaseModel):
    name: str
    available: bool
    error: str | None
    checkpoint_sha256: str | None
    findings: list[FindingConfig]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    device: str
    models: dict[str, bool]
    database: Literal["ok", "unavailable", "not_configured"]