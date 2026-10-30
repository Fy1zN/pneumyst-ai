"""Response shapes of the public API."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

Band = Literal["unlikely", "possible", "likely"]


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