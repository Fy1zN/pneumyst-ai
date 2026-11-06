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
View = Literal["PA", "AP", "lateral"]
Position = Literal["standing", "sitting", "lying"]
ImageSource = Literal["digital X-ray", "scanned film", "phone photo of film"]
YesNo = Literal["yes", "no"]
YesNoUnknown = Literal["yes", "no", "unknown"]
CoughDuration = Literal["under 2 weeks", "2 weeks or more"]
Smoking = Literal["never", "former", "current"]
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
    """De-identified details entered by the user, all optional; a missing answer means "not asked", not "no".

    Shown next to the results and on the report. Never used by the models or sent to the LLM.
    """

    case_number: str | None = Field(None, pattern=CASE_NUMBER_PATTERN)
    age_band: AgeBand | None = None
    sex: Sex | None = None
    reason: Reason | None = None
    view: View | None = None
    position: Position | None = None
    image_source: ImageSource | None = None
    cough: YesNo | None = None
    cough_duration: CoughDuration | None = None
    fever: YesNo | None = None
    night_sweats: YesNo | None = None
    weight_loss: YesNo | None = None
    breathlessness: YesNo | None = None
    chest_pain: YesNo | None = None
    coughing_blood: YesNo | None = None
    tb_contact: YesNoUnknown | None = None
    previous_tb: YesNoUnknown | None = None
    diabetes: YesNoUnknown | None = None
    smoking: Smoking | None = None


class WhoScreen(BaseModel):
    result: Literal["positive", "negative", "incomplete"]
    positive_symptoms: list[str]
    not_asked: list[str]


class Caution(BaseModel):
    code: str
    text: str


class CaseNotes(BaseModel):
    """Fixed rules applied to the case context. They sit next to the scores and never change them."""

    who_screen: WhoScreen | None
    cautions: list[Caution]


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
    case_notes: CaseNotes | None = None


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