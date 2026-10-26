"""Analysis pipeline: upload checks, image fingerprint, then screening on the shared models, one job at a time."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi.concurrency import run_in_threadpool

from backend.app.core.config import Settings
from backend.app.core.errors import ApiError
from ml.inference.screener import DEFAULT_RUNS, Screener, ScreeningResult

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"


@dataclass(frozen=True)
class ModelInfo:
    name: str
    findings: list[str]
    calibration: dict
    available: bool
    error: str | None
    checkpoint_sha256: str | None


@dataclass
class Analysis:
    id: str
    created_at: datetime
    image_sha256: str
    image_format: str
    result: ScreeningResult


def image_format(data: bytes) -> str | None:
    """Format from the file's leading bytes; the file name and declared content type are not trusted."""
    if data.startswith(PNG_SIGNATURE):
        return "png"
    if data.startswith(JPEG_SIGNATURE):
        return "jpeg"
    return None


def file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


class ScreeningService:
    def __init__(self, screener: Screener, models: list[ModelInfo], settings: Settings):
        self.screener = screener
        self.models = models
        self.settings = settings
        self._slot = threading.Lock()  # one screening at a time: the models share one small GPU

    @classmethod
    def load(cls, settings: Settings) -> "ScreeningService":
        screener = Screener.load(settings.repo, settings.device)
        models = []
        for m, (_, cal_file, _) in zip(screener.models, DEFAULT_RUNS):
            checkpoint = json.loads((settings.repo / "ml" / "configs" / cal_file).read_text())["checkpoint"]
            models.append(ModelInfo(m.name, m.findings, m.calibration, m.model is not None, m.error,
                                    file_sha256(settings.repo / checkpoint) if m.model is not None else None))
        return cls(screener, models, settings)

    @property
    def device(self) -> str:
        return str(self.screener.device)

    def check_upload(self, data: bytes) -> str:
        if not data:
            raise ApiError(422, "INVALID_IMAGE", "empty_file")
        if len(data) > self.settings.max_upload_bytes:
            raise ApiError(413, "FILE_TOO_LARGE", f"limit is {self.settings.max_upload_bytes // (1024 * 1024)} MB")
        fmt = image_format(data)
        if fmt is None:
            raise ApiError(422, "INVALID_IMAGE", "unsupported_format: only PNG and JPEG are accepted")
        return fmt

    async def analyze(self, analysis_id: str, data: bytes, explain: bool) -> Analysis:
        fmt = self.check_upload(data)
        digest = hashlib.sha256(data).hexdigest()
        created = datetime.now(timezone.utc)
        result = await run_in_threadpool(self._screen, data, explain)
        if result.error is not None:
            raise ApiError(422, "INVALID_IMAGE", result.error, analysis_id)
        if result.status == "failed":
            unavailable = all(m.model is None for m in self.screener.models)
            raise ApiError(503, "MODEL_UNAVAILABLE" if unavailable else "INFERENCE_FAILED",
                           "no model produced a result", analysis_id)
        return Analysis(analysis_id, created, digest, fmt, result)

    def _screen(self, data: bytes, explain: bool) -> ScreeningResult:
        if not self._slot.acquire(timeout=self.settings.queue_timeout_s):
            raise ApiError(503, "SERVER_BUSY", "the screening queue is full, try again shortly")
        try:
            return self.screener.screen(data, explain=explain)
        finally:
            self._slot.release()