"""Backend settings, read once from environment variables (.env during development)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    repo: Path = REPO
    device: str | None = None                 # None: CUDA when available, otherwise CPU
    max_upload_bytes: int = 20 * 1024 * 1024
    queue_timeout_s: float = 60.0             # longest wait for the single screening slot
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)


def load_settings() -> Settings:
    load_dotenv(REPO / ".env")
    device = os.environ.get("MODEL_DEVICE", "auto").strip().lower()
    origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
    return Settings(
        device=None if device in ("", "auto") else device,
        max_upload_bytes=int(os.environ.get("MAX_UPLOAD_MB", "20")) * 1024 * 1024,
        queue_timeout_s=float(os.environ.get("QUEUE_TIMEOUT_S", "60")),
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
    )