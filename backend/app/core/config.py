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
    database_url: str | None = None           # None: results are returned but not saved
    auth_mode: str = "clerk"                  # "clerk" verifies session tokens; "dev" trusts everyone (local only)
    dev_user_id: str = "local-dev"            # owner of all analyses in dev mode
    clerk_issuer: str | None = None
    clerk_jwks_url: str | None = None
    clerk_authorized_parties: tuple[str, ...] = ("http://localhost:3000",)
    storage_dir: Path | None = None           # None: images are not stored
    file_signing_key: str | None = None
    file_url_ttl_s: int = 300                 # lifetime of image links


def load_settings() -> Settings:
    load_dotenv(REPO / ".env")
    device = os.environ.get("MODEL_DEVICE", "auto").strip().lower()
    origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
    data_root = os.environ.get("DATA_ROOT", "").strip()
    storage_dir = os.environ.get("STORAGE_DIR", "").strip() or (str(Path(data_root) / "storage") if data_root else "")
    return Settings(
        device=None if device in ("", "auto") else device,
        max_upload_bytes=int(os.environ.get("MAX_UPLOAD_MB", "20")) * 1024 * 1024,
        queue_timeout_s=float(os.environ.get("QUEUE_TIMEOUT_S", "60")),
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        database_url=os.environ.get("DATABASE_URL", "").strip() or None,
        auth_mode=os.environ.get("AUTH_MODE", "clerk").strip().lower() or "clerk",
        dev_user_id=os.environ.get("DEV_USER_ID", "local-dev").strip() or "local-dev",
        clerk_issuer=os.environ.get("CLERK_ISSUER", "").strip() or None,
        clerk_jwks_url=os.environ.get("CLERK_JWKS_URL", "").strip() or None,
        clerk_authorized_parties=tuple(p.strip() for p in os.environ.get(
            "CLERK_AUTHORIZED_PARTIES", "http://localhost:3000").split(",") if p.strip()),
        storage_dir=Path(storage_dir) if storage_dir else None,
        file_signing_key=os.environ.get("FILE_SIGNING_KEY", "").strip() or None,
        file_url_ttl_s=int(os.environ.get("FILE_URL_TTL_S", "300")),
    )