"""Shared configuration for ML scripts."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional; plain environment variables also work
    pass


def data_root() -> Path:
    """Return the dataset root folder from the DATA_ROOT setting."""
    value = os.environ.get("DATA_ROOT")
    if not value:
        raise RuntimeError("DATA_ROOT is not set. Copy .env.example to .env and set DATA_ROOT.")
    root = Path(os.path.expandvars(value)).expanduser()
    if not root.exists():
        raise RuntimeError(f"DATA_ROOT does not exist: {root}")
    return root
