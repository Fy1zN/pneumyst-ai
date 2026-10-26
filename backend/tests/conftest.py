"""Shared fixtures: the real Screener and pipeline around tiny toy models, so tests need no checkpoints or GPU."""
from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.services.screening import ModelInfo, ScreeningService
from ml.datasets.targets import NIH_FINDINGS, TB
from ml.inference.screener import LoadedModel, Screener

BANDED = {"platt_a": 1.0, "platt_b": 0.0, "possible_threshold": 0.3, "likely_threshold": 0.7, "status": "banded"}


class ConstModel(nn.Module):
    def __init__(self, probs: list[float]):
        super().__init__()
        p = torch.tensor(probs)
        self.register_buffer("logits", torch.log(p / (1 - p)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(len(x), -1)


def calibration(findings: list[str]) -> dict:
    return {f: {**BANDED, "status": "insufficient_data"} if f == "Hernia" else BANDED for f in findings}


def make_service(nih: nn.Module | None = None, tb: nn.Module | None = None, **settings) -> ScreeningService:
    loaded = [LoadedModel("nih_test", NIH_FINDINGS, calibration(NIH_FINDINGS), nih,
                          None if nih is not None else "MODEL_UNAVAILABLE: FileNotFoundError"),
              LoadedModel("tb_test", [TB], calibration([TB]), tb,
                          None if tb is not None else "MODEL_UNAVAILABLE: FileNotFoundError")]
    models = [ModelInfo(m.name, m.findings, m.calibration, m.model is not None, m.error,
                        "0" * 64 if m.model is not None else None) for m in loaded]
    return ScreeningService(Screener(loaded, device="cpu"), models, Settings(**settings))


def default_models() -> tuple[nn.Module, nn.Module]:
    return ConstModel([0.1] * len(NIH_FINDINGS)), ConstModel([0.05])


@pytest.fixture
def client():
    with TestClient(create_app(Settings(), make_service(*default_models())), raise_server_exceptions=False) as c:
        yield c


def xray(size: tuple[int, int] = (400, 360), fmt: str = "PNG", blank: bool = False) -> bytes:
    rng = np.random.default_rng(0)
    arr = np.full(size, 90, np.uint8) if blank else rng.normal(120, 30, size).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()