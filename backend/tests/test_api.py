"""API behaviour for /health, /models and /analyze."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.tests.conftest import ConstModel, default_models, make_service, xray
from ml.datasets.targets import NIH_FINDINGS


def upload(client: TestClient, data: bytes, name: str = "cxr.png", **params):
    return client.post("/analyze", files={"file": (name, data, "image/png")}, params=params)


def run(service, data: bytes, **params):
    with TestClient(create_app(Settings(), service), raise_server_exceptions=False) as c:
        return upload(c, data, **params)


def test_health_and_models(client):
    body = client.get("/health").json()
    assert body == {"status": "degraded", "device": "cpu", "models": {"nih_test": True, "tb_test": True},
                    "database": "not_configured"}  
    models = client.get("/models").json()
    assert [m["name"] for m in models] == ["nih_test", "tb_test"] and len(models[0]["findings"]) == 14
    hernia = next(f for f in models[0]["findings"] if f["finding"] == "Hernia")
    assert hernia["band_status"] == "insufficient_data"


def test_health_reports_missing_models():
    nih, _ = default_models()
    with TestClient(create_app(Settings(), make_service(nih, None))) as c:
        assert c.get("/health").json()["status"] == "degraded"
    with TestClient(create_app(Settings(), make_service(None, None))) as c:
        assert c.get("/health").json()["status"] == "down"


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_analyze_returns_all_findings(client, fmt):
    r = upload(client, xray(fmt=fmt), name=f"cxr.{fmt.lower()}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success" and body["image_format"] == fmt.lower() and len(body["findings"]) == 15
    assert len(body["image_sha256"]) == 64 and body["preprocess"]["size"] == 320
    assert {f["band"] for f in body["findings"] if f["finding"] != "Hernia"} == {"unlikely"}
    assert body["stages"]["explain:nih_test"] == "skipped"


@pytest.mark.parametrize("data, detail", [
    (b"", "empty_file"),
    (b"just some text pretending to be a png", "unsupported_format"),
    (b"\x00" * 128 + b"DICM" + b"\x00" * 64, "unsupported_format"),
], ids=["empty", "text", "dicom"])
def test_rejects_non_images(client, data, detail):
    r = upload(client, data)
    assert r.status_code == 422 and r.json()["error_code"] == "INVALID_IMAGE"
    assert r.json()["detail"].startswith(detail) and r.json()["analysis_id"]


def test_rejects_oversized_upload():
    r = run(make_service(*default_models(), max_upload_bytes=1000), xray())
    assert r.status_code == 413 and r.json()["error_code"] == "FILE_TOO_LARGE"


def test_rejects_blank_image_with_screener_reason(client):
    r = upload(client, xray(blank=True))
    assert r.status_code == 422 and r.json()["detail"] == "blank_image"


def test_partial_result_is_still_returned():
    nih, _ = default_models()
    r = run(make_service(nih, None), xray())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "partial" and body["stages"]["tb_test"] == "unavailable"
    tb = next(f for f in body["findings"] if f["finding"] == "Tuberculosis")
    assert tb["score"] is None and tb["status"] == "unavailable"


def test_no_models_gives_503():
    r = run(make_service(None, None), xray())
    assert r.status_code == 503 and r.json()["error_code"] == "MODEL_UNAVAILABLE"


def test_heatmap_failure_keeps_scores():
    # Constant toy models give constant heatmaps, which the screener rejects as degenerate.
    service = make_service(ConstModel([0.5] * len(NIH_FINDINGS)), ConstModel([0.05]))
    body = run(service, xray()).json()
    assert body["status"] == "partial" and body["errors"]["explain:nih_test"] == "EXPLANATION_DEGENERATE"
    assert all(f["score"] is not None for f in body["findings"])
    body = run(service, xray(), explain=False).json()
    assert body["status"] == "success" and {f["heatmap"] for f in body["findings"]} == {"not_requested"}


def test_busy_queue_gives_503():
    service = make_service(*default_models(), queue_timeout_s=0.1)
    service._slot.acquire()
    try:
        r = run(service, xray())
    finally:
        service._slot.release()
    assert r.status_code == 503 and r.json()["error_code"] == "SERVER_BUSY"


def test_unexpected_crash_is_hidden(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(client.app.state.screening.screener, "screen", boom)
    r = upload(client, xray())
    assert r.status_code == 500 and r.json()["error_code"] == "INTERNAL_ERROR"
    assert "secret" not in r.text and r.json()["analysis_id"]