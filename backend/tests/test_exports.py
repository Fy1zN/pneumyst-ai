"""Case context on upload, the PDF report and the audit record."""
from __future__ import annotations

import io
import json

import pytest
from pypdf import PdfReader

from backend.app.agent.facts import build_facts
from backend.app.agent.report_graph import ReportWriter
from backend.app.storage.files import LocalStorage
from backend.tests.conftest import make_client, memory_store
from backend.tests.test_report import ScriptedModel
from backend.tests.test_storage import KEY, Clock, spot_service, spot_xray

CASE = {"case_number": "CASE-0042", "age_band": "40-49", "sex": "male", "reason": "TB screening"}


def upload(client, case: dict | None = None):
    return client.post("/analyze", files={"file": ("x.png", spot_xray(), "image/png")}, data=case or {})


def given(case: dict | None) -> dict | None:
    """The fields that were entered; responses list every field, with null for "not asked"."""
    return None if case is None else {k: v for k, v in case.items() if v is not None}


def pdf_text(content: bytes) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(content)).pages)


@pytest.fixture
def storage(tmp_path):
    return LocalStorage(tmp_path / "storage", KEY, ttl_s=300, clock=Clock())


def test_case_context_is_saved_and_returned():
    with make_client(memory_store(), service=spot_service()) as c:
        body = upload(c, CASE).json()
        saved = c.get(f"/analyses/{body['id']}").json()
        history = c.get("/analyses").json()
        plain = upload(c).json()
    assert given(body["case_context"]) == CASE and given(saved["case_context"]) == CASE
    assert history["items"][0]["case_number"] == "CASE-0042" and plain["case_context"] is None


@pytest.mark.parametrize("field, value", [
    ("case_number", "Ramesh Kumar"), ("case_number", "x" * 33), ("age_band", "42"), ("sex", "unknown"),
    ("reason", "he has a cough since Monday"),
], ids=["name-with-space", "too-long", "exact-age", "sex", "free-text-reason"])
def test_case_context_rejects_values_outside_the_rules(field, value):
    with make_client(memory_store(), service=spot_service()) as c:
        r = upload(c, {field: value})
    assert r.status_code == 422 and r.json()["error_code"] == "INVALID_REQUEST" and field in r.json()["detail"]


def test_case_context_never_reaches_the_llm():
    store = memory_store()
    model = ScriptedModel("too short")
    with make_client(store, service=spot_service(), report_writer=ReportWriter(model, "m", max_attempts=1)) as c:
        analysis_id = upload(c, CASE).json()["id"]
        c.get(f"/analyses/{analysis_id}/report")
    assert len(model.prompts) == 1
    assert not any(value in model.prompts[0] for value in CASE.values())


def test_pdf_report_contents_and_download(storage):
    store = memory_store()
    with make_client(store, service=spot_service(), storage=storage) as c:
        analysis_id = upload(c, CASE).json()["id"]
        r = c.get(f"/analyses/{analysis_id}/report.pdf")
        saved = c.get(f"/analyses/{analysis_id}/report.pdf", params={"download": "true"})
        summary = c.get(f"/analyses/{analysis_id}/report").json()["summary"]
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"] == 'inline; filename="pneumyst-CASE-0042-report.pdf"'
    assert saved.headers["content-disposition"].startswith("attachment;")
    text = pdf_text(r.content)
    for expected in (analysis_id, "NOT A DIAGNOSTIC REPORT", "CASE-0042", "40-49", "Provided by the user",
                     "Template summary", "Tuberculosis", "LIKELY", "Clinician review", "Patient name / ID"):
        assert expected in text, expected
    assert " ".join(summary.split()[:6]) in " ".join(text.split())
    images = PdfReader(io.BytesIO(r.content)).pages[0].images
    assert len(images) == 2  # the analysed X-ray and the one shown heatmap


def test_pdf_without_stored_images_still_works():
    with make_client(memory_store(), service=spot_service()) as c:
        analysis_id = upload(c).json()["id"]
        r = c.get(f"/analyses/{analysis_id}/report.pdf")
    assert r.status_code == 200 and "Images not stored" in pdf_text(r.content)
    assert f'filename="pneumyst-{analysis_id[:8]}-report.pdf"' in r.headers["content-disposition"]


def test_pdf_writes_the_summary_once():
    store = memory_store()
    with make_client(store, service=spot_service()) as c:
        analysis_id = upload(c).json()["id"]
    row = store.get("local-dev", analysis_id)
    model = ScriptedModel(ReportWriter(None).write(build_facts(row.findings, row.errors)).text)
    with make_client(store, service=spot_service(), report_writer=ReportWriter(model, "test-model")) as c:
        first = pdf_text(c.get(f"/analyses/{analysis_id}/report.pdf").content)
        second = pdf_text(c.get(f"/analyses/{analysis_id}/report.pdf").content)
    assert len(model.prompts) == 1 and "Written by test-model" in first and "Written by test-model" in second


def test_audit_record_is_complete_and_hides_storage(storage):
    with make_client(memory_store(), service=spot_service(), storage=storage) as c:
        analysis_id = upload(c, CASE).json()["id"]
        c.get(f"/analyses/{analysis_id}/report")
        r = c.get(f"/analyses/{analysis_id}/audit.json", params={"download": "true"})
    audit = r.json()
    assert r.headers["content-disposition"] == 'attachment; filename="pneumyst-CASE-0042-audit.json"'
    assert audit["audit_version"] == 2 and audit["app_version"] == "0.7.0"
    assert audit["analysis"]["id"] == analysis_id and audit["case_context"] == CASE
    assert len(audit["findings"]) == 15 and audit["models"][0]["checkpoint_sha256"]
    assert audit["stored_images"] == {"image": True, "heatmaps": ["Tuberculosis"]}
    assert audit["report"]["source"] == "template"
    assert {e["step"] for e in audit["events"]} >= {"input", "images", "report"}
    raw = json.dumps(audit)
    assert "analyses/" not in raw and "sig=" not in raw and "/files/" not in raw


def test_other_users_cannot_download():
    store = memory_store()
    with make_client(store, user="alice", service=spot_service()) as alice:
        analysis_id = upload(alice, CASE).json()["id"]
    with make_client(store, user="bob", service=spot_service()) as bob:
        for path in (f"/analyses/{analysis_id}/report.pdf", f"/analyses/{analysis_id}/audit.json",
                     "/analyses/not-a-uuid/report.pdf", "/analyses/not-a-uuid/audit.json"):
            r = bob.get(path)
            assert r.status_code == 404 and r.json()["error_code"] == "ANALYSIS_NOT_FOUND", path