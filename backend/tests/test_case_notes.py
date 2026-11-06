"""Clinical context: the WHO symptom screen, cautions, side-by-side display, and proof that scores never change."""
from __future__ import annotations

import pytest

from backend.app.agent.report_graph import ReportWriter
from backend.app.reporting.audit import ordered_stages
from backend.app.services.case_notes import case_notes, cautions, who_screen
from backend.tests.conftest import make_client, memory_store
from backend.tests.test_exports import pdf_text, upload
from backend.tests.test_report import ScriptedModel
from backend.tests.test_storage import spot_service

FULL = {"case_number": "CASE-0042", "age_band": "0-17", "sex": "female", "reason": "TB screening",
        "view": "AP", "position": "lying", "image_source": "phone photo of film", "cough": "yes",
        "cough_duration": "2 weeks or more", "fever": "no", "night_sweats": "yes", "weight_loss": "no",
        "breathlessness": "yes", "chest_pain": "no", "tb_contact": "yes", "previous_tb": "yes", "diabetes": "unknown",
        "smoking": "former"}


@pytest.mark.parametrize("answers, result, positive, not_asked", [
    ({"cough": "yes", "fever": "no"}, "positive", ["current cough"], ["night sweats", "weight loss"]),
    ({"cough": "no", "fever": "no", "night_sweats": "no", "weight_loss": "no"}, "negative", [], []),
    ({"cough": "no", "fever": "no"}, "incomplete", [], ["night sweats", "weight loss"]),
], ids=["positive", "negative", "incomplete"])
def test_who_screen(answers, result, positive, not_asked):
    assert who_screen(answers) == {"result": result, "positive_symptoms": positive, "not_asked": not_asked}


def test_who_screen_needs_at_least_one_answer():
    assert who_screen({"breathlessness": "yes", "age_band": "40-49"}) is None
    assert case_notes(None) is None and case_notes({}) is None


def test_each_caution_has_its_trigger():
    assert [c["code"] for c in cautions(FULL)] == ["CHILD", "AP_VIEW", "LYING_POSITION", "PHONE_PHOTO",
                                                   "PREVIOUS_TB"]
    assert [c["code"] for c in cautions({"view": "lateral"})] == ["LATERAL_VIEW"]
    assert cautions({"age_band": "18-29", "view": "PA", "position": "standing", "image_source": "digital X-ray",
                     "previous_tb": "no"}) == []


def test_case_context_never_changes_the_scores():
    with make_client(memory_store(), service=spot_service()) as c:
        plain = upload(c).json()
        with_context = upload(c, FULL).json()
    assert with_context["findings"] == plain["findings"]
    assert with_context["status"] == plain["status"] and with_context["stages"] == plain["stages"]


def test_notes_are_returned_with_the_analysis():
    with make_client(memory_store(), service=spot_service()) as c:
        body = upload(c, FULL).json()
        saved = c.get(f"/analyses/{body['id']}").json()
        plain = upload(c).json()
    assert body["case_notes"] == saved["case_notes"]
    assert body["case_notes"]["who_screen"]["result"] == "positive"
    assert "CHILD" in [c["code"] for c in body["case_notes"]["cautions"]]
    assert plain["case_notes"] is None and plain["case_context"] is None


@pytest.mark.parametrize("field, value", [
    ("view", "side"), ("position", "upright"), ("image_source", "screenshot"), ("cough", "maybe"),
    ("cough_duration", "3 days"), ("tb_contact", "perhaps"), ("smoking", "sometimes"),
], ids=["view", "position", "source", "cough", "duration", "contact", "smoking"])
def test_clinical_fields_only_accept_fixed_choices(field, value):
    with make_client(memory_store(), service=spot_service()) as c:
        r = upload(c, {field: value})
    assert r.status_code == 422 and r.json()["error_code"] == "INVALID_REQUEST" and field in r.json()["detail"]


def test_clinical_context_never_reaches_the_llm():
    model = ScriptedModel("too short")
    writer = ReportWriter(model, "m", max_attempts=1)
    with make_client(memory_store(), service=spot_service(), report_writer=writer) as c:
        analysis_id = upload(c, FULL).json()["id"]
        c.get(f"/analyses/{analysis_id}/report")
    prompt = model.prompts[0].lower()
    for word in ("case-0042", "phone photo", "night sweats", "2 weeks or more", "smoking", "lying", "0-17"):
        assert word not in prompt, word


def test_pdf_shows_screen_and_cautions_side_by_side():
    with make_client(memory_store(), service=spot_service()) as c:
        analysis_id = upload(c, FULL).json()["id"]
        text = " ".join(pdf_text(c.get(f"/analyses/{analysis_id}/report.pdf").content).split())
        plain_id = upload(c).json()["id"]
        plain = " ".join(pdf_text(c.get(f"/analyses/{plain_id}/report.pdf").content).split())
    for expected in ("WHO four-symptom screen POSITIVE: current cough, night sweats", "TB model (X-ray only)",
                     "2 weeks or more", "present: breathlessness", "not asked: coughing up blood",
                     "results for children are less reliable", "Photos of film", "Healed scars",
                     "The AI does not combine symptoms with the X-ray scores"):
        assert expected in text, expected
    assert "Caution:" not in plain and "WHO four-symptom screen Not recorded" in plain


def test_audit_has_notes_and_stages_in_pipeline_order():
    with make_client(memory_store(), service=spot_service()) as c:
        analysis_id = upload(c, FULL).json()["id"]
        audit = c.get(f"/analyses/{analysis_id}/audit.json").json()
    assert audit["case_notes"]["rules_version"] == 1 and audit["case_notes"]["who_screen"]["result"] == "positive"
    assert list(audit["stages"])[0] == "input" and list(audit["stages"])[-1] == "images"
    shuffled = {"images": "success", "explain:a": "success", "b": "success", "input": "success"}
    assert list(ordered_stages(shuffled)) == ["input", "b", "explain:a", "images"]