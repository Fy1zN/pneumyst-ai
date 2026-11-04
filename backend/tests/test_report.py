"""Report facts, guardrail checks, the LangGraph workflow with a scripted model, and the report endpoint."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from sqlalchemy import select

from backend.app.agent.checks import check_summary
from backend.app.agent.facts import build_facts, pct
from backend.app.agent.llm import make_chat_model
from backend.app.agent.report_graph import ReportWriter, template_summary
from backend.app.core.config import Settings
from backend.app.db.models import AuditEventRecord
from backend.tests.conftest import ConstModel, make_client, make_service, memory_store, xray
from ml.datasets.targets import NIH_FINDINGS


class ScriptedModel:
    """Stands in for a chat model: returns the scripted replies in order, raising any that are exceptions."""

    def __init__(self, *replies, model_name=None):
        self.replies = list(replies)
        self.prompts = []
        self.metadata = {"model_name": model_name} if model_name else {}

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=reply, response_metadata=self.metadata)


def finding(name, band, score, status="success", band_status="banded"):
    return {"finding": name, "display_name": name.replace("_", " ").capitalize(), "status": status,
            "score": score, "band": band, "band_status": band_status,
            "possible_threshold": 0.01, "likely_threshold": 0.05}


FINDINGS = [finding("Effusion", "likely", 0.33644), finding("Nodule", "possible", 0.037522),
            finding("Edema", "unlikely", 0.0078), finding("Hernia", None, 0.006238, band_status="insufficient_data")]
FACTS = build_facts(FINDINGS, {})
GOOD = ("The screening models scored Effusion as likely at 34%, above its likely threshold of 5.0%. "
        "Nodule was scored as possible at 3.8%, between its thresholds of 1.0% and 5.0%. "
        "Edema was scored unlikely. The heatmaps show the image regions that changed each score most "
        "and should be reviewed alongside the image.")


def test_facts_group_findings_and_format_scores():
    assert [f["finding"] for f in FACTS["likely"]] == ["Effusion"] and FACTS["likely"][0]["score"] == "34%"
    assert FACTS["possible"][0] == {"finding": "Nodule", "score": "3.8%", "possible_threshold": "1.0%",
                                    "likely_threshold": "5.0%"}
    assert FACTS["unlikely"] == ["Edema"] and FACTS["not_banded"] == [{"finding": "Hernia", "score": "0.62%"}]
    assert pct(0.00036) == "0.04%" and pct(0.5) == "50%"


def test_facts_report_unavailable_models_and_failed_heatmaps():
    findings = [finding("Effusion", None, None, status="unavailable")]
    facts = build_facts(findings, {"tb_v2": "MODEL_UNAVAILABLE: FileNotFoundError",
                                   "explain:nih_v1": "EXPLANATION_OOM", "images": "STORAGE_FAILED"})
    assert facts["unavailable"] == ["Effusion"]
    assert facts["problems"] == ["the tb_v2 model was unavailable", "heatmaps from the nih_v1 model could not be made"]


def test_good_summary_passes_checks():
    assert check_summary(GOOD, FACTS) == []


@pytest.mark.parametrize("text, expected", [
    (GOOD.replace("Nodule", "A small spot"), "Nodule (possible) is not mentioned"),
    (GOOD.replace("34%", "35%"), "numbers not in the facts: 35"),
    (GOOD + " You have pleural effusion.", "'you have'"),
    (GOOD + " This confirms the finding.", "'confirms'"),
    (GOOD + " Antibiotics are advised.", "treatment advice"),
    (GOOD.replace("Edema was scored unlikely.", "Edema is present."), "Edema was scored unlikely"),
    ("Effusion 34%, Nodule 3.8%.", "length is"),
], ids=["missing", "number", "you-have", "confirm", "treatment", "unlikely", "short"])
def test_checks_catch_bad_summaries(text, expected):
    assert any(expected in issue for issue in check_summary(text, FACTS))


def test_model_names_and_flagged_count_are_not_invented_numbers():
    text = GOOD + " In total 2 findings were flagged by the densenet121 v2 models."
    assert check_summary(text, FACTS) == []


def test_checks_require_failures_to_be_mentioned():
    facts = build_facts(FINDINGS + [finding("Tuberculosis", None, None, status="unavailable")], {})
    assert any("unavailable" in i for i in check_summary(GOOD, facts))
    assert check_summary(GOOD + " Tuberculosis results were unavailable.", facts) == []


@pytest.mark.parametrize("findings, errors", [
    (FINDINGS, {}),
    ([finding("Edema", "unlikely", 0.0078)], {}),
    (FINDINGS + [finding("Tuberculosis", None, None, status="unavailable")], {"explain:nih": "EXPLANATION_OOM"}),
], ids=["flagged", "nothing-flagged", "failures"])
def test_template_passes_its_own_checks(findings, errors):
    facts = build_facts(findings, errors)
    assert check_summary(template_summary(facts), facts) == []


def test_graph_accepts_a_good_first_draft():
    model = ScriptedModel(GOOD)
    result = ReportWriter(model, "test-model").write(FACTS)
    assert (result.source, result.text, result.model, result.attempts) == ("llm", GOOD, "test-model", 1)
    assert '"Effusion"' in model.prompts[0] and "rejected" not in model.prompts[0]


def test_graph_retries_with_the_issues_then_accepts():
    model = ScriptedModel(GOOD + " You have effusion.", GOOD)
    result = ReportWriter(model, "test-model").write(FACTS)
    assert result.source == "llm" and result.attempts == 2 and result.text == GOOD
    assert "'you have'" in model.prompts[1] and "You have effusion." in model.prompts[1]


def test_graph_falls_back_to_template_after_two_bad_drafts():
    model = ScriptedModel("Effusion is there.", "Still too short.")
    result = ReportWriter(model, "test-model").write(FACTS)
    assert result.source == "template" and result.reason == "CHECKS_FAILED" and result.model is None
    assert result.attempts == 2 and result.issues and result.text == template_summary(FACTS)


def test_graph_records_the_model_that_answered():
    model = ScriptedModel(GOOD, model_name="models/backup-model")
    assert ReportWriter(model, "main-model").write(FACTS).model == "backup-model"


def test_busy_main_model_is_answered_by_the_fallback_model():
    def busy(messages):
        raise RuntimeError("503 UNAVAILABLE")

    backup = RunnableLambda(lambda messages: AIMessage(content=GOOD, response_metadata={"model_name": "backup"}))
    result = ReportWriter(RunnableLambda(busy).with_fallbacks([backup]), "main").write(FACTS)
    assert (result.source, result.model, result.attempts) == ("llm", "backup", 1)


def test_graph_falls_back_when_the_llm_fails():
    result = ReportWriter(ScriptedModel(TimeoutError("slow")), "test-model").write(FACTS)
    assert (result.source, result.reason, result.attempts) == ("template", "LLM_FAILED", 1)


def test_graph_without_model_uses_template():
    result = ReportWriter(None, "unused").write(FACTS)
    assert (result.source, result.reason, result.attempts, result.model) == ("template", "LLM_NOT_CONFIGURED", 0, None)


def flagged_service():
    probs = [0.8, 0.02] + [0.1] * (len(NIH_FINDINGS) - 2)
    return make_service(ConstModel(probs), ConstModel([0.05]))


def test_report_endpoint_writes_once_then_serves_saved_report():
    store = memory_store()
    probe = ReportWriter(None).write  # the template text for this analysis, to script a passing LLM reply
    with make_client(store, service=flagged_service()) as c:
        analysis_id = c.post("/analyze", files={"file": ("x.png", xray(), "image/png")}).json()["id"]
        row = store.get("local-dev", analysis_id)
        reply = probe(build_facts(row.findings, row.errors)).text
    model = ScriptedModel(reply)
    with make_client(store, service=flagged_service(), report_writer=ReportWriter(model, "test-model")) as c:
        first = c.get(f"/analyses/{analysis_id}/report").json()
        second = c.get(f"/analyses/{analysis_id}/report").json()
    assert first == second and len(model.prompts) == 1
    assert first["source"] == "llm" and first["model"] == "test-model" and first["summary"] == reply
    assert "not a diagnosis" in first["disclaimer"]
    with store.db.session() as s:
        events = s.scalars(select(AuditEventRecord).where(AuditEventRecord.step == "report")).all()
    assert [(e.status, e.error_code) for e in events] == [("llm", None)]


def test_temporary_llm_failure_is_not_saved():
    store = memory_store()
    with make_client(store, service=flagged_service()) as c:
        analysis_id = c.post("/analyze", files={"file": ("x.png", xray(), "image/png")}).json()["id"]
        row = store.get("local-dev", analysis_id)
        reply = ReportWriter(None).write(build_facts(row.findings, row.errors)).text
    model = ScriptedModel(TimeoutError("busy"), reply)
    with make_client(store, service=flagged_service(), report_writer=ReportWriter(model, "test-model")) as c:
        first = c.get(f"/analyses/{analysis_id}/report").json()
        assert first["source"] == "template" and store.get("local-dev", analysis_id).report is None
        second = c.get(f"/analyses/{analysis_id}/report").json()
    assert second["source"] == "llm" and len(model.prompts) == 2
    with store.db.session() as s:
        events = s.scalars(select(AuditEventRecord).where(AuditEventRecord.step == "report")).all()
    assert [(e.status, e.error_code) for e in events] == [("template", "LLM_FAILED"), ("llm", None)]


def test_report_endpoint_falls_back_and_hides_other_users():
    store = memory_store()
    with make_client(store, user="alice", service=flagged_service()) as alice:
        analysis_id = alice.post("/analyze", files={"file": ("x.png", xray(), "image/png")}).json()["id"]
        report = alice.get(f"/analyses/{analysis_id}/report").json()
        assert report["source"] == "template" and report["model"] is None and "Atelectasis" in report["summary"]
    with make_client(store, user="bob") as bob:
        assert bob.get(f"/analyses/{analysis_id}/report").status_code == 404
        assert bob.get("/analyses/not-a-uuid/report").status_code == 404
    with make_client(None) as c:
        assert c.get(f"/analyses/{analysis_id}/report").json()["error_code"] == "DATABASE_UNAVAILABLE"


def test_chat_model_settings():
    assert make_chat_model(Settings()) is None
    with pytest.raises(RuntimeError, match="LLM_MODEL"):
        make_chat_model(Settings(llm_api_key="key"))
    with pytest.raises(RuntimeError, match="LLM_PROVIDER"):
        make_chat_model(Settings(llm_provider="claude", llm_api_key="key", llm_model="m"))
    gemini = make_chat_model(Settings(llm_api_key="key", llm_model="gemini-test"))
    openai = make_chat_model(Settings(llm_provider="openai", llm_api_key="key", llm_model="gpt-test"))
    assert type(gemini).__name__ == "ChatGoogleGenerativeAI" and type(openai).__name__ == "ChatOpenAI"
    both = make_chat_model(Settings(llm_api_key="key", llm_model="main", llm_fallback_model="backup",
                                    llm_thinking_level="low"))
    assert [m.model.removeprefix("models/") for m in (both.runnable, *both.fallbacks)] == ["main", "backup"]
    assert both.runnable.thinking_level == "low" and both.fallbacks[0].thinking_level == "low"