"""Warm-up, background reports, deadlines, rate limits, health details and request ids."""
from __future__ import annotations

import time

import pytest
from sqlalchemy import select

from backend.app.agent.assistant import Assistant
from backend.app.agent.facts import build_facts
from backend.app.agent.report_graph import ReportWriter
from backend.app.core.errors import ApiError
from backend.app.core.limits import RateLimiter
from backend.app.db.models import AuditEventRecord
from backend.app.storage.files import LocalStorage
from backend.tests.conftest import make_client, memory_store
from backend.tests.test_assistant import ToolScript
from backend.tests.test_exports import upload
from backend.tests.test_report import ScriptedModel
from backend.tests.test_storage import KEY, Clock, spot_service


class SlowModel(ScriptedModel):
    """A scripted chat model that takes `delay` seconds to answer, like a busy provider."""

    def __init__(self, *replies, delay: float):
        super().__init__(*replies)
        self.delay = delay

    def invoke(self, messages):
        time.sleep(self.delay)
        return super().invoke(messages)


def template_reply(store, analysis_id: str) -> str:
    """A reply that passes the checks for this analysis: the template text written for its own facts."""
    row = store.get("local-dev", analysis_id)
    return ReportWriter(None).write(build_facts(row.findings, row.errors)).text


def wait_for_report(store, analysis_id: str, timeout: float = 3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        row = store.get("local-dev", analysis_id)
        if row.report is not None:
            return row
        time.sleep(0.02)
    raise AssertionError("report was not saved in time")


def report_events(store, analysis_id: str) -> list[tuple]:
    with store.db.session() as s:
        rows = s.scalars(select(AuditEventRecord).where(AuditEventRecord.analysis_id == analysis_id,
                                                        AuditEventRecord.step.in_(["report", "ask"]))
                         .order_by(AuditEventRecord.id)).all()
    return [(e.step, e.status, e.error_code) for e in rows]


def test_warm_up_runs_once_at_startup_when_enabled(monkeypatch):
    calls = []
    for warmup in (True, False):
        service = spot_service()
        monkeypatch.setattr(service, "warm_up", lambda: calls.append(1) or 5.0)
        with make_client(service=service, warmup=warmup):
            pass
    assert calls == [1]
    assert spot_service().warm_up() > 0


def test_report_is_written_in_the_background_after_analyze():
    store = memory_store()
    with make_client(store, service=spot_service()) as c:
        analysis_id = upload(c).json()["id"]
    # The same image gives the same facts, so the first analysis's template passes the checks for the second.
    model = ScriptedModel(template_reply(store, analysis_id))
    with make_client(store, service=spot_service(), report_writer=ReportWriter(model, "test-model"),
                     background_reports=True) as c:
        analysis_id = upload(c).json()["id"]
        row = wait_for_report(store, analysis_id)
        report = c.get(f"/analyses/{analysis_id}/report").json()
    assert row.report_source == "llm" and report["source"] == "llm" and len(model.prompts) == 1


def test_report_request_waits_for_the_running_job_instead_of_calling_twice():
    store = memory_store()
    with make_client(store, service=spot_service()) as c:
        reply = template_reply(store, upload(c).json()["id"])
    model = SlowModel(reply, delay=0.3)
    with make_client(store, service=spot_service(), report_writer=ReportWriter(model, "test-model"),
                     background_reports=True) as c:
        analysis_id = upload(c).json()["id"]
        report = c.get(f"/analyses/{analysis_id}/report").json()
    assert report["source"] == "llm" and len(model.prompts) == 1


def test_deadline_returns_template_now_and_saves_the_late_summary():
    store = memory_store()
    with make_client(store, service=spot_service()) as c:
        analysis_id = upload(c).json()["id"]
    model = SlowModel(template_reply(store, analysis_id), delay=1.0)
    with make_client(store, service=spot_service(), report_writer=ReportWriter(model, "test-model"),
                     report_deadline_s=0.1) as c:
        t0 = time.monotonic()
        first = c.get(f"/analyses/{analysis_id}/report").json()
        waited = time.monotonic() - t0
        assert first["source"] == "template" and waited < 0.8
        wait_for_report(store, analysis_id)
        later = c.get(f"/analyses/{analysis_id}/report").json()
    assert later["source"] == "llm" and later["model"] == "test-model" and len(model.prompts) == 1
    assert report_events(store, analysis_id) == [("report", "template", "LLM_TIMEOUT"), ("report", "llm", None)]


def test_slow_assistant_gives_503_at_the_deadline():
    store = memory_store()
    with make_client(store, service=spot_service()) as c:
        analysis_id = upload(c).json()["id"]

    class SlowScript(ToolScript):
        def invoke(self, messages):
            time.sleep(0.5)
            return super().invoke(messages)

    with make_client(store, service=spot_service(), assistant=Assistant(SlowScript("late answer")),
                     ask_deadline_s=0.1) as c:
        r = c.post(f"/analyses/{analysis_id}/ask", json={"question": "Which findings are flagged?"})
    assert r.status_code == 503 and r.json()["error_code"] == "ASSISTANT_UNAVAILABLE"
    assert report_events(store, analysis_id) == [("ask", "failed", "LLM_TIMEOUT")]


def test_rate_limits_return_429_with_retry_after():
    with make_client(memory_store(), service=spot_service(), rate_analyze_per_hour=2) as c:
        assert [upload(c).status_code for _ in range(2)] == [200, 200]
        r = upload(c)
    assert r.status_code == 429 and r.json()["error_code"] == "RATE_LIMITED"
    assert 0 < int(r.headers["retry-after"]) <= 3600
    store = memory_store()
    with make_client(store, service=spot_service(), assistant=Assistant(ToolScript("The model scored it.")),
                     rate_ask_per_hour=1) as c:
        analysis_id = upload(c).json()["id"]
        codes = [c.post(f"/analyses/{analysis_id}/ask", json={"question": "Which findings are flagged?"}).status_code
                 for _ in range(2)]
    assert codes == [200, 429]


def test_limiter_counts_each_user_separately_and_forgets_after_an_hour():
    clock = Clock(0.0)
    limiter = RateLimiter({"analyze": 2, "ask": 0}, clock=clock)
    limiter.check("analyze", "alice")
    limiter.check("analyze", "alice")
    limiter.check("analyze", "bob")
    with pytest.raises(ApiError) as exc:
        limiter.check("analyze", "alice")
    assert exc.value.status_code == 429 and exc.value.headers == {"Retry-After": "3600"}
    clock.now = 3600.0
    limiter.check("analyze", "alice")
    for _ in range(50):
        limiter.check("ask", "alice")


def test_health_reports_configuration_without_secrets(tmp_path):
    storage = LocalStorage(tmp_path, KEY)
    with make_client(storage=storage, assistant=Assistant(ToolScript()), llm_api_key="secret-key",
                     llm_model="gemini-test") as c:
        body = c.get("/health").json()
    assert body["storage"] == "configured" and body["llm"] == "configured" and body["llm_provider"] == "gemini"
    assert body["auth_mode"] == "dev" and body["version"] == "0.9.0" and "secret-key" not in str(body)


def test_every_response_carries_a_request_id():
    with make_client() as c:
        made = c.get("/health").headers["x-request-id"]
        kept = c.get("/health", headers={"X-Request-ID": "frontend-123"}).headers["x-request-id"]
        replaced = c.get("/health", headers={"X-Request-ID": "bad id\nwith newline"}).headers["x-request-id"]
        failed = c.get("/analyses/not-a-uuid").headers["x-request-id"]
    assert len(made) == 12 and kept == "frontend-123" and replaced != "bad id\nwith newline" and failed