"""Assistant agent: tool routing, user scoping, guardrails, step limit, failures and what gets audited."""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy import select

from backend.app.agent.assistant import Assistant
from backend.app.agent.facts import pct
from backend.app.agent.tools import make_tools
from backend.app.db.models import AuditEventRecord
from backend.tests.conftest import ConstModel, make_client, make_service, memory_store
from backend.tests.test_exports import upload
from backend.tests.test_storage import spot_service
from ml.datasets.targets import NIH_FINDINGS

CASE = {"case_number": "CASE-0042", "night_sweats": "yes", "previous_tb": "yes"}


class ToolScript:
    """Stands in for a tool-calling chat model: each turn returns the next scripted reply.

    A list of (tool, args) pairs becomes tool calls, a string becomes the final answer, an exception is raised.
    """

    def __init__(self, *turns, model_name: str = "test-model"):
        self.turns = list(turns)
        self.model_name = model_name
        self.seen: list[list] = []
        self.bound: list[str] = []

    def bind_tools(self, tools):
        self.bound = [t.name for t in tools]
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        turn = self.turns.pop(0) if self.turns else [("get_model_card", {})]
        if isinstance(turn, Exception):
            raise turn
        if isinstance(turn, list):
            calls = [{"name": name, "args": args, "id": f"call-{len(self.seen)}-{i}"}
                     for i, (name, args) in enumerate(turn)]
            return AIMessage(content="", tool_calls=calls)
        return AIMessage(content=turn, response_metadata={"model_name": self.model_name})

    def tool_results(self) -> list[str]:
        return [str(m.content) for m in self.seen[-1] if isinstance(m, ToolMessage)]


def ask(client, analysis_id: str, question: str = "Why is tuberculosis flagged?"):
    return client.post(f"/analyses/{analysis_id}/ask", json={"question": question})


def ask_events(store, analysis_id: str) -> list[tuple]:
    with store.db.session() as s:
        rows = s.scalars(select(AuditEventRecord).where(AuditEventRecord.analysis_id == analysis_id,
                                                        AuditEventRecord.step.like("ask%"))
                         .order_by(AuditEventRecord.id)).all()
    return [(e.step, e.status, e.error_code) for e in rows]


def test_answer_grounded_in_tool_results():
    store = memory_store()
    with make_client(store, service=spot_service()) as c:
        analysis_id = upload(c, CASE).json()["id"]
    tb = next(f for f in store.get("local-dev", analysis_id).findings if f["finding"] == "Tuberculosis")
    answer = f"The model scored Tuberculosis at {pct(tb['score'])} as {tb['band']}, so its heatmap is shown."
    model = ToolScript([("get_analysis", {})], answer)
    with make_client(store, service=spot_service(), assistant=Assistant(model, "test-model")) as c:
        body = ask(c, analysis_id).json()
    assert body == {"analysis_id": analysis_id, "answer": answer, "refused": False, "tools_used": ["get_analysis"],
                    "model": "test-model", "disclaimer": body["disclaimer"]} and "not a diagnosis" in body["disclaimer"]
    assert model.bound == ["get_analysis", "list_history", "compare_analyses", "get_finding_info", "get_model_card"]
    result = json.loads(model.tool_results()[0])
    assert result["analysis_id"] == analysis_id and len(result["findings"]) == 15
    assert ask_events(store, analysis_id) == [("ask:get_analysis", "success", None), ("ask", "answered", None)]


def test_case_context_and_question_are_never_exposed():
    store = memory_store()
    with make_client(store, service=spot_service()) as c:
        analysis_id = upload(c, CASE).json()["id"]
    model = ToolScript([("get_analysis", {}), ("list_history", {})], "The model scored Tuberculosis as likely.")
    question = "My name is Ramesh and I have night sweats, is it TB?"
    with make_client(store, service=spot_service(), assistant=Assistant(model)) as c:
        ask(c, analysis_id, question)
    exposed = " ".join(model.tool_results()).lower()
    assert "case-0042" not in exposed and "night sweats" not in exposed and "previous_tb" not in exposed
    with store.db.session() as s:
        stored = " ".join(f"{e.step} {e.status} {e.error_code}" for e in s.scalars(select(AuditEventRecord)))
    assert "Ramesh" not in stored and "night" not in stored


def test_invented_numbers_and_medical_advice_are_refused():
    store = memory_store()
    with make_client(store, service=spot_service()) as c:
        analysis_id = upload(c).json()["id"]
    for answer in ("The model scored Tuberculosis at 99% as likely.", "You have tuberculosis; start antibiotics."):
        model = ToolScript([("get_analysis", {})], answer)
        with make_client(store, service=spot_service(), assistant=Assistant(model)) as c:
            body = ask(c, analysis_id).json()
        assert body["refused"] and body["model"] is None and "qualified clinician" in body["answer"]
    assert ask_events(store, analysis_id)[-1] == ("ask", "refused", "CHECKS_FAILED")


def test_tools_cannot_read_another_users_analysis():
    store = memory_store()
    with make_client(store, user="alice", service=spot_service()) as alice:
        alice_id = upload(alice).json()["id"]
    with make_client(store, user="bob", service=spot_service()) as bob:
        bob_id = upload(bob).json()["id"]
    model = ToolScript([("get_analysis", {"analysis_id": alice_id}), ("compare_analyses",
                                                                      {"other_analysis_id": alice_id})],
                       "I could not find that analysis.")
    with make_client(store, user="bob", service=spot_service(), assistant=Assistant(model)) as bob:
        assert ask(bob, alice_id).status_code == 404
        assert not ask(bob, bob_id).json()["refused"]
    assert all("no such analysis" in r for r in model.tool_results())


def test_step_limit_stops_endless_tool_calls():
    store = memory_store()
    with make_client(store, service=spot_service()) as c:
        analysis_id = upload(c).json()["id"]
    model = ToolScript()  # keeps asking for the model card forever
    with make_client(store, service=spot_service(), assistant=Assistant(model, max_turns=3)) as c:
        body = ask(c, analysis_id).json()
    assert body["refused"] and len(model.seen) <= 3
    assert ask_events(store, analysis_id)[-1] == ("ask", "refused", "STEP_LIMIT")


def test_llm_failure_and_missing_assistant_give_503():
    store = memory_store()
    with make_client(store, service=spot_service()) as c:
        analysis_id = upload(c).json()["id"]
        missing = ask(c, analysis_id)
    with make_client(store, service=spot_service(), assistant=Assistant(ToolScript(TimeoutError("busy")))) as c:
        failed = ask(c, analysis_id)
    assert missing.status_code == 503 and missing.json()["detail"] == "the assistant is not configured"
    assert failed.status_code == 503 and failed.json()["error_code"] == "ASSISTANT_UNAVAILABLE"
    assert ask_events(store, analysis_id) == [("ask", "failed", "LLM_FAILED")]


def test_question_length_is_limited():
    store = memory_store()
    with make_client(store, service=spot_service(), assistant=Assistant(ToolScript("unused"))) as c:
        analysis_id = upload(c).json()["id"]
        for question in ("?", "x" * 501):
            r = ask(c, analysis_id, question)
            assert r.status_code == 422 and r.json()["error_code"] == "INVALID_REQUEST"


def test_history_compare_and_knowledge_tools():
    store = memory_store()
    quiet = make_service(ConstModel([0.1] * len(NIH_FINDINGS)), ConstModel([0.05]))
    flagged = make_service(ConstModel([0.8] + [0.1] * (len(NIH_FINDINGS) - 1)), ConstModel([0.05]))
    with make_client(store, service=quiet) as c:
        first = upload(c).json()["id"]
    with make_client(store, service=flagged) as c:
        second = upload(c).json()["id"]
    tools = {t.name: t for t in make_tools(store, "local-dev", second)}
    history = json.loads(tools["list_history"].invoke({}))
    assert history["total"] == 2 and history["analyses"][0]["flagged"] == ["Atelectasis (likely)"]
    changes = json.loads(tools["compare_analyses"].invoke({"other_analysis_id": first}))["band_changes"]
    assert changes == [{"finding": "Atelectasis", "earlier": "unlikely", "now": "likely", "earlier_score": "10%",
                        "score_now": "80%"}]
    info = json.loads(tools["get_finding_info"].invoke({"finding": "pleural_thickening"}))
    assert info["finding"] == "Pleural thickening" and "0.781" in info["model_reliability"]
    assert "known_findings" in json.loads(tools["get_finding_info"].invoke({"finding": "flu"}))
    assert "0.813" in tools["get_model_card"].invoke({})
    assert "no such analysis" in tools["get_analysis"].invoke({"analysis_id": "not-a-uuid"})