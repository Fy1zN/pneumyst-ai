"""Report workflow as a LangGraph graph: the LLM drafts, rule-based checks review, and a template is the fallback.

    start -> draft -> check -> done (source "llm")
               ^        |
               +--------+  issues fed back, at most max_attempts drafts
    draft or check gives up -> template -> done (source "template")
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from backend.app.agent.checks import check_summary

logger = logging.getLogger("pneumyst.report")

DISCLAIMER = ("Research prototype. This screening summary is not a diagnosis and must be reviewed by a qualified "
              "clinician together with the image and the patient's history.")

RULES = """You write the summary section of a chest X-ray screening report produced by AI models.
You receive the screening results as JSON. Write one plain paragraph of 40 to 150 words for a clinician.
Rules:
- Mention every finding in "likely" and "possible" by its exact name, likely ones first, with its score as given.
- Do not name findings from "unlikely" individually unless you call them unlikely in the same sentence.
- Use only numbers that appear in the JSON, written exactly as given.
- If "unavailable" or "problems" is not empty, say plainly which parts were unavailable or failed.
- Phrase results as "the model scored X as likely", never as findings or conditions that are present.
- Do not diagnose, confirm or rule out anything, and give no treatment or medication advice.
- Do not add a disclaimer, title, list or markdown; one paragraph only."""


class ReportState(TypedDict, total=False):
    facts: dict
    draft: str
    issues: list[str]
    attempts: int
    text: str
    source: str
    reason: str
    model_used: str


@dataclass
class ReportResult:
    text: str
    source: str                       # "llm" | "template"
    model: str | None
    attempts: int
    reason: str | None = None         # why the template was used; LLM_FAILED reports are not kept
    issues: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


def template_summary(facts: dict) -> str:
    def listed(items: list[dict]) -> str:
        return ", ".join(f"{f['finding']} ({f['score']})" for f in items)

    parts = []
    if facts["likely"]:
        parts.append(f"The screening models scored these findings as likely: {listed(facts['likely'])}.")
    if facts["possible"]:
        parts.append(f"These findings were scored as possible: {listed(facts['possible'])}.")
    if not facts["likely"] and not facts["possible"]:
        parts.append("No finding reached the possible threshold in this screening.")
    if facts["unlikely"]:
        parts.append(f"The remaining {len(facts['unlikely'])} banded findings were scored unlikely.")
    if facts["not_banded"]:
        parts.append(f"Scores without a reliable band are reported but not interpreted: {listed(facts['not_banded'])}.")
    if facts["unavailable"]:
        parts.append(f"Results were unavailable for: {', '.join(facts['unavailable'])}.")
    for problem in facts["problems"]:
        if "could not" in problem:
            parts.append(f"Note: {problem}.")
    parts.append("Each band compares a calibrated model score with thresholds fixed on validation data, "
                 "and the heatmaps show which image regions changed each score most.")
    return " ".join(parts)


class ReportWriter:
    def __init__(self, model: Runnable | None, model_name: str | None = None, max_attempts: int = 2):
        self.model = model
        self.model_name = model_name if model is not None else None
        self.max_attempts = max_attempts
        self.graph = self._build()

    def _build(self):
        g = StateGraph(ReportState)
        g.add_node("draft", self._draft)
        g.add_node("check", self._check)
        g.add_node("template", self._template)
        g.add_conditional_edges(START, lambda s: "draft" if self.model is not None else "template")
        g.add_conditional_edges("draft", lambda s: "template" if s.get("reason") else "check")
        g.add_conditional_edges("check", self._after_check)
        g.add_edge("template", END)
        return g.compile()

    def _draft(self, state: ReportState) -> ReportState:
        attempts = state.get("attempts", 0) + 1
        prompt = "Screening results:\n" + json.dumps(state["facts"], indent=2)
        if state.get("issues"):
            prompt += ("\n\nYour previous summary was rejected for these reasons:\n- " + "\n- ".join(state["issues"])
                       + "\n\nPrevious summary:\n" + state["draft"] + "\n\nWrite a corrected summary.")
        try:
            reply = self.model.invoke([SystemMessage(RULES), HumanMessage(prompt)])
        except Exception as exc:  # any provider error ends the LLM path; the template still produces a report
            logger.warning("report LLM call failed: %s: %s", type(exc).__name__, exc)
            return {"attempts": attempts, "reason": "LLM_FAILED"}
        # With a fallback model configured, the reply names the model that actually answered.
        used = reply.response_metadata.get("model_name") or self.model_name
        return {"attempts": attempts, "draft": reply.text.strip(), "model_used": used.removeprefix("models/")}

    def _check(self, state: ReportState) -> ReportState:
        issues = check_summary(state["draft"], state["facts"])
        if issues:
            logger.info("report draft %d rejected: %s", state["attempts"], "; ".join(issues))
            return {"issues": issues}
        return {"issues": [], "text": state["draft"], "source": "llm"}

    def _after_check(self, state: ReportState) -> str:
        if not state["issues"]:
            return END
        return "draft" if state["attempts"] < self.max_attempts else "template"

    def _template(self, state: ReportState) -> ReportState:
        reason = state.get("reason") or ("CHECKS_FAILED" if self.model is not None else "LLM_NOT_CONFIGURED")
        return {"text": template_summary(state["facts"]), "source": "template", "reason": reason}

    def write(self, facts: dict) -> ReportResult:
        t0 = time.perf_counter()
        out = self.graph.invoke({"facts": facts, "attempts": 0})
        return ReportResult(text=out["text"], source=out["source"],
                            model=out.get("model_used") if out["source"] == "llm" else None,
                            attempts=out.get("attempts", 0), reason=out.get("reason"),
                            issues=out.get("issues", []), latency_ms=(time.perf_counter() - t0) * 1000)