"""Read-only tools for the assistant, bound to the signed-in user.

The user id comes from the verified session, never from the model, so a tool can only ever read that user's
analyses. Case context is left out of every result: symptoms and patient details never reach the LLM.
"""
from __future__ import annotations

import json
import uuid

from langchain_core.tools import BaseTool, tool

from backend.app.agent.facts import build_facts, pct
from backend.app.agent.knowledge import FINDINGS, MODEL_CARD
from backend.app.db.models import AnalysisRecord
from backend.app.repositories.analyses import AnalysisStore

NOT_FOUND = json.dumps({"error": "no such analysis for this user"})


def band_of(f: dict) -> str:
    if f["score"] is None:
        return "unavailable"
    return f["band"] or "not banded"


def describe(row: AnalysisRecord) -> dict:
    findings = [{"finding": f["display_name"], "band": band_of(f),
                 "score": pct(f["score"]) if f["score"] is not None else None,
                 "possible_from": pct(f["possible_threshold"]) if f["possible_threshold"] is not None else None,
                 "likely_from": pct(f["likely_threshold"]) if f["likely_threshold"] is not None else None,
                 "heatmap": f["heatmap"]} for f in row.findings]
    return {"analysis_id": row.id, "date": row.created_at.date().isoformat(), "status": row.status,
            "findings": findings, "problems": build_facts(row.findings, row.errors)["problems"]}


def make_tools(store: AnalysisStore, user_id: str, current_id: str) -> list[BaseTool]:
    def load(analysis_id: str) -> AnalysisRecord | None:
        try:
            return store.get(user_id, str(uuid.UUID(analysis_id)))
        except ValueError:
            return None

    @tool
    def get_analysis(analysis_id: str = current_id) -> str:
        """Findings with band, score and thresholds, plus processing problems, for one of the user's analyses.
        Defaults to the analysis the user is asking about."""
        row = load(analysis_id)
        return json.dumps(describe(row)) if row else NOT_FOUND

    @tool
    def list_history(limit: int = 5) -> str:
        """The user's most recent analyses (newest first) with their date and flagged findings."""
        rows, total = store.history(user_id, max(1, min(limit, 10)), 0)
        items = [{"analysis_id": r.id, "date": r.created_at.date().isoformat(),
                  "flagged": [f"{f['display_name']} ({f['band']})" for f in r.findings
                              if f["band"] in ("likely", "possible")]} for r in rows]
        return json.dumps({"total": total, "analyses": items})

    @tool
    def compare_analyses(other_analysis_id: str, analysis_id: str = current_id) -> str:
        """Findings whose band changed between two of the user's analyses, with both scores."""
        first, second = load(other_analysis_id), load(analysis_id)
        if first is None or second is None:
            return NOT_FOUND
        before = {f["finding"]: f for f in first.findings}
        changes = []
        for f in second.findings:
            old = before.get(f["finding"])
            if old is not None and band_of(old) != band_of(f):
                changes.append({"finding": f["display_name"], "earlier": band_of(old), "now": band_of(f),
                                "earlier_score": pct(old["score"]) if old["score"] is not None else None,
                                "score_now": pct(f["score"]) if f["score"] is not None else None})
        return json.dumps({"earlier": {"analysis_id": first.id, "date": first.created_at.date().isoformat()},
                           "now": {"analysis_id": second.id, "date": second.created_at.date().isoformat()},
                           "band_changes": changes})

    @tool
    def get_finding_info(finding: str) -> str:
        """What a finding means and how reliable its model is, from the project's own evaluation."""
        name = next((n for n in FINDINGS if n.lower() == finding.replace("_", " ").strip().lower()), None)
        if name is None:
            return json.dumps({"error": "unknown finding", "known_findings": list(FINDINGS)})
        meaning, reliability = FINDINGS[name]
        return json.dumps({"finding": name, "meaning": meaning, "model_reliability": reliability})

    @tool
    def get_model_card() -> str:
        """How the screening models were trained and tested, how bands and heatmaps work, and known limits."""
        return json.dumps(MODEL_CARD)

    return [get_analysis, list_history, compare_analyses, get_finding_info, get_model_card]