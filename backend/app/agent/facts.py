"""The facts a report is written from: findings by band, formatted scores and anything that did not work.

This is everything the LLM sees. It holds finding names and numbers only, never the image or personal data.
"""
from __future__ import annotations

ERROR_TEXT = {"MODEL_UNAVAILABLE": "was unavailable", "INFERENCE_FAILED": "failed"}


def pct(x: float) -> str:
    """Scores as short percentages; small values keep enough digits to stay distinct from zero."""
    v = x * 100
    if v >= 10:
        return f"{v:.0f}%"
    return f"{v:.1f}%" if v >= 1 else f"{v:.2f}%"


def build_facts(findings: list[dict], errors: dict[str, str]) -> dict:
    facts = {"likely": [], "possible": [], "unlikely": [], "not_banded": [], "unavailable": [], "problems": []}
    for f in findings:
        name = f["display_name"]
        if f["status"] != "success" or f["score"] is None:
            facts["unavailable"].append(name)
        elif f["band"] in ("likely", "possible"):
            facts[f["band"]].append({"finding": name, "score": pct(f["score"]),
                                     "possible_threshold": pct(f["possible_threshold"]),
                                     "likely_threshold": pct(f["likely_threshold"])})
        elif f["band"] == "unlikely":
            facts["unlikely"].append(name)
        else:
            facts["not_banded"].append({"finding": name, "score": pct(f["score"])})
    for stage, code in errors.items():
        if code.startswith(tuple(ERROR_TEXT)):
            facts["problems"].append(f"the {stage} model {ERROR_TEXT[code.split(':')[0]]}")
        elif stage.startswith("explain:"):
            facts["problems"].append(f"heatmaps from the {stage.removeprefix('explain:')} model could not be made")
    for band in ("likely", "possible"):
        facts[band].sort(key=lambda f: -float(f["score"].rstrip("%")))
    return facts