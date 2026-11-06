"""The audit record of one analysis: everything needed to trace a reported number back to how it was computed.

Storage keys and signed links are left out; the record says which images exist, not where they are.
"""
from __future__ import annotations

from datetime import datetime

from backend.app.db.models import AnalysisRecord, AuditEventRecord
from backend.app.services.case_notes import RULES_VERSION, case_notes

AUDIT_VERSION = 2  # raise when the layout changes, so older records can still be read correctly


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def ordered_stages(stages: dict[str, str]) -> dict[str, str]:
    """Pipeline order (input, models, heatmaps, images), since MySQL does not keep the order of JSON keys."""
    def rank(name: str) -> int:
        if name == "input":
            return 0
        if name.startswith("explain:"):
            return 2
        return 3 if name == "images" else 1
    return dict(sorted(stages.items(), key=lambda item: (rank(item[0]), item[0])))


def build_audit(row: AnalysisRecord, events: list[AuditEventRecord], app_version: str,
                generated_at: datetime) -> dict:
    keys = row.heatmap_keys or {}
    report = None
    if row.report is not None:
        report = {"summary": row.report, "source": row.report_source, "model": row.report_model,
                  "created_at": iso(row.report_created_at)}
    return {
        "audit_version": AUDIT_VERSION,
        "app_version": app_version,
        "generated_at": iso(generated_at),
        "analysis": {"id": row.id, "user_id": row.user_id, "created_at": iso(row.created_at), "status": row.status,
                     "image_sha256": row.image_sha256, "image_format": row.image_format},
        "case_context": row.case_context,
        "case_notes": {"rules_version": RULES_VERSION, **notes} if (notes := case_notes(row.case_context)) else None,
        "models": row.models,
        "preprocess": row.preprocess,
        "findings": row.findings,
        "stages": ordered_stages(row.stages),
        "errors": row.errors,
        "timings_ms": row.timings_ms,
        "stored_images": {"image": "image" in keys, "heatmaps": sorted(keys.get("heatmaps", {}))},
        "report": report,
        "events": [{"step": e.step, "status": e.status, "error_code": e.error_code, "latency_ms": e.latency_ms,
                    "created_at": iso(e.created_at)} for e in events],
    }