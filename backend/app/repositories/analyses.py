"""Saved analyses. Every query is filtered by the owner, so one user can never read another user's data."""
from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from backend.app.db.models import AnalysisRecord, AuditEventRecord, UserRecord
from backend.app.db.session import Database
from backend.app.services.screening import Analysis


class AnalysisStore:
    def __init__(self, db: Database):
        self.db = db

    def save(self, user_id: str, analysis: Analysis, models: list[dict]) -> None:
        result = analysis.result.to_dict()
        try:
            self._save(user_id, analysis, result, models)
        except IntegrityError:
            # Another request created the same new user at the same moment; the user row now exists.
            self._save(user_id, analysis, result, models)

    def _save(self, user_id: str, analysis: Analysis, result: dict, models: list[dict]) -> None:
        timings = result["timings_ms"]
        with self.db.session.begin() as s:
            if s.get(UserRecord, user_id) is None:
                s.add(UserRecord(id=user_id))
                s.flush()
            s.add(AnalysisRecord(
                id=analysis.id, user_id=user_id, created_at=analysis.created_at, status=result["status"],
                image_sha256=analysis.image_sha256, image_format=analysis.image_format,
                findings=result["findings"], stages=result["stages"], errors=result["errors"], models=models,
                preprocess=result["preprocess"], timings_ms=timings))
            s.flush()
            for step, status in result["stages"].items():
                s.add(AuditEventRecord(
                    analysis_id=analysis.id, step=step, status=status, error_code=result["errors"].get(step),
                    latency_ms=timings.get("preprocess" if step == "input" else step),
                    created_at=analysis.created_at))

    def history(self, user_id: str, limit: int, offset: int) -> tuple[list[AnalysisRecord], int]:
        with self.db.session() as s:
            total = s.scalar(select(func.count()).select_from(AnalysisRecord)
                             .where(AnalysisRecord.user_id == user_id))
            rows = s.scalars(select(AnalysisRecord).where(AnalysisRecord.user_id == user_id)
                             .order_by(AnalysisRecord.created_at.desc(), AnalysisRecord.id.desc())
                             .limit(limit).offset(offset)).all()
        return list(rows), int(total or 0)

    def get(self, user_id: str, analysis_id: str) -> AnalysisRecord | None:
        with self.db.session() as s:
            return s.scalar(select(AnalysisRecord).where(AnalysisRecord.id == analysis_id,
                                                         AnalysisRecord.user_id == user_id))

    def audit_events(self, user_id: str, analysis_id: str) -> list[AuditEventRecord]:
        with self.db.session() as s:
            return list(s.scalars(select(AuditEventRecord).join(AnalysisRecord)
                                  .where(AnalysisRecord.id == analysis_id, AnalysisRecord.user_id == user_id)
                                  .order_by(AuditEventRecord.id)).all())

    def delete(self, user_id: str, analysis_id: str) -> bool:
        with self.db.session.begin() as s:
            owned = s.scalar(select(AnalysisRecord.id).where(AnalysisRecord.id == analysis_id,
                                                             AnalysisRecord.user_id == user_id))
            if owned is None:
                return False
            s.execute(delete(AuditEventRecord).where(AuditEventRecord.analysis_id == analysis_id))
            s.execute(delete(AnalysisRecord).where(AnalysisRecord.id == analysis_id))
        return True