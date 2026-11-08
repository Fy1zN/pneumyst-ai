"""Report summaries written off the request path, one job per analysis, with a deadline for whoever waits.

A job started right after /analyze is usually done before anyone opens the report. A request that arrives while
the job runs waits for that same job instead of starting a second LLM call. If the wait passes the deadline, the
caller gets the template at once and the job keeps running; when it finishes, its result is saved for next time.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from backend.app.agent.facts import build_facts
from backend.app.agent.report_graph import ReportResult, ReportWriter, template_summary
from backend.app.db.models import AnalysisRecord
from backend.app.repositories.analyses import AnalysisStore

logger = logging.getLogger("pneumyst.reports")


class ReportJobs:
    def __init__(self, writer: ReportWriter, store: AnalysisStore, deadline_s: float, workers: int = 2):
        self.writer = writer
        self.store = store
        self.deadline_s = deadline_s
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="report")
        self._running: dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(self, user_id: str, analysis_id: str, findings: list[dict], errors: dict) -> Future:
        with self._lock:
            job = self._running.get(analysis_id)
            if job is None:
                job = self._pool.submit(self._write, user_id, analysis_id, findings, errors)
                self._running[analysis_id] = job
                job.add_done_callback(lambda _: self._forget(analysis_id))
            return job

    def _forget(self, analysis_id: str) -> None:
        with self._lock:
            self._running.pop(analysis_id, None)

    def _write(self, user_id: str, analysis_id: str, findings: list[dict], errors: dict) -> ReportResult:
        result = self.writer.write(build_facts(findings, errors))
        self.store.save_report(user_id, analysis_id, result)
        return result

    def wait(self, user_id: str, row: AnalysisRecord) -> ReportResult:
        """The summary from the job for this analysis; raises TimeoutError when it takes longer than the deadline."""
        return self.submit(user_id, row.id, row.findings, row.errors).result(timeout=self.deadline_s)

    @staticmethod
    def late(row: AnalysisRecord) -> ReportResult:
        """What a caller gets when the deadline passes: the template, never saved."""
        text = template_summary(build_facts(row.findings, row.errors))
        return ReportResult(text=text, source="template", model=None, attempts=0, reason="LLM_TIMEOUT")

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)