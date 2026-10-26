"""Errors returned to API clients as {"error_code", "detail", "analysis_id"}."""
from __future__ import annotations


class ApiError(Exception):
    def __init__(self, status_code: int, error_code: str, detail: str, analysis_id: str | None = None):
        super().__init__(detail)
        self.status_code = status_code
        self.error_code = error_code
        self.detail = detail
        self.analysis_id = analysis_id