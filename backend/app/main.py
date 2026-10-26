"""PNEUMYST AI API. Run from the repository root: uvicorn backend.app.main:app"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api import analyze, health
from backend.app.core.config import Settings, load_settings
from backend.app.core.errors import ApiError
from backend.app.schemas.analysis import ErrorResponse
from backend.app.services.screening import ScreeningService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pneumyst")


def create_app(settings: Settings | None = None, screening: ScreeningService | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.screening = screening or ScreeningService.load(settings)
        for m in app.state.screening.models:
            logger.info("model %s: %s", m.name, "loaded" if m.available else m.error)
        yield

    app = FastAPI(title="PNEUMYST AI", version="0.1.0", lifespan=lifespan,
                  description="Research screening API for chest X-rays. Not a diagnostic device.")
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["Authorization", "Content-Type"])
    app.include_router(health.router)
    app.include_router(analyze.router)

    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError) -> JSONResponse:
        body = ErrorResponse(error_code=exc.error_code, detail=exc.detail, analysis_id=exc.analysis_id)
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        body = ErrorResponse(error_code="INTERNAL_ERROR", detail="unexpected server error")
        return JSONResponse(status_code=500, content=body.model_dump())

    return app


app = create_app()