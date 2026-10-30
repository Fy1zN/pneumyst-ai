from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from backend.app.api.deps import get_storage
from backend.app.core.errors import ApiError
from backend.app.schemas.analysis import ErrorResponse
from backend.app.storage.files import LocalStorage

router = APIRouter(tags=["files"])


# No Clerk check here: images load through <img> tags, so the signed link is the proof of access.
@router.get("/files/{key:path}", response_class=FileResponse,
            responses={200: {"content": {"image/png": {}}}, 403: {"model": ErrorResponse},
                       404: {"model": ErrorResponse}})
async def get_file(key: str, expires: int = Query(...), sig: str = Query(...),
                   storage: LocalStorage | None = Depends(get_storage)) -> FileResponse:
    if storage is None or not storage.verify(key, expires, sig):
        raise ApiError(403, "FORBIDDEN", "link is invalid or expired")
    path = storage.path(key)
    if not path.is_file():
        raise ApiError(404, "FILE_NOT_FOUND", "file no longer exists")
    return FileResponse(path, media_type="image/png",
                        headers={"Cache-Control": f"private, max-age={storage.ttl_s}",
                                 "X-Content-Type-Options": "nosniff"})