"""PNG files of one analysis: the 320 px image the models saw, and an overlay for every heatmap that is shown.

The original upload is never stored. PNGs are encoded from pixel arrays, so no metadata or file name survives.
"""
from __future__ import annotations

import io
import logging
import time

import numpy as np
from PIL import Image

from backend.app.services.screening import Analysis
from backend.app.storage.files import LocalStorage

logger = logging.getLogger("pneumyst.storage")
STAGE = "images"


def png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def save_images(storage: LocalStorage, analysis: Analysis) -> dict:
    """Store the images, record the outcome as the "images" stage, and return the keys (empty on failure)."""
    res = analysis.result
    t0 = time.perf_counter()
    prefix = f"analyses/{analysis.id}"
    keys = {"image": f"{prefix}/image.png", "heatmaps": {name: f"{prefix}/heatmaps/{name}.png" for name in res.maps}}
    try:
        storage.put(keys["image"], png(res.image))
        for name, key in keys["heatmaps"].items():
            storage.put(key, png(res.overlay(name)))
    except (OSError, ValueError):
        logger.exception("images of analysis %s could not be stored", analysis.id)
        remove_images(storage, analysis.id)
        res.stages[STAGE], res.errors[STAGE] = "failed", "STORAGE_FAILED"
        return {}
    finally:
        res.timings_ms[STAGE] = (time.perf_counter() - t0) * 1000
    res.stages[STAGE] = "success"
    return keys


def remove_images(storage: LocalStorage, analysis_id: str) -> None:
    try:
        storage.delete_analysis(analysis_id)
    except OSError:
        logger.exception("images of analysis %s could not be deleted", analysis_id)


def image_links(storage: LocalStorage | None, keys: dict | None, base_url: str) -> dict | None:
    if storage is None or not keys:
        return None
    base = base_url.rstrip("/")
    return {"image": base + storage.url(keys["image"]),
            "heatmaps": {name: base + storage.url(key) for name, key in keys["heatmaps"].items()}}