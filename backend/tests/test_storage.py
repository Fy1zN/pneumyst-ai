"""Stored images, signed links, per-user access, deletion and storage failures."""
from __future__ import annotations

import io
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from backend.app.core.config import Settings
from backend.app.db.session import Database
from backend.app.main import make_storage
from backend.app.repositories.analyses import AnalysisStore
from backend.app.storage.files import LocalStorage
from backend.tests.conftest import ConstModel, make_client, make_service, memory_store
from ml.datasets.targets import NIH_FINDINGS, TB

KEY = "k" * 32


class SpotModel(nn.Module):
    """One output: high when the bright spot near the top-left of the image is present."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, :1, 24:72, 40:88].mean(dim=(1, 2, 3)) * 1.5 - 1).unsqueeze(1)


class Clock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def spot_xray() -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.normal(120, 30, (400, 360)).clip(0, 255).astype(np.uint8)
    arr[40:80, 40:80] = 250
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def spot_service():
    return make_service(ConstModel([0.1] * len(NIH_FINDINGS)), SpotModel())


def upload(client, **params):
    return client.post("/analyze", files={"file": ("ramesh_kumar.png", spot_xray(), "image/png")}, params=params)


def fetch(client, url: str):
    return client.get(urlsplit(url).path, params={k: v[0] for k, v in parse_qs(urlsplit(url).query).items()})


@pytest.fixture
def storage(tmp_path):
    return LocalStorage(tmp_path / "storage", KEY, ttl_s=300, clock=Clock())


def test_analyze_stores_image_and_shown_heatmaps_only(storage):
    with make_client(memory_store(), service=spot_service(), storage=storage) as c:
        body = upload(c).json()
        assert body["stages"]["images"] == "success" and "images" not in body["errors"]
        shown = {f["finding"] for f in body["findings"] if f["heatmap"] == "shown"}
        assert shown == {TB} and set(body["images"]["heatmaps"]) == shown
        image = fetch(c, body["images"]["image"])
        heatmap = fetch(c, body["images"]["heatmaps"][TB])
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(image.content)) as im:
        assert im.size == (320, 320) and im.mode == "L" and not im.info and not im.text
    with Image.open(io.BytesIO(heatmap.content)) as im:
        assert im.size == (320, 320) and im.mode == "RGB"
    folder = storage.root / "analyses" / body["id"]
    assert sorted(p.name for p in folder.rglob("*.png")) == ["Tuberculosis.png", "image.png"]
    assert b"ramesh" not in b"".join(p.read_bytes() for p in folder.rglob("*.png"))


def test_explain_false_stores_only_the_image(storage):
    with make_client(memory_store(), service=spot_service(), storage=storage) as c:
        body = upload(c, explain="false").json()
    assert body["images"]["heatmaps"] == {}
    assert [p.name for p in (storage.root / "analyses" / body["id"]).rglob("*.png")] == ["image.png"]


def test_history_gives_fresh_links_and_keys_stay_private(storage):
    with make_client(memory_store(), service=spot_service(), storage=storage) as c:
        body = upload(c).json()
        storage.clock.now += 100
        saved = c.get(f"/analyses/{body['id']}").json()
        assert fetch(c, saved["images"]["heatmaps"][TB]).status_code == 200
    assert saved["images"]["image"] != body["images"]["image"]
    assert "heatmap_keys" not in saved and all(f["heatmap"] for f in saved["findings"])


@pytest.mark.parametrize("change", ["expired", "signature", "other_file", "traversal", "missing_sig"])
def test_bad_links_are_refused(storage, change):
    with make_client(memory_store(), service=spot_service(), storage=storage) as c:
        body = upload(c).json()
        url = urlsplit(body["images"]["image"])
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        path = url.path
        if change == "expired":
            storage.clock.now += 301
        elif change == "signature":
            params["sig"] = "0" * 64
        elif change == "other_file":
            path = path.replace("image.png", f"heatmaps/{TB}.png")
        elif change == "traversal":
            path = path.replace("image.png", "..%2F..%2F..%2F..%2F.env")
        else:
            params.pop("sig")
        r = c.get(path, params=params)
    assert r.status_code in (403, 404, 422) and r.headers["content-type"] == "application/json"
    if change != "missing_sig":
        assert r.status_code == 403 and r.json()["error_code"] == "FORBIDDEN"


def test_signed_link_survives_only_for_its_lifetime(storage):
    url = storage.url(f"analyses/{'0' * 8}-0000-0000-0000-{'0' * 12}/image.png")
    params = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
    key = urlsplit(url).path.removeprefix("/files/")
    assert storage.verify(key, int(params["expires"]), params["sig"])
    storage.clock.now += 300
    assert storage.verify(key, int(params["expires"]), params["sig"])
    storage.clock.now += 1
    assert not storage.verify(key, int(params["expires"]), params["sig"])


def test_other_users_get_no_links(storage):
    store = memory_store()
    with make_client(store, user="alice", service=spot_service(), storage=storage) as alice:
        analysis_id = upload(alice).json()["id"]
    with make_client(store, user="bob", service=spot_service(), storage=storage) as bob:
        r = bob.get(f"/analyses/{analysis_id}")
    assert r.status_code == 404 and "images" not in r.json()


def test_delete_removes_files(storage):
    with make_client(memory_store(), service=spot_service(), storage=storage) as c:
        body = upload(c).json()
        assert (storage.root / "analyses" / body["id"]).exists()
        assert c.delete(f"/analyses/{body['id']}").status_code == 204
        assert fetch(c, body["images"]["image"]).status_code == 404
    assert not (storage.root / "analyses" / body["id"]).exists()


def test_storage_failure_keeps_results(storage, monkeypatch):
    def broken(key, data):
        raise OSError("disk full")

    monkeypatch.setattr(storage, "put", broken)
    with make_client(memory_store(), service=spot_service(), storage=storage) as c:
        body = upload(c).json()
        saved = c.get(f"/analyses/{body['id']}").json()
    assert body["persistence"] == "saved" and len(body["findings"]) == 15
    assert body["stages"]["images"] == "failed" and body["errors"]["images"] == "STORAGE_FAILED"
    assert body["images"] is None and saved["images"] is None
    assert not (storage.root / "analyses" / body["id"]).exists()


def test_database_failure_removes_stored_images(storage):
    broken = AnalysisStore(Database("sqlite:////nonexistent-folder/pneumyst.db"))
    with make_client(broken, service=spot_service(), storage=storage) as c:
        body = upload(c).json()
    assert body["persistence"] == "not_saved" and body["images"] is None and "images" not in body["stages"]
    assert not (storage.root / "analyses").exists() or not any((storage.root / "analyses").iterdir())


def test_without_storage_results_are_saved_without_images():
    with make_client(memory_store(), service=spot_service()) as c:
        body = upload(c).json()
    assert body["persistence"] == "saved" and body["images"] is None
    assert body["stages"]["images"] == "skipped" and body["errors"]["images"] == "STORAGE_NOT_CONFIGURED"


def test_storage_needs_a_signing_key(tmp_path):
    assert make_storage(Settings()) is None
    for key in (None, "short"):
        with pytest.raises(RuntimeError, match="FILE_SIGNING_KEY"):
            make_storage(Settings(storage_dir=tmp_path, file_signing_key=key))
    assert isinstance(make_storage(Settings(storage_dir=tmp_path, file_signing_key=KEY)), LocalStorage)


def test_files_route_without_storage_is_forbidden():
    with make_client(service=spot_service()) as c:
        r = c.get(f"/files/analyses/{'0' * 8}-0000-0000-0000-{'0' * 12}/image.png", params={"expires": 1, "sig": "x"})
        assert c.app.version == "0.4.0" and r.status_code == 403