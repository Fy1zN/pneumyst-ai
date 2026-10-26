"""Saving analyses, history, deletion, per-user isolation and database failures."""
from __future__ import annotations

from datetime import datetime

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, select

from backend.app.core.config import REPO
from backend.app.db.models import AuditEventRecord, Base
from backend.app.db.session import Database
from backend.app.repositories.analyses import AnalysisStore
from backend.tests.conftest import ConstModel, make_client, make_service, memory_store, xray
from ml.datasets.targets import NIH_FINDINGS


def upload(client, data: bytes | None = None, **params):
    return client.post("/analyze", files={"file": ("cxr.png", data or xray(), "image/png")}, params=params)


def test_analyze_saves_result_and_audit_events():
    store = memory_store()
    with make_client(store) as c:
        body = upload(c).json()
        assert body["persistence"] == "saved" and "persistence" not in body["errors"]
        saved = c.get(f"/analyses/{body['id']}").json()
    assert saved["findings"] == body["findings"] and saved["stages"] == body["stages"]
    assert saved["models"] == body["models"] and saved["persistence"] == "saved"
    assert datetime.fromisoformat(saved["created_at"]) == datetime.fromisoformat(body["created_at"])
    assert datetime.fromisoformat(saved["created_at"]).utcoffset().total_seconds() == 0
    with store.db.session() as s:
        events = s.scalars(select(AuditEventRecord).where(AuditEventRecord.analysis_id == body["id"])).all()
    assert {e.step for e in events} == set(body["stages"]) and all(e.status for e in events)
    assert next(e for e in events if e.step == "input").latency_ms is not None


def test_history_is_newest_first_and_paged():
    service = make_service(ConstModel([0.8] + [0.1] * (len(NIH_FINDINGS) - 1)), ConstModel([0.05]))
    with make_client(memory_store(), service=service) as c:
        ids = [upload(c, explain="false").json()["id"] for _ in range(3)]
        page = c.get("/analyses", params={"limit": 2}).json()
        rest = c.get("/analyses", params={"limit": 2, "offset": 2}).json()
    assert page["total"] == 3 and [i["id"] for i in page["items"]] == ids[::-1][:2]
    assert [i["id"] for i in rest["items"]] == [ids[0]]
    flagged = page["items"][0]["flagged"]
    assert flagged == [{"finding": "Atelectasis", "display_name": "Atelectasis", "band": "likely"}]


def test_users_cannot_see_or_delete_each_other():
    store = memory_store()
    with make_client(store, user="alice") as alice:
        analysis_id = upload(alice).json()["id"]
    with make_client(store, user="bob") as bob:
        assert bob.get(f"/analyses/{analysis_id}").status_code == 404
        assert bob.delete(f"/analyses/{analysis_id}").status_code == 404
        assert bob.get("/analyses").json()["total"] == 0
    with make_client(store, user="alice") as alice:
        assert alice.get(f"/analyses/{analysis_id}").status_code == 200


def test_delete_removes_analysis_and_audit_events():
    store = memory_store()
    with make_client(store) as c:
        analysis_id = upload(c).json()["id"]
        assert c.delete(f"/analyses/{analysis_id}").status_code == 204
        r = c.get(f"/analyses/{analysis_id}")
        assert r.status_code == 404 and r.json()["error_code"] == "ANALYSIS_NOT_FOUND"
        assert c.delete(f"/analyses/{analysis_id}").status_code == 404
    with store.db.session() as s:
        assert s.scalars(select(AuditEventRecord)).all() == []


def test_malformed_id_is_not_found():
    with make_client(memory_store()) as c:
        assert c.get("/analyses/not-a-uuid").status_code == 404


def test_database_failure_still_returns_results():
    broken = AnalysisStore(Database("sqlite:////nonexistent-folder/pneumyst.db"))
    with make_client(broken) as c:
        r = upload(c)
        assert r.status_code == 200 and len(r.json()["findings"]) == 15
        assert r.json()["persistence"] == "not_saved" and r.json()["errors"]["persistence"] == "STORAGE_FAILED"
        assert c.get("/analyses").json()["error_code"] == "DATABASE_UNAVAILABLE"
        assert c.get("/health").json()["database"] == "unavailable"


def test_without_database_results_are_not_saved():
    with make_client(None) as c:
        body = upload(c).json()
        assert body["persistence"] == "not_saved" and body["errors"]["persistence"] == "DATABASE_NOT_CONFIGURED"
        r = c.get("/analyses")
        assert r.status_code == 503 and r.json()["error_code"] == "DATABASE_UNAVAILABLE"


def test_health_with_database():
    with make_client(memory_store()) as c:
        assert c.get("/health").json()["status"] == "ok"


def test_migration_matches_models(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'migrated.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(Config(str(REPO / "backend" / "alembic.ini")), "head")
    engine = create_engine(url)
    with engine.connect() as conn:
        assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []
    engine.dispose()