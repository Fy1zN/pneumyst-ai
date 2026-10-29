"""Clerk token verification with a locally generated key pair standing in for Clerk."""
from __future__ import annotations

import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.auth.clerk import ClerkVerifier
from backend.app.core.config import Settings
from backend.app.db.models import UserRecord
from backend.app.main import create_app
from backend.tests.conftest import default_models, make_service, memory_store, xray

ISSUER = "https://example-app.clerk.accounts.dev"
FRONTEND = "http://localhost:3000"
KID = "test-key"
SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class StaticJWKS:
    """Serves one public key by key id, like Clerk's JWKS endpoint."""

    def __init__(self, error: Exception | None = None):
        self.error = error

    def get_signing_key_from_jwt(self, token: str):
        if self.error is not None:
            raise self.error
        if jwt.get_unverified_header(token).get("kid") != KID:
            raise jwt.PyJWKClientError("no key with this kid")
        return SimpleNamespace(key=SIGNING_KEY.public_key())


def token(key=SIGNING_KEY, kid: str = KID, algorithm: str = "RS256", **overrides) -> str:
    now = int(time.time())
    claims = {"iss": ISSUER, "sub": "user_alice", "azp": FRONTEND, "iat": now, "nbf": now - 1, "exp": now + 60}
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm=algorithm, headers={"kid": kid})


def clerk_client(store=None, jwks: StaticJWKS | None = None) -> TestClient:
    settings = Settings(auth_mode="clerk", clerk_issuer=ISSUER, clerk_jwks_url="https://unused.invalid",
                        clerk_authorized_parties=(FRONTEND,))
    verifier = ClerkVerifier(ISSUER, "https://unused.invalid", (FRONTEND,), jwks_client=jwks or StaticJWKS())
    return TestClient(create_app(settings, make_service(*default_models()), store, verifier),
                      raise_server_exceptions=False)


def auth(t: str) -> dict:
    return {"Authorization": f"Bearer {t}"}


def upload(client, headers: dict):
    return client.post("/analyze", files={"file": ("cxr.png", xray(), "image/png")}, headers=headers)


def test_health_is_public_and_everything_else_needs_a_token():
    with clerk_client(memory_store()) as c:
        assert c.get("/health").status_code == 200
        for r in (c.get("/models"), c.get("/analyses"), upload(c, {})):
            assert r.status_code == 401 and r.json()["error_code"] == "UNAUTHENTICATED"


def test_valid_token_identifies_the_user():
    store = memory_store()
    with clerk_client(store) as c:
        r = upload(c, auth(token()))
        assert r.status_code == 200 and r.json()["persistence"] == "saved"
        assert c.get("/analyses", headers=auth(token())).json()["total"] == 1
        assert c.get("/models", headers=auth(token())).status_code == 200
    with store.db.session() as s:
        assert s.scalars(select(UserRecord.id)).all() == ["user_alice"]


@pytest.mark.parametrize("bad", [
    token(exp=int(time.time()) - 120),
    token(nbf=int(time.time()) + 120),
    token(iss="https://someone-else.clerk.accounts.dev"),
    token(azp="https://evil.example"),
    token(azp=None),
    token(sub=None),
    token(key=OTHER_KEY),
    token(kid="unknown-key"),
    token()[:-4] + "AAAA",
    jwt.encode({"iss": ISSUER, "sub": "user_alice", "azp": FRONTEND, "exp": int(time.time()) + 60},
               None, algorithm="none"),
    jwt.encode({"iss": ISSUER, "sub": "user_alice", "azp": FRONTEND, "exp": int(time.time()) + 60},
               "a-shared-secret-that-is-long-enough-for-hmac-sha256", algorithm="HS256", headers={"kid": KID}),
    "not-a-jwt",
], ids=["expired", "not-yet-valid", "wrong-issuer", "wrong-origin", "no-origin", "no-subject", "foreign-key",
        "unknown-kid", "tampered", "alg-none", "alg-hs256", "garbage"])
def test_bad_tokens_are_rejected(bad):
    with clerk_client(memory_store()) as c:
        r = c.get("/analyses", headers=auth(bad))
    assert r.status_code == 401 and r.json() == {"error_code": "UNAUTHENTICATED",
                                                 "detail": "invalid or expired session", "analysis_id": None}


def test_users_are_isolated_by_token():
    store = memory_store()
    with clerk_client(store) as c:
        analysis_id = upload(c, auth(token(sub="user_alice"))).json()["id"]
        assert c.get(f"/analyses/{analysis_id}", headers=auth(token(sub="user_bob"))).status_code == 404
        assert c.get(f"/analyses/{analysis_id}", headers=auth(token(sub="user_alice"))).status_code == 200


def test_unreachable_clerk_gives_503():
    with clerk_client(memory_store(), StaticJWKS(jwt.PyJWKClientConnectionError("timed out"))) as c:
        r = c.get("/analyses", headers=auth(token()))
    assert r.status_code == 503 and r.json()["error_code"] == "AUTH_UNAVAILABLE"


def test_clerk_mode_without_configuration_refuses_to_start():
    app = create_app(Settings(auth_mode="clerk"), make_service(*default_models()))
    with pytest.raises(RuntimeError, match="CLERK_ISSUER"):
        with TestClient(app):
            pass
    app = create_app(Settings(auth_mode="open"), make_service(*default_models()))
    with pytest.raises(RuntimeError, match="AUTH_MODE"):
        with TestClient(app):
            pass


def test_swagger_offers_bearer_login():
    with clerk_client() as c:
        schemes = c.get("/openapi.json").json()["components"]["securitySchemes"]
    assert schemes["HTTPBearer"]["scheme"] == "bearer"


def test_verifier_accepts_a_real_clerk_shaped_token():
    claims = ClerkVerifier(ISSUER, "", (FRONTEND,), jwks_client=StaticJWKS()).verify(token(sid="sess_1"))
    assert claims["sub"] == "user_alice" and claims["sid"] == "sess_1"