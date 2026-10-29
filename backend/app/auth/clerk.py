"""Verification of Clerk session tokens: RS256 JWTs signed with the Clerk instance's private key."""
from __future__ import annotations

import jwt
from jwt import PyJWKClient, PyJWKClientConnectionError


class TokenError(Exception):
    """The token is missing a requirement; the reason is logged, never returned to the client."""


class AuthUnavailable(Exception):
    """Clerk's public keys could not be fetched."""


class ClerkVerifier:
    def __init__(self, issuer: str, jwks_url: str, authorized_parties: tuple[str, ...], leeway_s: int = 5,
                 jwks_client=None):
        self.issuer = issuer.rstrip("/")
        self.authorized_parties = set(authorized_parties)
        self.leeway_s = leeway_s
        # Keys are cached, so Clerk is contacted only on start-up and when it rotates its keys.
        self.jwks = jwks_client or PyJWKClient(jwks_url, cache_keys=True, lifespan=3600, timeout=5)

    def verify(self, token: str) -> dict:
        try:
            key = self.jwks.get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=["RS256"], issuer=self.issuer, leeway=self.leeway_s,
                                options={"require": ["exp", "iat", "iss", "sub"]})
        except PyJWKClientConnectionError as exc:
            raise AuthUnavailable(str(exc)) from exc
        except jwt.PyJWTError as exc:
            raise TokenError(str(exc)) from exc
        # azp is the origin the token was issued to; it stops tokens from other sites being replayed here.
        if self.authorized_parties and claims.get("azp") not in self.authorized_parties:
            raise TokenError(f"unexpected authorized party: {claims.get('azp')!r}")
        return claims