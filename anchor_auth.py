"""OAuth token verification for Anchor Memory's remote MCP transport."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any

import jwt
from jwt import PyJWKClient
from mcp.server.auth.provider import AccessToken, TokenVerifier


logger = logging.getLogger(__name__)

OAUTH_SCOPES = ("anchor:read", "anchor:write", "anchor:admin")


class Auth0JWTVerifier(TokenVerifier):
    """Verify Auth0 RS256 access tokens without exposing their contents."""

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_uri: str | None = None,
        algorithms: Iterable[str] = ("RS256",),
    ):
        if not issuer.startswith("https://"):
            raise ValueError("OAuth issuer must use HTTPS")
        if not audience.startswith("https://"):
            raise ValueError("OAuth audience must use HTTPS")
        self.issuer = issuer.rstrip("/") + "/"
        self.audience = audience
        self.algorithms = tuple(algorithms)
        self.jwks_uri = jwks_uri or f"{self.issuer}.well-known/jwks.json"
        self._jwk_client = PyJWKClient(self.jwks_uri)

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token or token.count(".") != 2:
            return None
        try:
            claims = await asyncio.to_thread(self._decode, token)
        except (jwt.PyJWTError, ValueError, TypeError) as exc:
            logger.warning("Rejected OAuth access token: %s", type(exc).__name__)
            return None
        except Exception as exc:  # Network/JWKS failures must fail closed.
            logger.warning("OAuth signing-key lookup failed: %s", type(exc).__name__)
            return None

        scopes = self._extract_scopes(claims)
        missing_scopes = [scope for scope in OAUTH_SCOPES if scope not in scopes]
        if missing_scopes:
            logger.warning(
                "OAuth access token is missing required Anchor scopes: %s",
                ", ".join(missing_scopes),
            )
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            logger.warning("Rejected OAuth access token without a subject")
            return None

        return AccessToken(
            token=token,
            client_id=str(claims.get("azp") or claims.get("client_id") or "unknown"),
            subject=subject,
            scopes=scopes,
            expires_at=claims.get("exp"),
            resource=self.audience,
            claims=claims,
        )

    def _decode(self, token: str) -> dict[str, Any]:
        signing_key = self._jwk_client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            key=signing_key.key,
            algorithms=list(self.algorithms),
            issuer=self.issuer,
            audience=self.audience,
            options={
                "require": ["exp", "iat", "sub"],
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_iss": True,
                "verify_aud": True,
            },
        )

    @staticmethod
    def _extract_scopes(claims: dict[str, Any]) -> list[str]:
        scopes: list[str] = []
        for claim_name in ("scope", "scp", "permissions"):
            value = claims.get(claim_name)
            if isinstance(value, str):
                candidates = value.split()
            elif isinstance(value, (list, tuple)):
                candidates = [str(item) for item in value]
            else:
                continue
            for candidate in candidates:
                if candidate and candidate not in scopes:
                    scopes.append(candidate)
        return scopes
