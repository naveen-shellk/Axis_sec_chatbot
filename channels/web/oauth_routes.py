"""
chatbot_web/src/api/oauth_routes.py
-------------------------------------
OAuth2 client_credentials token endpoint.

Matches the Postman collection spec:
  POST /oauth/token
  Authorization: Basic base64(clientId:clientSecret)
  Body: grant_type=client_credentials

Returns a short-lived Bearer JWT that is then used on POST /api/chat.

For local dev / testing:
  clientId     = WEB_CLIENT_ID     (default: wxc_asl_chatbot)
  clientSecret = WEB_CLIENT_SECRET (default: dev-secret-change-in-prod)
  Token issued = WEB_API_TOKEN     (same static token used by /api/chat)
  expires_in   = 3600s (1 hour) — token never actually expires in dev,
                 we just return the same static token every time.

This allows the Postman collection to work as documented:
  1. POST /oauth/token → stores accessToken
  2. POST /api/chat with Bearer accessToken → works

Production: replace with a real OAuth2 server (e.g. AWS Cognito or a proper JWT issuer).
"""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["OAuth2"])

_CLIENT_ID     = os.getenv("WEB_CLIENT_ID",     "wxc_asl_chatbot")
_CLIENT_SECRET = os.getenv("WEB_CLIENT_SECRET",  "dev-secret-change-in-prod")
_ACCESS_TOKEN  = os.getenv("WEB_API_TOKEN",       "dev-token-change-in-prod")
_EXPIRES_IN    = int(os.getenv("TOKEN_EXPIRES_IN", "3600"))


def _extract_basic_credentials(authorization: str | None) -> tuple[str, str] | None:
    """Parse 'Basic base64(id:secret)' → (client_id, client_secret)."""
    if not authorization or not authorization.lower().startswith("basic "):
        return None
    try:
        decoded = base64.b64decode(authorization[6:]).decode("utf-8")
        client_id, _, client_secret = decoded.partition(":")
        return client_id, client_secret
    except Exception:
        return None


@router.post(
    "/oauth/token",
    summary="Get Access Token (OAuth2 client_credentials)",
    description="""
Exchange **client_id / client_secret** for a short-lived Bearer token.

**Method:** Basic Auth (clientId:clientSecret) in Authorization header, or body params.

**Body:** `grant_type=client_credentials` (application/x-www-form-urlencoded)

**Returns:** `{ access_token, token_type, expires_in }`

Use the returned `access_token` as `Authorization: Bearer <token>` on `POST /api/chat`.

**Dev credentials:**
- clientId: `wxc_asl_chatbot`
- clientSecret: value of `WEB_CLIENT_SECRET` env var (default: `dev-secret-change-in-prod`)
""",
    tags=["OAuth2"],
    responses={
        200: {
            "description": "Token issued",
            "content": {
                "application/json": {
                    "example": {
                        "access_token": "dev-token-change-in-prod",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                    }
                }
            },
        },
        401: {
            "description": "Invalid client credentials",
            "content": {
                "application/json": {
                    "example": {
                        "error": "invalid_client",
                        "error_description": "Client authentication failed",
                    }
                }
            },
        },
    },
)
async def token_endpoint(request: Request) -> JSONResponse:
    """Issue a Bearer token for valid client credentials."""

    # ── Extract credentials from Basic Auth header or body ────────────────────
    authorization = request.headers.get("Authorization")
    credentials   = _extract_basic_credentials(authorization)

    if credentials is None:
        # Fall back to body params
        try:
            form = await request.form()
            client_id_body     = form.get("client_id", "")
            client_secret_body = form.get("client_secret", "")
            if client_id_body and client_secret_body:
                credentials = (client_id_body, client_secret_body)
        except Exception:
            pass

    if credentials is None:
        logger.warning("[OAUTH] Token request with no credentials")
        return JSONResponse(
            status_code=401,
            content={
                "error": "invalid_client",
                "error_description": "Client authentication failed — provide Basic Auth or client_id/client_secret in body",
            },
        )

    client_id, client_secret = credentials

    # ── Validate credentials ──────────────────────────────────────────────────
    if client_id != _CLIENT_ID or client_secret != _CLIENT_SECRET:
        logger.warning("[OAUTH] Invalid credentials client_id=%r", client_id)
        return JSONResponse(
            status_code=401,
            content={
                "error": "invalid_client",
                "error_description": "Client authentication failed",
            },
        )

    # ── Validate grant_type ───────────────────────────────────────────────────
    try:
        form = await request.form()
        grant_type = form.get("grant_type", "")
    except Exception:
        grant_type = ""

    if grant_type != "client_credentials":
        return JSONResponse(
            status_code=400,
            content={
                "error": "unsupported_grant_type",
                "error_description": f"grant_type must be 'client_credentials', got '{grant_type}'",
            },
        )

    logger.info("[OAUTH] Token issued for client_id=%r", client_id)
    return JSONResponse(
        content={
            "access_token": _ACCESS_TOKEN,
            "token_type":   "Bearer",
            "expires_in":   _EXPIRES_IN,
        }
    )
