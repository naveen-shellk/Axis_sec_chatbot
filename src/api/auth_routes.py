"""
chatbot_web/src/api/auth_routes.py
-----------------------------------
Mock phone-based OTP authentication endpoints (production-shaped).

  POST /auth/send-otp      { "phone": "9000000002" }
  POST /auth/verify-otp    { "phone": "9000000002", "otp": "123456" }
                           → { "verified": true, "sub_account_id": "6033593" }

The customer authenticates with their REGISTERED MOBILE NUMBER. On successful
OTP verification the backend resolves the Sub-Account ID from the number — the
customer never types an account ID.

For the demo the logic is backed by src/auth/otp_service.py (fixed OTP + a 1:1
phone→sub_account map). In production, swap the service internals for a real
OTP store + customer-search-by-mobile lookup — the request/response contract
here stays the same.

NOTE: These endpoints are intentionally unauthenticated (they ARE the auth
step). The chat handler consults the same otp_service to gate account flows.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import os

from src.auth.otp_service import send_otp, verify_otp

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Authentication (mock OTP)"])

# ── App-access gate ────────────────────────────────────────────────────────────
# A single shared password that unlocks the chat widget. On success we hand back
# the chat Bearer token (WEB_API_TOKEN) so the widget can call /api/chat. The
# password itself never lives in the page source — it is verified here.
_APP_PASSWORD = os.getenv("CHAT_APP_PASSWORD", "axis@2026")
_CHAT_TOKEN   = os.getenv("WEB_API_TOKEN", "dev-token-change-in-prod")


class AppLoginRequest(BaseModel):
    password: str = Field(..., description="Shared app-access password for the chat widget")


class SendOtpRequest(BaseModel):
    phone: str = Field(..., description="Customer's registered mobile number")


class VerifyOtpRequest(BaseModel):
    phone: str = Field(..., description="Customer's registered mobile number")
    otp:   str = Field(..., description="One-time password entered by the customer")


@router.post(
    "/auth/app-login",
    summary="App-access login (unlock the chat widget)",
    description="Verify the shared app-access password and, on success, return "
                "the chat Bearer token used for /api/chat. The password is never "
                "exposed in the frontend — it is checked server-side here.",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {"token": "asl_test!@", "token_type": "Bearer"}
                }
            }
        },
        401: {
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid password"}
                }
            }
        },
    },
)
def app_login_endpoint(body: AppLoginRequest) -> JSONResponse:
    if body.password != _APP_PASSWORD:
        logger.warning("[APP-LOGIN] failed attempt")
        return JSONResponse(status_code=401, content={"detail": "Invalid password"})
    logger.info("[APP-LOGIN] success — chat token issued")
    return JSONResponse(content={"token": _CHAT_TOKEN, "token_type": "Bearer"})


@router.post(
    "/auth/send-otp",
    summary="Send OTP (mock)",
    description="Mock-dispatch a one-time password to the customer's registered "
                "mobile number. Always reports success; the actual allow/deny "
                "happens at verification.",
)
def send_otp_endpoint(body: SendOtpRequest) -> JSONResponse:
    result = send_otp(body.phone)
    return JSONResponse(content=result)


@router.post(
    "/auth/verify-otp",
    summary="Verify OTP (mock)",
    description="Verify the OTP for a mobile number and resolve the Sub-Account "
                "ID. Succeeds only for a registered demo number with the correct "
                "code; otherwise returns verified=false with reason 'OTP invalid'.",
    responses={
        200: {
            "content": {
                "application/json": {
                    "examples": {
                        "valid":   {"value": {"verified": True,  "sub_account_id": "6033593", "reason": None}},
                        "invalid": {"value": {"verified": False, "sub_account_id": None, "reason": "OTP invalid"}},
                    }
                }
            }
        }
    },
)
def verify_otp_endpoint(body: VerifyOtpRequest) -> JSONResponse:
    result = verify_otp(body.phone, body.otp)
    status = 200 if result["verified"] else 401
    return JSONResponse(status_code=status, content=result)
