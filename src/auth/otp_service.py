"""
chatbot_web/src/auth/otp_service.py
------------------------------------
Mock phone-based OTP authentication service for the demo.

Production intent (see infra DFD): a customer authenticates with their
REGISTERED MOBILE NUMBER. The backend sends an OTP to that number and, once
verified, looks up the customer's Sub-Account ID from the mobile number. The
customer never types an account ID — the server resolves it.

For the demo we mock that service:

  - send_otp(phone)             → pretends to dispatch an OTP to the number.
  - verify_otp(phone, otp)      → validates the OTP AND resolves the Sub-Account
                                  ID from the number.
  - lookup_sub_account_by_phone(phone) → the phone → sub_account_id resolver
                                  (in production: a CRM / customer-DB lookup).

Rules (demo):
  * A single FIXED OTP is accepted (env MOCK_OTP, default "123456").
  * The OTP + lookup only succeed for a REGISTERED demo phone (the 1:1 map
    below, overridable via env OTP_PHONE_MAP). Any other number is rejected
    with "OTP invalid" — mimicking the backend having no customer record.
  * One phone maps to exactly one Sub-Account ID (1:1).
  * No retry cap / no resend (accept-or-reject only), per demo scope.

Swap-out for production:
  Replace _phone_map() with a real "get sub-account by mobile" API call, the
  fixed OTP with a generated code + store, and send_otp with the real SMS/email
  dispatch. The public signatures below stay the same so callers
  (auth_routes.py, entry/handler.py) don't change.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ── Fixed demo OTP ─────────────────────────────────────────────────────────────
_DEFAULT_OTP = "123456"

# ── Demo phone → Sub-Account ID map (1:1) ───────────────────────────────────────
# Auto-generated demo mobile numbers for the 7 demo accounts. Overridable via env
# OTP_PHONE_MAP as "phone:subacct,phone:subacct,...".
_DEFAULT_PHONE_MAP: dict[str, str] = {
    "9000000001": "7542154",
    "9000000002": "6033593",
    "9000000003": "7201565",
    "9000000004": "5905533",
    "9000000005": "155036",
    "9000000006": "8505999",
    "9000000007": "8755851",
}


def _fixed_otp() -> str:
    return os.getenv("MOCK_OTP", _DEFAULT_OTP).strip()


def _normalize_phone(phone: str | None) -> str:
    """
    Reduce a phone number to its comparable form: digits only, and if it has a
    country code / leading zero we keep the last 10 digits (Indian mobile).
    Handles inputs like "+91 90000 00001", "09000000001", "9000000001".
    """
    if not phone:
        return ""
    digits = "".join(ch for ch in str(phone) if ch.isdigit())
    if len(digits) > 10:
        digits = digits[-10:]
    return digits


def _phone_map() -> dict[str, str]:
    raw = os.getenv("OTP_PHONE_MAP", "").strip()
    if raw:
        out: dict[str, str] = {}
        for pair in raw.split(","):
            if ":" in pair:
                p, s = pair.split(":", 1)
                out[_normalize_phone(p)] = s.strip()
        return out
    return dict(_DEFAULT_PHONE_MAP)


def lookup_sub_account_by_phone(phone: str | None) -> str | None:
    """
    Resolve a registered mobile number to its Sub-Account ID.

    Production: replace with a real customer-search-by-mobile API call.
    Returns the sub_account_id string, or None if the number isn't registered.
    """
    return _phone_map().get(_normalize_phone(phone))


def is_registered_phone(phone: str | None) -> bool:
    """True if the mobile number maps to a demo sub-account."""
    return lookup_sub_account_by_phone(phone) is not None


def send_otp(phone: str) -> dict:
    """
    Mock-dispatch an OTP to the given mobile number.

    Always reports success (like a real service that doesn't leak whether a
    number is registered). The actual allow/deny happens at verify time.
    """
    normalized = _normalize_phone(phone)
    logger.info("[OTP] send_otp phone=***%s (mock — fixed OTP)", normalized[-4:] if normalized else "")
    return {
        "status": "sent",
        "phone": normalized,
        # A real service would NEVER return the OTP. We expose a hint only in
        # non-production so the demo UI / tester knows what to enter.
        "hint": None if os.getenv("ENVIRONMENT") == "production" else _fixed_otp(),
        "message": "An OTP has been sent to your registered mobile number.",
    }


def verify_otp(phone: str, otp: str) -> dict:
    """
    Verify an entered OTP for a mobile number and resolve the Sub-Account ID.

    Passes ONLY when both:
      1. the number is a registered demo phone (maps to a sub-account), AND
      2. the OTP matches the fixed demo OTP.

    On success returns the resolved sub_account_id. Any other combination →
    verified=False with reason "OTP invalid".

    Returns {"verified": bool, "sub_account_id": str|None, "reason": str|None}.
    """
    normalized = _normalize_phone(phone)
    code = str(otp).strip()

    sub_account_id = lookup_sub_account_by_phone(normalized)
    if sub_account_id is None:
        logger.info("[OTP] verify FAIL phone=***%s not registered", normalized[-4:] if normalized else "")
        return {"verified": False, "sub_account_id": None, "reason": "OTP invalid"}

    if code != _fixed_otp():
        logger.info("[OTP] verify FAIL phone=***%s wrong code", normalized[-4:])
        return {"verified": False, "sub_account_id": None, "reason": "OTP invalid"}

    logger.info("[OTP] verify OK phone=***%s → sub_account_id=%s", normalized[-4:], sub_account_id)
    return {"verified": True, "sub_account_id": sub_account_id, "reason": None}
