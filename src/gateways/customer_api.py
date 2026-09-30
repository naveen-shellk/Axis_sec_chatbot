"""
chatbot_web/src/gateways/customer_api.py
------------------------------------------
All calls route through AgentCoreGatewayClient — see gateway_client.py.
This module re-exports the same interface so existing flow handler imports
(from src.gateways.customer_api import get_customer_profile) keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.gateways.gateway_client import (
    get_customer_profile as _gw_get_profile,
    mask_email,
    mask_mobile,
    get_masked_email,
)


@dataclass
class CustomerProfile:
    sub_account_id:       str
    account_status:       str   # "active" | "deactivated" | "purged"
    name:                 str
    registered_email:     str
    phone:                str
    account_opening_date: str
    portal_status:        int   # 1 = FTL done, 0 = FTL pending
    deactivation_code:    str
    deactivation_reason:  str
    demat_account_no:     str = ""   # primary DP account number
    trading_account_no:   str = ""   # sub_account_id / trading ID
    raw:                  dict = field(default_factory=dict)


def _map_status(raw_status: str) -> str:
    s = str(raw_status).strip().lower()
    if s == "e":   return "active"
    if s == "d":   return "deactivated"
    if s == "p":   return "purged"
    return "active"


def get_customer_profile(sub_account_id: str) -> CustomerProfile:
    """
    Fetch full customer profile via AgentCore Gateway.
    Returns a typed CustomerProfile dataclass.
    """
    raw = _gw_get_profile(sub_account_id)

    account_status = _map_status(raw.get("accountStatus", "E"))
    device_list    = raw.get("Device") or []
    portal_status  = 1 if device_list else 0
    deact_code     = str(raw.get("entStatusLov") or raw.get("deactivationCode") or "").strip().upper()
    opening_date   = raw.get("accountOpeningDate") or raw.get("openingDate") or ""
    # The profile API returns the mobile under "phoneNo" (as an int); older
    # shapes used "mobileNo"/"phone". Accept any of them and stringify.
    phone          = str(raw.get("mobileNo") or raw.get("phoneNo") or raw.get("phone") or "").strip()

    # Extract primary demat account number from dpAccountDetails
    dp_accounts    = raw.get("dpAccountDetails") or []
    demat_no       = ""
    if dp_accounts:
        # prefer default DP, else first one
        default_dp = next((d for d in dp_accounts if d.get("dpDefault")), dp_accounts[0])
        demat_no   = default_dp.get("dpAccountNo") or default_dp.get("dpId") or ""

    # Trading account = sub_account_id (parentTradingId)
    trading_no = raw.get("parentTradingId") or raw.get("subAccountId") or sub_account_id

    return CustomerProfile(
        sub_account_id      = sub_account_id,
        account_status      = account_status,
        name                = raw.get("name") or raw.get("customerName") or "",
        registered_email    = raw.get("email") or "",
        phone               = phone,
        account_opening_date= opening_date,
        portal_status       = portal_status,
        deactivation_code   = deact_code,
        deactivation_reason = raw.get("deactivationReason") or "",
        demat_account_no    = demat_no,
        trading_account_no  = trading_no,
        raw                 = raw,
    )


def mask_account(account_no: str) -> str:
    """Mask a bank / demat / trading account number, showing only the last 4."""
    s = str(account_no or "").strip()
    if not s or len(s) < 4:
        return s
    return f"{'X' * (len(s) - 4)}{s[-4:]}"


def mask_pan(pan: str) -> str:
    """Mask a PAN — show first 2 and last 1 char (e.g. AB*****1F → AB******F)."""
    s = str(pan or "").strip().upper()
    if not s or len(s) < 4:
        return s
    return f"{s[:2]}{'*' * (len(s) - 3)}{s[-1]}"


__all__ = [
    "CustomerProfile",
    "get_customer_profile",
    "mask_email",
    "mask_mobile",
    "mask_account",
    "mask_pan",
    "get_masked_email",
]
