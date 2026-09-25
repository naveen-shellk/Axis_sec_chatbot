"""
chatbot_web/src/flows/account_details/handler.py
--------------------------------------------------
Account Details flow — post-login, one-shot: fetch profile and display.
"""

from __future__ import annotations

import logging

from src.core.conversation import run_conversation_turn
from src.core.session_store import save_session
from src.shared.session_end import handle_session_end
from models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

_SYS = """\
You are displaying account details for an Axis Direct customer.
You MUST respond with valid JSON ONLY. No markdown fences, no conversational text before or after the JSON.

Return exactly this JSON structure:
{
  "message": "Please find the requested details below:\n\nAccount Details:\nTrading ID: {trading_id}\nDP ID: {dp_id}\nDemat Account Number: {demat_account}\nAccount Status: {account_status}\nLinked Bank Account: {masked_bank}\nNominee: {nominee}\n\nContact Details:\nName: {customer_name}\nIncome Range: {income_range}\nRegistered Email ID: {masked_email}\nRegistered Mobile No: {masked_mobile}\nAddress (Correspondence): {address}\n\nFor any queries or to place an order:\nRI  - 022-40508080 / 022-61480808\nNRI - 022-61480809",
  "quick_replies": ["Go back to main menu", "End Chat"],
  "flow_action": "session_end",
  "reasoning": "displayed account details"
}

Rules:
1. Substitute all {placeholders} with the actual values provided in the context.
2. If any field is unavailable or null, use "N/A".
3. Return ONLY the valid JSON object.
"""


def handle_account_details(state: SessionState, customer_message: str) -> tuple[InternalMessageResponse, SessionState]:

    fs = state.flow_state

    if fs in ("start", "account_details"):
        from src.gateways.customer_api import (
            get_customer_profile, mask_email, mask_mobile, mask_account,
        )

        _ERROR_MSG = (
            "We were unable to retrieve your account details at this time.\n\n"
            "Please try again later or contact support:\n"
            "📞 022-40508080 / 022-61480808"
        )

        profile_data: dict = {}
        api_failed = False

        try:
            profile = get_customer_profile(state.sub_account_id or "")
            raw     = profile.raw.get("data", profile.raw)

            # If profile is empty dict (API returned nothing), treat as failure
            if not raw:
                raise ValueError("Empty profile response from API")

            trading_id  = raw.get("parentTradingId") or raw.get("subAccountId") or state.sub_account_id or "N/A"
            dp_accounts = raw.get("dpAccountDetails", [])
            default_dp  = next((d for d in dp_accounts if d.get("dpDefault")), dp_accounts[0] if dp_accounts else {})
            dp_id       = default_dp.get("dpId", "N/A")
            demat_acct  = default_dp.get("dpAccountNo", "N/A")

            status_map  = {"e": "Active", "d": "Deactivated", "p": "Purged"}
            acct_status = status_map.get(str(raw.get("accountStatus","e")).strip().lower(), "Active")

            bank_accounts = raw.get("bankAccountDetails", [])
            default_bank  = next((b for b in bank_accounts if b.get("isDefault")), bank_accounts[0] if bank_accounts else {})
            masked_bank   = mask_account(str(default_bank.get("bankAccountNo", ""))) or "N/A"

            nominees    = raw.get("nomineeDetails", [{}])
            nominee_name = nominees[0].get("nomineeName", "N/A") if nominees else "N/A"

            customer_name = profile.name or "N/A"
            income_range  = raw.get("incomeRange") or raw.get("annualIncome") or "N/A"
            address       = raw.get("address") or raw.get("correspondenceAddress") or "N/A"

            profile_data = {
                "trading_id":     trading_id,
                "dp_id":          dp_id,
                "demat_account":  demat_acct,
                "account_status": acct_status,
                "masked_bank":    masked_bank,
                "nominee":        nominee_name,
                "customer_name":  customer_name,
                "income_range":   income_range,
                "masked_email":   mask_email(profile.registered_email) or "N/A",
                "masked_mobile":  mask_mobile(profile.phone) or "N/A",
                "address":        address,
            }
        except Exception as exc:
            logger.error("[ACCOUNT_DETAILS] profile fetch failed: %s", exc)
            api_failed = True

        if api_failed:
            hist = state.history + [
                {"role": "user",      "content": customer_message},
                {"role": "assistant", "content": _ERROR_MSG},
            ]
            ns = state.model_copy(update={"flow_state": "session_end_response", "history": hist})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=_ERROR_MSG,
                    quick_reply_options=["Go back to main menu", "End Chat"],
                    flow_state="session_end_response",
                    status="error",
                ),
                ns,
            )

        # Format response directly from retrieved profile data (deterministic & instant)
        formatted_message = (
            "Please find the requested details below:\n\n"
            "Account Details:\n"
            f"Trading ID: {profile_data.get('trading_id', 'N/A')}\n"
            f"DP ID: {profile_data.get('dp_id', 'N/A')}\n"
            f"Demat Account Number: {profile_data.get('demat_account', 'N/A')}\n"
            f"Account Status: {profile_data.get('account_status', 'Active')}\n"
            f"Linked Bank Account: {profile_data.get('masked_bank', 'N/A')}\n"
            f"Nominee: {profile_data.get('nominee', 'N/A')}\n\n"
            "Contact Details:\n"
            f"Name: {profile_data.get('customer_name', 'N/A')}\n"
            f"Income Range: {profile_data.get('income_range', 'N/A')}\n"
            f"Registered Email ID: {profile_data.get('masked_email', 'N/A')}\n"
            f"Registered Mobile No: {profile_data.get('masked_mobile', 'N/A')}\n"
            f"Address (Correspondence): {profile_data.get('address', 'N/A')}\n\n"
            "For any queries or to place an order:\n"
            "RI  - 022-40508080 / 022-61480808\n"
            "NRI - 022-61480809"
        )

        hist = state.history + [
            {"role": "user",      "content": customer_message},
            {"role": "assistant", "content": formatted_message},
        ]
        new_state = state.model_copy(update={
            # Keep flow set so the terminal session_end_response state routes
            # back here (→ handle_session_end) on the next turn. Setting flow=None
            # orphans the end node and misroutes "Go back to main menu"/"End Chat".
            "flow": "account_details", "flow_state": "session_end_response", "history": hist,
        })
        save_session(state.conversation_id, new_state)
        return (
            InternalMessageResponse(
                reply_message=formatted_message,
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state="session_end_response", status="ok",
            ),
            new_state,
        )

    if fs == "session_end_response":
        return handle_session_end(state, customer_message)

    logger.warning("[ACCOUNT_DETAILS] unknown flow_state %r — reset", fs)
    return handle_account_details(state.model_copy(update={"flow_state": "start"}), customer_message)
