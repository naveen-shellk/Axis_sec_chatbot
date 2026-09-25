"""
chatbot_web/src/flows/brokerage/handler.py
-------------------------------------------
Brokerage & Charges flow — post-login.

Optimised pattern:
  - LLM handles customer INPUT (extracts charges type, date from free text)
  - Response messages are HARDCODED
  - LLM only called when exact/partial match fails

State machine:
  start → account check → charges_type_selection
    → Trading Charges → fetch ledger → hardcoded charges display
    → DP Charges      → dp_date_selection → send DP bill → hardcoded confirm
  session_end_response → shared
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from src.core.llm import call_intent_llm
from src.core.session_store import save_session
from src.shared.session_end import handle_session_end
from models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

_DP_CHARGES_LINK = "https://www.axisdirect.in/charges"
CHARGES_TYPES    = ["Trading Charges", "DP Charges"]


def _last_30_days() -> list[str]:
    today = date.today()
    return [(today - timedelta(days=i)).strftime("%d-%m-%Y") for i in range(1, 31)]


# ── Hardcoded messages ────────────────────────────────────────────────────────

def _msg_charges_type() -> str:
    return "Which type of charges information do you need?"

def _msg_trading_charges(closing: str, opening: str, masked_email: str) -> str:
    balance = float(closing) if closing else 0.0
    if balance > 0:
        return (
            f"Your current ledger balance:\n\n"
            f"• Opening Balance: ₹{opening}\n"
            f"• Closing Balance: ₹{closing}\n\n"
            f"You have outstanding charges. A detailed statement has been "
            f"sent to {masked_email}."
        )
    return (
        f"Your current ledger balance:\n\n"
        f"• Opening Balance: ₹{opening}\n"
        f"• Closing Balance: ₹{closing}\n\n"
        f"No outstanding charges at this time."
    )

def _msg_dp_date() -> str:
    return "Please select the date for your DP charges statement:"

def _msg_dp_sent(masked_email: str) -> str:
    return (
        f"Your DP charges statement has been sent to {masked_email}.\n\n"
        f"You can also view the complete DP charges schedule here:\n"
        f"🔗 {_DP_CHARGES_LINK}"
    )

def _msg_deactivated() -> str:
    return (
        "Your account is currently deactivated. Please contact our support team "
        "to reactivate.\n\n📞 Customer Care: 022-40508080 / 022-61480808"
    )

def _msg_reprompt() -> str:
    return "I didn't catch that. Please select from the options:"


# ── LLM extraction (input only) ───────────────────────────────────────────────

_EXTRACT_CHARGES_SYS = """\
The customer is choosing between "Trading Charges" and "DP Charges" on Axis Direct.
Extract their choice from the message.
Return JSON only:
{"selected": "<Trading Charges|DP Charges|null>", "reasoning": "<one line>"}
"""

_EXTRACT_DATE_SYS = """\
The customer is selecting a date (DD-MM-YYYY format) for a DP charges statement on Axis Direct.
Available dates are in context. Extract the date they meant.
Return JSON only:
{"selected": "<DD-MM-YYYY or null>", "reasoning": "<one line>"}
"""


def _extract(system: str, message: str, context: dict | None = None) -> dict:
    ctx  = "\n".join(f"{k}: {v}" for k, v in (context or {}).items())
    full = f"{ctx}\n\nCustomer: {message}" if ctx else f"Customer: {message}"
    result = call_intent_llm(system, [{"role": "user", "content": [{"text": full}]}])
    return result.get("parsed") or {}


def _match(text: str, options: list[str]) -> str | None:
    tl = text.strip().lower()
    for opt in options:
        if opt.lower() == tl:
            return opt
    for opt in options:
        if opt.lower() in tl or tl in opt.lower():
            return opt
    return None


def _hist(state: SessionState, msg: str, reply: str) -> list:
    return state.history + [
        {"role": "user",      "content": msg},
        {"role": "assistant", "content": reply},
    ]


# ── Main handler ──────────────────────────────────────────────────────────────

def handle_brokerage(
    state: SessionState,
    customer_message: str,
) -> tuple[InternalMessageResponse, SessionState]:

    fs = state.flow_state

    # ── START: account check ──────────────────────────────────────────────────
    if fs == "start":
        from src.core.strands_agent import get_profile
        try:
            profile = get_profile(state.sub_account_id or "")
            status  = profile.account_status
        except Exception:
            status = "active"

        if status in ("deactivated", "purged"):
            reply = _msg_deactivated()
            ns = state.model_copy(update={
                "flow": "brokerage", "flow_state": "session_end_response",
                "history": _hist(state, customer_message, reply),
            })
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply,
                    quick_reply_options=["Go back to main menu", "End Chat"],
                    flow_state="session_end_response", status="end",
                ), ns,
            )

        reply = _msg_charges_type()
        ns = state.model_copy(update={
            "flow": "brokerage", "flow_state": "charges_type_selection",
            "history": _hist(state, customer_message, reply),
        })
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=reply,
                quick_reply_options=CHARGES_TYPES,
                flow_state="charges_type_selection", status="ok",
            ), ns,
        )

    # ── CHARGES TYPE SELECTION ────────────────────────────────────────────────
    if fs == "charges_type_selection":
        # 1. Exact/partial match
        charges_type = _match(customer_message, CHARGES_TYPES)

        # 2. LLM extraction for free text
        if not charges_type:
            parsed       = _extract(_EXTRACT_CHARGES_SYS, customer_message)
            charges_type = _match(parsed.get("selected", ""), CHARGES_TYPES)

        if not charges_type:
            reply = _msg_reprompt()
            ns = state.model_copy(update={"history": _hist(state, customer_message, reply)})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply, quick_reply_options=CHARGES_TYPES,
                    flow_state="charges_type_selection", status="reprompt",
                ), ns,
            )

        # ── Trading Charges: fetch ledger → hardcoded display ─────────────────
        if charges_type == "Trading Charges":
            from src.gateways.statement_api import get_ledger
            from src.gateways.customer_api import get_customer_profile, mask_email

            today_str = date.today().strftime("%d-%m-%Y")
            ledger    = None
            try:
                from src.core.strands_agent import run_tool
                ledger = run_tool("get_ledger_balance",
                                  sub_account_id=state.sub_account_id or "",
                                  start_date=today_str, end_date=today_str)
            except Exception as exc:
                logger.warning("[BROKERAGE] agent tool path failed: %s — direct fallback", exc)
            if ledger is None:
                ledger = get_ledger(state.sub_account_id or "", today_str, today_str)

            if not ledger.get("success", False):
                _err = (
                    "We were unable to retrieve your charges information at this time.\n\n"
                    "Please try again later or contact support: 📞 022-40508080 / 022-61480808"
                )
                ns = state.model_copy(update={
                    "flow_state": "session_end_response",
                    "history": _hist(state, customer_message, _err),
                })
                save_session(state.conversation_id, ns)
                return (
                    InternalMessageResponse(
                        reply_message=_err,
                        quick_reply_options=["Go back to main menu", "End Chat"],
                        flow_state="session_end_response", status="error",
                    ), ns,
                )

            try:
                from src.core.strands_agent import get_profile
                profile = get_profile(state.sub_account_id or "")
                masked  = mask_email(profile.registered_email) or "your registered email"
            except Exception:
                masked = "your registered email"

            reply = _msg_trading_charges(
                closing=str(ledger.get("closing_balance", "0")),
                opening=str(ledger.get("opening_balance", "0")),
                masked_email=masked,
            )
            ns = state.model_copy(update={
                "flow_state": "session_end_response",
                "history": _hist(state, customer_message, reply),
            })
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply,
                    quick_reply_options=["Go back to main menu", "End Chat"],
                    flow_state="session_end_response", status="ok",
                ), ns,
            )

        # ── DP Charges: show date picker ──────────────────────────────────────
        dates = _last_30_days()[:10]
        reply = _msg_dp_date()
        ns = state.model_copy(update={
            "flow_state": "dp_date_selection",
            "history": _hist(state, customer_message, reply),
        })
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=reply, quick_reply_options=dates,
                flow_state="dp_date_selection", status="ok",
            ), ns,
        )

    # ── DP DATE SELECTION ─────────────────────────────────────────────────────
    if fs == "dp_date_selection":
        dates = _last_30_days()[:10]

        # 1. Exact match
        selected_date = _match(customer_message, dates)

        # 2. LLM extraction
        if not selected_date:
            parsed        = _extract(_EXTRACT_DATE_SYS, customer_message, {"dates": ", ".join(dates)})
            selected_date = _match(parsed.get("selected", ""), dates) or parsed.get("selected")

        if not selected_date:
            reply = _msg_reprompt()
            ns = state.model_copy(update={"history": _hist(state, customer_message, reply)})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply, quick_reply_options=dates,
                    flow_state="dp_date_selection", status="reprompt",
                ), ns,
            )

        from src.gateways.statement_api import send_dp_bill
        from src.gateways.customer_api import get_customer_profile, mask_email

        result = None
        try:
            from src.core.strands_agent import run_tool
            result = run_tool("send_dp_bill",
                              sub_account_id=state.sub_account_id or "",
                              start_date=selected_date, end_date=selected_date)
        except Exception as exc:
            logger.warning("[BROKERAGE] agent tool path failed: %s — direct fallback", exc)
        if result is None:
            result = send_dp_bill(state.sub_account_id or "", selected_date, selected_date)

        if not result.get("success", False):
            _err = (
                "We were unable to send your DP charges statement at this time.\n\n"
                "Please try again later or contact support: 📞 022-40508080 / 022-61480808"
            )
            ns = state.model_copy(update={
                "flow_state": "session_end_response",
                "history": _hist(state, customer_message, _err),
            })
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=_err,
                    quick_reply_options=["Go back to main menu", "End Chat"],
                    flow_state="session_end_response", status="error",
                ), ns,
            )

        try:
            from src.core.strands_agent import get_profile
            profile = get_profile(state.sub_account_id or "")
            masked  = mask_email(profile.registered_email) or "your registered email"
        except Exception:
            masked = result.get("masked_email", "your registered email")

        reply = _msg_dp_sent(masked)
        ns = state.model_copy(update={
            "flow_state": "session_end_response",
            "history": _hist(state, customer_message, reply),
        })
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=reply,
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state="session_end_response", status="ok",
            ), ns,
        )

    if fs == "session_end_response":
        return handle_session_end(state, customer_message)

    logger.warning("[BROKERAGE] unknown flow_state %r — reset", fs)
    return handle_brokerage(state.model_copy(update={"flow_state": "start"}), customer_message)
