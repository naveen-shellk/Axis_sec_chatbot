"""
chatbot_web/src/flows/closure/handler.py
------------------------------------------
Account Closure flow — triggered ONLY when customer explicitly asks about closure.
NOT in the main menu — intent-only entry via LLM classifier.

Revised flow (per spec + UX feedback):

  start
    ├── Fetch profile + call closure API
    │   API response reason → scenario detection
    │
    ├── Scenario: Already Closed
    │     Message: "Dear <name>, Your Demat account <demat> & Trading Account <trading>
    │               is already closed."
    │     → session end
    │
    ├── Scenario: Closure is in Process
    │     Message: "Dear <name>, The closure of your Demat account <demat> & Trading
    │               Account <trading> is in progress, you will receive a confirmation
    │               mail on your registered E-mail ID once your account is closed."
    │     → session end
    │
    └── Scenario: New Request — 5-step sequence
          Step 1 (new_request_msg1):
            Qwen LLM generates warm/empathetic Message 1:
            "Dear <name>, We are sorry to know that you wish to close your account…
             We would like to know if there's anything we can do to change your decision."
            Quick replies: ["I still want to close", "Go back to main menu"]
            → wait for confirmation

          Step 2 (confirm_proceed):
            Customer confirms they still want to close
            YES ("still want", "proceed", "close", "yes", "continue") →
              → new_request_msg2
            NO ("no", "back", "menu", "cancel", "changed my mind") →
              → thank you + go to menu

          Step 3 (new_request_msg2):
            Hardcoded: Portal login deeplink + REQUEST ACCOUNT CLOSURE tab + ticket link
            Quick replies: ["Continue", "Go back to main menu"]

          Step 4 (new_request_msg3):
            Hardcoded: NRI/offline form details (CDSL / NSDL download paths)
            Quick replies: ["Main Menu", "End Chat"]

          Step 5 (new_request_msg4):
            Hardcoded: Thank you + Main Menu

LLM usage:
  - Message 1 (new_request_msg1): Qwen — warm empathetic message, uses customer name
  - All other messages: hardcoded (spec content must not be altered)

Gateway used: create_closure_request()
  → closure-api___create_account_closure
  → POST http://10.9.161.132:6013/api/create_account_closure/v2

API response reason → scenario:
  "already closed" / "account already closed"    → already_closed
  "already in process" / "under process"         → in_progress
  "Closure Processed" / success                  → new_request
  "holdings present" / "outstanding"             → new_request
"""

from __future__ import annotations

import logging
import os

from src.core.conversation import run_conversation_turn
from src.core.session_store import save_session
from src.shared.session_end import handle_session_end
from models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_PORTAL_LOGIN_URL = "https://login.axisdirect.in/?redirectPage=My%20Page"
_FEATURES_URL     = "https://www.axisdirect.in"
_TICKET_URL       = "https://simplehai.axisdirect.in/portal/index.php/supportPortal/raise-query"
_CDSL_FORM_PATH   = (
    "www.axisdirect.in >> Support >> Downloads >> Demat Forms (CDSL) – 20 "
    ">> CDSL- Application for closing a Demat Account"
)
_NSDL_FORM_PATH   = (
    "www.axisdirect.in >> Support >> Downloads >> Demat Forms (NSDL) – 34 "
    ">> NSDL - Application for closing a Demat Account"
)

# ── Scenario detection keywords ───────────────────────────────────────────────
_ALREADY_CLOSED_PHRASES = (
    "already closed",
    "account already closed",
    "customer account already closed",
    "demat account closed",
    "cannot be closed",
    "purged",
    "demat and trading account",
    "permanently closed",
)
_IN_PROGRESS_PHRASES = (
    "already in process",
    "already created",
    "request already",
    "under process",
    "closure is in process",
    "in progress",
    "closure processed",
    "request created successfully",
)

# ── LLM system prompt for Message 1 (new_request only) ───────────────────────
_MSG1_SYS = """\
You are a warm, empathetic customer service assistant for Axis Direct (Axis Securities Limited).
The customer has requested to close their trading/demat account.

Write Message 1 of the account closure response. Use EXACTLY this structure — do not add or remove sections:

1. Address the customer by name (customer_name from context).
2. Express genuine regret that they want to close their account — be warm, not robotic.
3. Encourage them to visit the features page (features_url from context) to reconsider.
4. Ask if there is anything Axis Direct can do to change their decision.

Rules:
- Keep the tone professional, empathetic, and human — NOT formal corporate-speak.
- Do NOT mention the closure process, deeplinks, or next steps — that comes later.
- Do NOT ask for account details.
- Max 4 sentences.

Return ONLY valid JSON, no markdown fences:
{
  "message": "<the warm empathetic Message 1 text>",
  "quick_replies": ["I still want to close my account", "Go back to main menu"],
  "flow_action": "wait_confirm",
  "reasoning": "<one line>"
}
"""


# ── Message builders (exact spec content — hardcoded, no LLM) ─────────────────

def _msg_already_closed(name: str, demat: str, trading: str) -> str:
    return (
        f"Dear {name},\n\n"
        f"Your Demat account {demat} & Trading Account {trading} is already closed."
    )


def _msg_in_progress(name: str, demat: str, trading: str) -> str:
    return (
        f"Dear {name},\n\n"
        f"The closure of your Demat account {demat} & Trading Account {trading} is in progress. "
        f"You will receive a confirmation mail on your registered E-mail ID with us once your account is closed."
    )


def _msg_new_request_2() -> str:
    return (
        f"Click here {_PORTAL_LOGIN_URL} to login to your account & scroll down to the "
        "\"REQUEST ACCOUNT CLOSURE\" tab to begin the closure process. "
        f"If you need more information click here {_TICKET_URL} to raise a ticket."
    )


def _msg_new_request_3() -> str:
    return (
        "For NRI account, Non – Individual Account, Joint Holder Demat account or who do not have "
        "access to the online portal, please submit the following documents at the nearest Axis Bank branch:\n\n"
        "i. Submit a Transfer cum closure form\n"
        f"   (Download Path: CDSL – {_CDSL_FORM_PATH})\n"
        f"   (NSDL – {_NSDL_FORM_PATH})\n\n"
        "ii. Self Attested PAN Card copy\n\n"
        "iii. Stamped and signed Client Master List (CML) of the target account "
        "(If there are Holdings to be transferred to another account)"
    )


def _msg_new_request_4() -> str:
    return "Thank you! Please click below on the Main Menu tab for more services."


def _msg_changed_mind(name: str) -> str:
    return (
        f"That's great to hear, {name}! We're glad you've decided to stay with Axis Direct. "
        "Feel free to reach out if you need any help with your account."
    )


# ── Scenario detection ─────────────────────────────────────────────────────────

def _detect_scenario(api_result: dict) -> str:
    """Returns: 'already_closed' | 'in_progress' | 'new_request'"""
    api_resp = api_result.get("api_response", {})
    reason   = ""

    if isinstance(api_resp, dict):
        reason = str(
            api_resp.get("reason") or
            api_resp.get("message") or
            api_resp.get("status") or ""
        ).lower()
    elif isinstance(api_resp, str):
        reason = api_resp.lower()

    logger.info("[CLOSURE] API reason: %r", reason)

    if any(p in reason for p in _ALREADY_CLOSED_PHRASES):
        if "purged" in reason or "permanently closed" in reason:
            return "already_closed"
        if "already closed" in reason or "cannot be closed" in reason:
            return "already_closed"

    if any(p in reason for p in _IN_PROGRESS_PHRASES):
        return "in_progress"

    return "new_request"


# ── Intent helpers for confirm_proceed ───────────────────────────────────────

def _wants_to_proceed(text: str) -> bool:
    """Customer confirms they still want closure."""
    t = text.strip().lower()
    return any(w in t for w in (
        "yes", "still", "proceed", "close", "confirm", "sure",
        "want to close", "go ahead", "continue",
    ))


def _wants_to_cancel(text: str) -> bool:
    """Customer changed their mind / wants to go back."""
    t = text.strip().lower()
    return any(w in t for w in (
        "no", "back", "cancel", "changed", "never mind", "nevermind",
        "main menu", "menu", "don't", "dont", "stop",
    ))


# ── Gateway call ──────────────────────────────────────────────────────────────

def _call_closure_api(
    sub_account_id: str,
    email: str,
    name: str,
    closure_type: str = "demat",
    dp_account_no: str = "",
) -> dict:
    from src.gateways.gateway_client import create_closure_request
    logger.info("[CLOSURE] Calling Gateway create_closure_request sub=%s type=%s",
                sub_account_id, closure_type)
    try:
        return create_closure_request(
            sub_account_id,
            email,
            name,
            type_of_account_closure=closure_type,
            dp_account_no=dp_account_no,
        )
    except Exception as exc:
        logger.error("[CLOSURE] Gateway call failed: %s", exc)
        return {"api_response": {"reason": "error"}, "error": str(exc)}


# ── Main handler ──────────────────────────────────────────────────────────────

def handle_closure(
    state: SessionState,
    customer_message: str,
) -> tuple[InternalMessageResponse, SessionState]:
    """
    Account closure flow handler.
    Triggered by intent classifier — not by main menu button.
    Requires sub_account_id (post-login).
    """
    fs = state.flow_state

    # ── START: fetch profile + call closure API ───────────────────────────────
    if fs == "start":
        sub_id  = state.sub_account_id or ""
        name    = ""
        demat   = sub_id
        trading = sub_id
        email   = ""

        if sub_id:
            try:
                from src.gateways.customer_api import get_customer_profile
                profile = get_customer_profile(sub_id)
                name    = profile.name or sub_id
                demat   = profile.demat_account_no or sub_id
                trading = profile.trading_account_no or sub_id
                email   = profile.registered_email or ""
            except Exception as exc:
                logger.warning("[CLOSURE] profile fetch failed: %s", exc)
                name = sub_id

        api_result = _call_closure_api(sub_id, email, name)
        scenario   = _detect_scenario(api_result)

        logger.info("[CLOSURE] conv=%s sub=%s scenario=%s",
                    state.conversation_id, sub_id, scenario)

        collected = {
            **state.collected_data,
            "name":    name,
            "demat":   demat,
            "trading": trading,
            "scenario": scenario,
        }

        # ── Already closed ────────────────────────────────────────────────────
        if scenario == "already_closed":
            msg = _msg_already_closed(name, demat, trading)
            hist = state.history + [
                {"role": "user",      "content": customer_message},
                {"role": "assistant", "content": msg},
            ]
            ns = state.model_copy(update={
                "flow": "closure",
                "flow_state": "session_end_response",
                "collected_data": collected,
                "history": hist,
            })
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=msg,
                    quick_reply_options=["Go back to main menu", "End Chat"],
                    flow_state="session_end_response",
                    status="ok",
                ),
                ns,
            )

        # ── In progress ───────────────────────────────────────────────────────
        if scenario == "in_progress":
            msg = _msg_in_progress(name, demat, trading)
            hist = state.history + [
                {"role": "user",      "content": customer_message},
                {"role": "assistant", "content": msg},
            ]
            ns = state.model_copy(update={
                "flow": "closure",
                "flow_state": "session_end_response",
                "collected_data": collected,
                "history": hist,
            })
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=msg,
                    quick_reply_options=["Go back to main menu", "End Chat"],
                    flow_state="session_end_response",
                    status="ok",
                ),
                ns,
            )

        # ── New request — Message 1 via Qwen LLM ─────────────────────────────
        resp = run_conversation_turn(
            state_data={"flow": "closure", "flow_state": "new_request_msg1"},
            history=state.history,
            customer_message=customer_message,
            backend_data={
                "customer_name": name,
                "features_url":  _FEATURES_URL,
            },
            system_override=_MSG1_SYS,
        )
        msg1 = resp.get("message", _msg_new_request_1_fallback(name))

        hist = state.history + [
            {"role": "user",      "content": customer_message},
            {"role": "assistant", "content": msg1},
        ]
        ns = state.model_copy(update={
            "flow":          "closure",
            "flow_state":    "confirm_proceed",
            "collected_data": collected,
            "history":       hist,
        })
        save_session(state.conversation_id, ns)
        logger.info("[CLOSURE] conv=%s → Message 1 sent, waiting for confirmation",
                    state.conversation_id)
        return (
            InternalMessageResponse(
                reply_message=msg1,
                quick_reply_options=["I still want to close my account", "Go back to main menu"],
                flow_state="confirm_proceed",
                status="ok",
            ),
            ns,
        )

    # ── CONFIRM PROCEED — wait for customer's decision ────────────────────────
    if fs == "confirm_proceed":
        name = state.collected_data.get("name", "")

        # Customer changed their mind → thank + exit
        if _wants_to_cancel(customer_message):
            msg = _msg_changed_mind(name)
            hist = state.history + [
                {"role": "user",      "content": customer_message},
                {"role": "assistant", "content": msg},
            ]
            ns = state.model_copy(update={
                "flow_state": "session_end_response",
                "history":    hist,
            })
            save_session(state.conversation_id, ns)
            logger.info("[CLOSURE] conv=%s → customer changed mind, ending flow",
                        state.conversation_id)
            return (
                InternalMessageResponse(
                    reply_message=msg,
                    quick_reply_options=["Go back to main menu", "End Chat"],
                    flow_state="session_end_response",
                    status="ok",
                ),
                ns,
            )

        # Customer confirms they still want to proceed → ask closure type first
        if _wants_to_proceed(customer_message):
            _TYPE_ASK_MSG = (
                "Understood. Before I proceed, please let me know which account you'd like to close:"
            )
            hist = state.history + [
                {"role": "user",      "content": customer_message},
                {"role": "assistant", "content": _TYPE_ASK_MSG},
            ]
            ns = state.model_copy(update={
                "flow_state": "closure_type_selection",
                "history":    hist,
            })
            save_session(state.conversation_id, ns)
            logger.info("[CLOSURE] conv=%s → confirmed, asking closure type",
                        state.conversation_id)
            return (
                InternalMessageResponse(
                    reply_message=_TYPE_ASK_MSG,
                    quick_reply_options=["Demat Account", "Trading Account", "Both"],
                    flow_state="closure_type_selection",
                    status="ok",
                ),
                ns,
            )

        # Ambiguous — re-ask
        re_ask = (
            "Please let us know — would you still like to proceed with closing your account, "
            "or can we help you with something else?"
        )
        hist = state.history + [
            {"role": "user",      "content": customer_message},
            {"role": "assistant", "content": re_ask},
        ]
        ns = state.model_copy(update={"history": hist})
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=re_ask,
                quick_reply_options=["I still want to close my account", "Go back to main menu"],
                flow_state="confirm_proceed",
                status="reprompt",
            ),
            ns,
        )

    # ── CLOSURE TYPE SELECTION ────────────────────────────────────────────────
    if fs == "closure_type_selection":
        msg_lower = customer_message.strip().lower()

        # Detect which type the customer selected
        if any(w in msg_lower for w in ("both", "demat and trading", "demat & trading", "all")):
            closure_type = "demat and trading"
            display_type = "Demat & Trading Account"
        elif any(w in msg_lower for w in ("trading", "trade")):
            closure_type = "trading"
            display_type = "Trading Account"
        elif any(w in msg_lower for w in ("demat", "dp", "both", "demat account")):
            closure_type = "demat"
            display_type = "Demat Account"
        else:
            # Ambiguous — re-ask
            re_ask = "Please select which account you'd like to close:"
            hist = state.history + [
                {"role": "user",      "content": customer_message},
                {"role": "assistant", "content": re_ask},
            ]
            ns = state.model_copy(update={"history": hist})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=re_ask,
                    quick_reply_options=["Demat Account", "Trading Account", "Both"],
                    flow_state="closure_type_selection",
                    status="reprompt",
                ),
                ns,
            )

        # Store closure type and send Message 2
        msg2 = _msg_new_request_2()
        hist = state.history + [
            {"role": "user",      "content": customer_message},
            {"role": "assistant", "content": msg2},
        ]
        ns = state.model_copy(update={
            "flow_state": "new_request_msg3",
            "history":    hist,
            "collected_data": {**state.collected_data, "closure_type": closure_type},
        })
        save_session(state.conversation_id, ns)
        logger.info("[CLOSURE] conv=%s → closure_type=%r, sending Message 2 (deeplink)",
                    state.conversation_id, closure_type)
        return (
            InternalMessageResponse(
                reply_message=msg2,
                quick_reply_options=["Continue", "Go back to main menu"],
                flow_state="new_request_msg3",
                status="ok",
            ),
            ns,
        )

    # ── NEW REQUEST — Message 3 (offline / NRI) ───────────────────────────────
    if fs == "new_request_msg3":
        if any(s in customer_message.lower() for s in ("main menu", "back", "cancel")):
            return _go_to_menu(state)

        msg3 = _msg_new_request_3()
        hist = state.history + [
            {"role": "user",      "content": customer_message},
            {"role": "assistant", "content": msg3},
        ]
        ns = state.model_copy(update={"flow_state": "new_request_msg4", "history": hist})
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=msg3,
                quick_reply_options=["Main Menu", "End Chat"],
                flow_state="new_request_msg4",
                status="ok",
            ),
            ns,
        )

    # ── NEW REQUEST — Message 4 (Thank you) ───────────────────────────────────
    if fs == "new_request_msg4":
        msg4 = _msg_new_request_4()
        hist = state.history + [
            {"role": "user",      "content": customer_message},
            {"role": "assistant", "content": msg4},
        ]
        ns = state.model_copy(update={"flow_state": "session_end_response", "history": hist})
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=msg4,
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state="session_end_response",
                status="ok",
            ),
            ns,
        )

    # ── SESSION END ───────────────────────────────────────────────────────────
    if fs == "session_end_response":
        return handle_session_end(state, customer_message)

    logger.warning("[CLOSURE] unknown flow_state %r — reset", fs)
    return handle_closure(
        state.model_copy(update={"flow_state": "start"}), customer_message
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _msg_new_request_1_fallback(name: str) -> str:
    """Used only if Qwen LLM call fails."""
    return (
        f"Dear {name},\n\n"
        f"We are sorry to know that you wish to close your account with us. "
        f"We urge you to once click here {_FEATURES_URL} to know the features of your account. "
        "We would like to know if there's anything we can do to change your decision."
    )


def _go_to_menu(state: SessionState) -> tuple[InternalMessageResponse, SessionState]:
    ns = state.model_copy(update={
        "flow": None, "flow_state": "main_menu", "collected_data": {},
    })
    save_session(state.conversation_id, ns)
    return (
        InternalMessageResponse(
            reply_message="Taking you back to the main menu. How can I help you?",
            quick_reply_options=[],
            flow_state="main_menu",
            status="route_to_entry",
        ),
        ns,
    )
