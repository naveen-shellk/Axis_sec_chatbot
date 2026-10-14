"""
chatbot_web/src/flows/login_query/handler.py
---------------------------------------------
Login Query flow — post-login.

Decision tree:
  start → fetch profile
    → Active + FTL done     → show login troubleshooting links
    → Active + FTL pending  → show FTL activation steps
    → Deactivated           → check deactivation code → show specific action or raise ticket
"""

from __future__ import annotations

import logging

from agent.core.conversation import run_conversation_turn
from agent.core.session_store import save_session
from agent.shared.session_end import handle_session_end
from agent.models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

# Deactivation reason codes + messages now live in agent/core/deactivation.py
# (shared across all post-login flows). login_query calls deactivation_message().

_TROUBLESHOOT_LINKS = {
    "login_guide":      "https://www.axisdirect.in/blogs/how-do-i-login-access-unlock-change-password-of-my-account",
    "app_issues_guide": "https://www.axisdirect.in/blogs/what-should-i-do-when-i-face-app-stuck-error-404-site-not-found-website-not-resp",
}
_FTL_LINKS = {
    "activation_link": "https://simplehai.axisdirect.in/app/index.php/user/auth/activateUser",
    "guide_link":      "https://www.axisdirect.in/blogs/how-do-i-login-access-unlock-change-password-of-my-account",
}
_TROUBLESHOOT_SYS = """\
You are helping an Axis Direct customer who is having login trouble.
Account is ACTIVE and FTL is complete. Provide troubleshooting guidance with links from context.
Return JSON:
{"message": "<troubleshooting message with login_guide and app_issues_guide links>",
 "quick_replies": ["Go back to main menu", "End Chat"],
 "flow_action": "session_end", "reasoning": ""}
"""

_FTL_SYS = """\
You are helping an Axis Direct customer complete their First Time Login (FTL).
Explain FTL activation steps using activation_link and guide_link from context.
Return JSON:
{"message": "<FTL explanation with activation link and guide>",
 "quick_replies": ["Go back to main menu", "End Chat"],
 "flow_action": "session_end", "reasoning": ""}
"""

def _llm(state, msg, bd, sys, next_fs):
    resp = run_conversation_turn(
        state_data={"flow": "login_query", "flow_state": next_fs},
        history=state.history,
        customer_message=msg,
        backend_data=bd,
        system_override=sys,
    )
    hist = state.history + [
        {"role": "user",      "content": msg},
        {"role": "assistant", "content": resp["message"]},
    ]
    return resp, hist


def handle_login_query(state: SessionState, customer_message: str) -> tuple[InternalMessageResponse, SessionState]:

    fs = state.flow_state

    if fs == "start":
        from agent.core.langchain_agent import get_profile
        _ERROR_MSG = (
            "We were unable to retrieve your account information at this time.\n\n"
            "Please try again later or contact support: 📞 022-40508080 / 022-61480808"
        )
        try:
            profile       = get_profile(state.sub_account_id or "")
            if not profile or not hasattr(profile, "account_status"):
                raise ValueError("Empty profile")
            acct_status   = profile.account_status
            portal_status = profile.portal_status
            deact_code    = profile.deactivation_code.upper()
        except Exception as exc:
            logger.error("[LOGIN_QUERY] profile fetch failed: %s — showing error", exc)
            hist = state.history + [
                {"role": "user",      "content": customer_message},
                {"role": "assistant", "content": _ERROR_MSG},
            ]
            ns = state.model_copy(update={"flow_state": "session_end_response", "history": hist})
            save_session(state.conversation_id, ns)
            return (InternalMessageResponse(
                reply_message=_ERROR_MSG,
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state="session_end_response", status="error",
            ), ns)

        if acct_status == "active":
            if portal_status == 1:
                resp, hist = _llm(state, customer_message, _TROUBLESHOOT_LINKS, _TROUBLESHOOT_SYS, "session_end_response")
            else:
                resp, hist = _llm(state, customer_message, _FTL_LINKS, _FTL_SYS, "session_end_response")
        else:
            # Deactivated/purged → shared empathetic, reason-aware message
            # (automated code → reason + reactivation step; escalate/unknown →
            # apology + support ticket). One source of truth across all flows.
            from agent.core.deactivation import deactivation_message
            _msg = deactivation_message(
                deact_code, profile.name,
                purged=(acct_status == "purged"),
            )
            hist = state.history + [
                {"role": "user",      "content": customer_message},
                {"role": "assistant", "content": _msg},
            ]
            ns = state.model_copy(update={"flow_state": "session_end_response", "history": hist})
            save_session(state.conversation_id, ns)
            logger.info("[LOGIN_QUERY] conv=%s deact_code=%r → shared deactivation message",
                        state.conversation_id, deact_code)
            return (InternalMessageResponse(
                reply_message=_msg,
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state="session_end_response", status="ok",
            ), ns)

        ns = state.model_copy(update={"flow_state": "session_end_response", "history": hist})
        save_session(state.conversation_id, ns)
        return (InternalMessageResponse(reply_message=resp["message"],
                                        quick_reply_options=["Go back to main menu", "End Chat"],
                                        flow_state="session_end_response", status="ok"), ns)

    if fs == "session_end_response":
        return handle_session_end(state, customer_message)

    logger.warning("[LOGIN_QUERY] unknown flow_state %r — reset", fs)
    return handle_login_query(state.model_copy(update={"flow_state": "start"}), customer_message)
