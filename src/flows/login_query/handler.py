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

from src.core.conversation import run_conversation_turn
from src.core.session_store import save_session
from src.shared.session_end import handle_session_end
from models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

_ESCALATE_CODES = frozenset({
    "D004","D006","D008","D012","D013","D014","D016",
    "D017","D020","D021","D022","D025",
})

_CODE_MESSAGES = {
    "D005": "Your account is deactivated because your KYC registration is not confirmed. Complete KYC activation here: https://simplehai.axisdirect.in/app/index.php/user/auth/activateUser",
    "D002": "Your Demat account is closed. Follow the Demat account activation and linking process: https://www.axisdirect.in",
    "D003": "Your account is deactivated due to a bank account issue. Please update your bank account details: https://www.axisdirect.in",
    "D007": "Your account requires KYC attribute verification. Submit the required KYC documents: https://www.axisdirect.in",
    "D009": "Your account is deactivated due to Account Opening Charges (AOC) not being recovered. Complete reactivation: https://www.axisdirect.in",
    "D010": "Your account has a duplicate email/mobile issue. Please contact our support team: https://www.axisdirect.in/contact-us",
    "D018": "Your NRI account has a bank linking issue. Please update your bank details: https://www.axisdirect.in",
    "D019": "Your account requires a FATCA declaration. Complete the FATCA process: https://www.axisdirect.in",
    "D024": "Your account has been dormant for more than 24 months. Follow the dormant account reactivation process: https://www.axisdirect.in",
}

_TROUBLESHOOT_LINKS = {
    "login_guide":      "https://www.axisdirect.in/blogs/how-do-i-login-access-unlock-change-password-of-my-account",
    "app_issues_guide": "https://www.axisdirect.in/blogs/what-should-i-do-when-i-face-app-stuck-error-404-site-not-found-website-not-resp",
}
_FTL_LINKS = {
    "activation_link": "https://simplehai.axisdirect.in/app/index.php/user/auth/activateUser",
    "guide_link":      "https://www.axisdirect.in/blogs/how-do-i-login-access-unlock-change-password-of-my-account",
}
_TICKET_LINK = "https://simplehai.axisdirect.in/portal/index.php/supportPortal/raise-query"

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

_DEACT_SYS = """\
You are informing a deactivated Axis Direct account customer of the specific action they need to take.
Use deactivation_message from context verbatim. Be empathetic.
Return JSON:
{"message": "<empathetic message with the deactivation_message action>",
 "quick_replies": ["Go back to main menu", "End Chat"],
 "flow_action": "session_end", "reasoning": ""}
"""

_TICKET_SYS = """\
You are telling an Axis Direct customer their account issue requires manual review.
Apologise and provide the ticket_link from context so they can raise a support request.
Return JSON:
{"message": "<apologetic message + ticket link>",
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
        from src.core.strands_agent import get_profile
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
            if deact_code and deact_code in _CODE_MESSAGES and deact_code not in _ESCALATE_CODES:
                # Self-serviceable deactivation — show specific action link
                resp, hist = _llm(state, customer_message,
                                  {"deactivation_message": _CODE_MESSAGES[deact_code]},
                                  _DEACT_SYS, "session_end_response")
                ns = state.model_copy(update={"flow_state": "session_end_response", "history": hist})
                save_session(state.conversation_id, ns)
                return (InternalMessageResponse(reply_message=resp["message"],
                                                quick_reply_options=["Go back to main menu", "End Chat"],
                                                flow_state="session_end_response", status="ok"), ns)
            else:
                # Escalate codes or unknown deact code → direct live agent (no confirmation)
                from src.flows.need_more_help.handler import _is_business_hours
                import os as _os
                if _is_business_hours():
                    _ESCALATE_MSG = (
                        "Your account requires manual review by our support team.\n\n"
                        "Let me connect you to a customer service representative "
                        "who will assist you with your account issue. Please hold on."
                    )
                    hist = state.history + [
                        {"role": "user",      "content": customer_message},
                        {"role": "assistant", "content": _ESCALATE_MSG},
                    ]
                    ns = state.model_copy(update={
                        "flow_state": "escalated",
                        "escalate":   True,
                        "history":    hist,
                    })
                    save_session(state.conversation_id, ns)
                    logger.info("[LOGIN_QUERY] conv=%s deact_code=%r → escalating (within hours)",
                                state.conversation_id, deact_code)
                    return (InternalMessageResponse(
                        reply_message=_ESCALATE_MSG,
                        quick_reply_options=[],
                        flow_state="escalated",
                        status="escalate",
                        eventid="1002",
                    ), ns)
                else:
                    _ticket = _os.getenv(
                        "SUPPORT_TICKET_URL",
                        "https://simplehai.axisdirect.in/portal/index.php/supportPortal/raise-query",
                    )
                    _OOH_MSG = (
                        "Your account issue requires manual review by our support team.\n\n"
                        "Our live agents are available Monday to Friday, 9:00 AM – 6:00 PM IST.\n"
                        f"Please raise a support ticket: 🎫 {_ticket}"
                    )
                    hist = state.history + [
                        {"role": "user",      "content": customer_message},
                        {"role": "assistant", "content": _OOH_MSG},
                    ]
                    ns = state.model_copy(update={"flow_state": "session_end_response", "history": hist})
                    save_session(state.conversation_id, ns)
                    logger.info("[LOGIN_QUERY] conv=%s deact_code=%r → out of hours ticket",
                                state.conversation_id, deact_code)
                    return (InternalMessageResponse(
                        reply_message=_OOH_MSG,
                        quick_reply_options=["Go back to main menu", "End Chat"],
                        flow_state="session_end_response",
                        status="ok",
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
