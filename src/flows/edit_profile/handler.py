"""
chatbot_web/src/flows/edit_profile/handler.py
-----------------------------------------------
Edit Profile — pre-login flow, no tools needed (static deeplink).

Pre-login boundary: no account data available. Show login portal link.

State machine:
  start → show Axis Direct portal deeplink → session_end_response
"""

from __future__ import annotations

import logging
import os

from src.core.session_store import save_session
from src.shared.session_end import handle_session_end
from models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

_PORTAL = os.getenv("AXIS_DIRECT_PORTAL_URL", "https://login.axisdirect.in")

_DEEPLINK_MSG = (
    "To edit your profile details, please log in to your Axis Direct account:\n\n"
    f"🔗 Login here: {_PORTAL}\n\n"
    "Once logged in, you can update from My Profile / Account Settings:\n"
    "  • Email address & mobile number\n"
    "  • Mailing address\n"
    "  • Bank account details\n"
    "  • Nominee details\n"
    "  • PAN information\n\n"
    "Note: Some changes (bank details, nominee) require KYC documents or OTP verification."
)


def handle_edit_profile(
    state: SessionState,
    customer_message: str,
) -> tuple[InternalMessageResponse, SessionState]:

    fs = state.flow_state

    if fs in ("start", "edit_profile"):
        # Static deeplink — no LLM needed, URLs must not be altered
        reply = _DEEPLINK_MSG
        hist = state.history + [
            {"role": "user",      "content": customer_message},
            {"role": "assistant", "content": reply},
        ]

        # Account status check after showing deeplink (post-login context only)
        next_fs = "account_status_check" if state.sub_account_id else "session_end_response"

        new_state = state.model_copy(update={
            "flow": "edit_profile", "flow_state": next_fs, "history": hist,
        })
        save_session(state.conversation_id, new_state)
        logger.info("[EDIT_PROFILE] conv=%s deeplink sent", state.conversation_id)
        return (
            InternalMessageResponse(
                reply_message=reply,
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state=next_fs,
                status="ok",
            ),
            new_state,
        )

    # ── Account status check after deeplink (post-login only) ─────────────────
    if fs == "account_status_check":
        from src.core.strands_agent import get_profile
        try:
            profile = get_profile(state.sub_account_id or "")
            status  = profile.account_status
        except Exception:
            status = "active"

        if status in ("deactivated", "purged"):
            deact_msg = (
                f"We noticed your account is currently {status}. "
                "Please contact our support team to reactivate:\n\n"
                "📞 Customer Care: 022-40508080 / 022-61480808\n"
                "🌐 https://www.axisdirect.in/contact-us"
            )
            hist = state.history + [
                {"role": "user",      "content": customer_message},
                {"role": "assistant", "content": deact_msg},
            ]
            ns = state.model_copy(update={"flow_state": "session_end_response", "history": hist})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=deact_msg,
                    quick_reply_options=["Go back to main menu", "End Chat"],
                    flow_state="session_end_response",
                    status="ok",
                ),
                ns,
            )

        # Active — show thank you
        thank_you = "Thank you for using Axis Direct! Is there anything else I can help you with?"
        hist = state.history + [
            {"role": "user",      "content": customer_message},
            {"role": "assistant", "content": thank_you},
        ]
        ns = state.model_copy(update={"flow_state": "session_end_response", "history": hist})
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=thank_you,
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state="session_end_response",
                status="ok",
            ),
            ns,
        )

    if fs == "session_end_response":
        return handle_session_end(state, customer_message)

    logger.warning("[EDIT_PROFILE] unknown flow_state %r — reset", fs)
    return handle_edit_profile(state.model_copy(update={"flow_state": "start"}), customer_message)
