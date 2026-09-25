"""
chatbot_web/src/flows/bank_query/handler.py
---------------------------------------------
Bank Query — pre-login flow, no Strands tools needed (static redirect).

State machine:
  start → show Axis Bank redirect message → session_end_response
"""

from __future__ import annotations

import logging
import os

from src.core.session_store import save_session
from src.shared.session_end import handle_session_end
from models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

_CARE  = os.getenv("AXIS_BANK_CARE_NUMBER", "1860 419 5555 / 1860 500 5555")
_WEB   = os.getenv("AXIS_BANK_WEBSITE",     "https://www.axisbank.com")
_BRANCH = os.getenv("AXIS_BANK_BRANCH_LOCATOR", "https://branch.axisbank.com/")

_REDIRECT_MSG = (
    "Thank you for reaching out! For any queries related to your loan account, "
    "credit card or savings bank account, you can contact their customer support at:\n\n"
    f"📞 Customer Care: {_CARE}\n"
    f"🌐 Website: {_WEB}\n"
    f"Branch locator: {_BRANCH}"
)


def handle_bank_query(
    state: SessionState,
    customer_message: str,
) -> tuple[InternalMessageResponse, SessionState]:

    fs = state.flow_state

    if fs in ("start", "bank_query"):
        # Static redirect — no LLM needed, URLs must not be altered
        reply = _REDIRECT_MSG
        hist = state.history + [
            {"role": "user",      "content": customer_message},
            {"role": "assistant", "content": reply},
        ]
        new_state = state.model_copy(update={
            "flow": "bank_query", "flow_state": "session_end_response", "history": hist,
        })
        save_session(state.conversation_id, new_state)
        logger.info("[BANK_QUERY] conv=%s redirect sent", state.conversation_id)
        return (
            InternalMessageResponse(
                reply_message=reply,
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state="session_end_response",
                status="ok",
            ),
            new_state,
        )

    if fs == "session_end_response":
        return handle_session_end(state, customer_message)

    logger.warning("[BANK_QUERY] unknown flow_state %r — reset", fs)
    return handle_bank_query(state.model_copy(update={"flow_state": "start"}), customer_message)
