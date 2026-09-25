"""
chatbot_web/src/flows/ipo/handler.py
--------------------------------------
IPO flow — post-login, one-shot deeplink with customer name personalisation.
"""

from __future__ import annotations

import logging

from src.core.conversation import run_conversation_turn
from src.core.session_store import save_session
from src.shared.session_end import handle_session_end
from models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

_IPO_URL = "https://simplehai.axisdirect.in/ipos"

_IPO_SYS = """\
You are helping an Axis Direct customer with IPO applications.
Write a warm, personalised message:
1. Address customer by name (customer_name from context, or "Valued Customer")
2. Thank them for choosing Axis Direct
3. Direct them to the IPO link from context
4. Tell them to click Main Menu for other queries

Return JSON:
{"message": "<personalised IPO message with URL>",
 "quick_replies": ["Go back to main menu", "End Chat"],
 "flow_action": "session_end", "reasoning": ""}
"""


def handle_ipo(state: SessionState, customer_message: str) -> tuple[InternalMessageResponse, SessionState]:

    fs = state.flow_state

    if fs in ("start", "ipo_redirect"):
        customer_name = state.customer_name or "Valued Customer"
        if not customer_name or customer_name == "Valued Customer":
            try:
                from src.gateways.customer_api import get_customer_profile
                profile = get_customer_profile(state.sub_account_id or "")
                customer_name = profile.name or "Valued Customer"
                state = state.model_copy(update={"customer_name": customer_name})
            except Exception as exc:
                logger.debug("[IPO] profile fetch failed: %s", exc)

        resp = run_conversation_turn(
            state_data={"flow": "ipo", "flow_state": "ipo_redirect"},
            history=state.history,
            customer_message=customer_message,
            backend_data={"customer_name": customer_name, "ipo_url": _IPO_URL},
            system_override=_IPO_SYS,
        )
        hist = state.history + [
            {"role": "user",      "content": customer_message},
            {"role": "assistant", "content": resp["message"]},
        ]
        new_state = state.model_copy(update={
            # Keep flow set so the terminal session_end_response state routes
            # back here (→ handle_session_end) on the next turn. flow=None
            # orphans the end node and misroutes "Go back to main menu"/"End Chat".
            "flow": "ipo", "flow_state": "session_end_response", "history": hist,
        })
        save_session(state.conversation_id, new_state)
        return (
            InternalMessageResponse(
                reply_message=resp["message"],
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state="session_end_response", status="ok",
            ),
            new_state,
        )

    if fs == "session_end_response":
        return handle_session_end(state, customer_message)

    logger.warning("[IPO] unknown flow_state %r — reset", fs)
    return handle_ipo(state.model_copy(update={"flow_state": "start"}), customer_message)
