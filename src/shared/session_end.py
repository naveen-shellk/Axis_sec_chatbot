"""
chatbot_web/src/shared/session_end.py
---------------------------------------
Shared session-end block, reused by every flow.

Both flows end the same way:
  - Show "Go back to main menu" / "End Chat" options
  - Customer taps "Go back to main menu" → clear flow, return to entry router
  - Customer taps "End Chat" → clear session entirely
  - No selection within TTL → session expires naturally
"""

from __future__ import annotations

import logging

from src.core.session_store import clear_session, save_session
from models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

_END_OPTIONS = ["Go back to main menu", "End Chat"]

_GOODBYE_MSG = (
    "Thank you for reaching out to Axis Direct! "
    "Have a great day. If you need help again, we're always here."
)

_BACK_TO_MENU_MSG = (
    "Taking you back to the main menu. How can I help you?"
)

_MAIN_MENU_OPTIONS = [
    "Bank Query",
    "How To Trade",
    "Need More Help",
    "Edit Profile",
]


def handle_session_end(
    state: SessionState,
    customer_message: str,
) -> tuple[InternalMessageResponse, SessionState]:
    """
    Handle the session-end decision node.
    Called when flow_state == "session_end_response".

    Returns:
      - "Go back to main menu" → status="route_to_entry", flow reset
      - "End Chat" or unrecognised → status="end", session cleared
    """
    import re as _re
    msg_lower = customer_message.strip().lower()
    # Normalise for exact-navigation comparison: strip trailing punctuation.
    msg_norm  = _re.sub(r"[.!?]+$", "", msg_lower).strip()

    go_back_signals = {
        "go back", "go back to main menu", "main menu", "back", "menu", "home",
        "start over", "restart", "other", "yes", "ok", "okay",
    }
    end_signals = {
        "end", "end chat", "bye", "goodbye", "exit", "no", "done",
        "thanks", "thank you", "that's all", "thats all", "no thanks",
    }

    # Only treat the turn as a NAVIGATION command when the WHOLE message is that
    # command (an exact button tap or a bare word), NOT when a trigger word
    # merely appears inside a longer sentence. A real new request like "what are
    # Axis Bank's HOME loan options" must fall through to fresh-intent re-routing
    # instead of matching the "home" go-back word.
    _MAX_NAV_WORDS = 4  # navigation taps are short; sentences are longer

    def _matches(signals: set[str]) -> bool:
        # Exact whole-message match (after punctuation strip) always wins.
        if msg_norm in signals:
            return True
        # Otherwise, only allow a single-word trigger when the message itself is
        # short (<= _MAX_NAV_WORDS) — so "back", "menu please" count, but a full
        # sentence containing the word does not.
        if len(msg_norm.split()) > _MAX_NAV_WORDS:
            return False
        for s in signals:
            if " " in s:
                if s in msg_norm:
                    return True
            elif _re.search(rf"\b{_re.escape(s)}\b", msg_norm):
                return True
        return False

    if _matches(go_back_signals):
        new_state = state.model_copy(update={
            "flow": None,
            "flow_state": "main_menu",
            "collected_data": {},
        })
        save_session(state.conversation_id, new_state)
        logger.info("[SESSION_END] conv=%s → back to main menu", state.conversation_id)
        return (
            InternalMessageResponse(
                reply_message=_BACK_TO_MENU_MSG,
                quick_reply_options=_MAIN_MENU_OPTIONS,
                flow_state="main_menu",
                status="route_to_entry",
            ),
            new_state,
        )

    if _matches(end_signals):
        clear_session(state.conversation_id)
        logger.info("[SESSION_END] conv=%s → end chat", state.conversation_id)
        return (
            InternalMessageResponse(
                reply_message=_GOODBYE_MSG,
                quick_reply_options=[],
                flow_state="ended",
                status="end",
            ),
            state.model_copy(update={"flow_state": "ended"}),
        )

    # Not a session-end signal — treat as a new intent, reset flow and re-route
    # This handles the case where customer sends a new message after a flow completes
    logger.info("[SESSION_END] conv=%s — fresh intent detected, resetting flow", state.conversation_id)
    new_state = state.model_copy(update={
        "flow": None,
        "flow_state": "main_menu",
        "collected_data": {},
    })
    save_session(state.conversation_id, new_state)
    # Return route_to_entry so entry handler re-classifies the message
    return (
        InternalMessageResponse(
            reply_message="",   # entry handler will fill this after re-routing
            quick_reply_options=[],
            flow_state="main_menu",
            status="route_to_entry",
        ),
        new_state,
    )
