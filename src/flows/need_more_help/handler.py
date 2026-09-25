"""
chatbot_web/src/flows/need_more_help/handler.py
-------------------------------------------------
Need More Help flow — matches the approved flow diagram exactly.

State machine:
  start
    ├── Within business hours?
    │       YES → Ask confirmation "Do you want to connect to a live agent?"
    │               → confirm_agent
    │                   YES → escalate (eventid 1002) → Stop
    │                   NO  → Thank You + main menu options → session_end_response
    │                   Ambiguous → re-ask
    │
    │       NO  → Show ticket deeplink message
    │               → out_of_hours_thankyou
    │                   "Go back to main menu" → main menu
    │                   "End Chat"             → feedback_message → stop
    │                   No selection (idle)    → 15-min TTL handles cleanup

Timestamp-based no-response (within hours only):
  When bot sends the confirmation prompt, confirm_sent_at is stored.
  On next customer message, elapsed time is checked:
    < 60s              → normal YES/NO handling
    60s–120s (count=0) → first miss: re-send prompt, reset timestamp, count=1
    >60s    (count≥1)  → second miss: Thank You + session end

Business hours: Monday–Friday 9:00 AM – 6:00 PM IST (UTC+5:30)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone, timedelta

from src.core.session_store import save_session
from src.shared.session_end import handle_session_end
from models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

_TICKET_LINK = os.getenv(
    "SUPPORT_TICKET_URL",
    "https://simplehai.axisdirect.in/portal/index.php/supportPortal/raise-query",
)

# ── Business hours ────────────────────────────────────────────────────────────
_IST_OFFSET = timedelta(hours=5, minutes=30)
_BIZ_START  = 9
_BIZ_END    = 18
_BIZ_DAYS   = {0, 1, 2, 3, 4}   # Monday=0 … Friday=4

# ── Timeout windows (within-hours confirmation only) ──────────────────────────
_TIMEOUT_SECS = 60   # seconds per window

# ── Messages ──────────────────────────────────────────────────────────────────

# Within hours
_CONFIRM_MSG = (
    "Would you like me to connect you to a live agent?\n\n"
    "Our customer service team is ready to help you."
)
_TIMEOUT_REPROMPT_MSG = (
    "We noticed you haven't responded yet. Would you still like to connect "
    "to a live agent?\n\n"
    "Our customer service team is ready to help you."
)
_LIVE_AGENT_MSG = (
    "I understand you need more help! Let me connect you to one of our "
    "customer service representatives who will assist you further.\n\n"
    "Please hold on — a live agent will be with you shortly."
)
_NO_AGENT_MSG = (
    "No problem! If you need any further assistance, feel free to ask.\n\n"
    "Thank you for reaching out to Axis Direct!"
)
_TIMEOUT_END_MSG = (
    "Thank you for reaching out to Axis Direct! "
    "If you need help again, we're always here."
)

# Out of hours
_TICKET_MSG = (
    "Our live agents are available Monday to Friday, 9:00 AM – 6:00 PM IST.\n\n"
    "You can raise a support ticket and our team will get back to you:\n"
    f"🎫 Create ticket: {_TICKET_LINK}"
)
_OUT_OF_HOURS_THANKYOU_MSG = (
    "Thank you for reaching out to Axis Direct!\n\n"
    "Is there anything else I can help you with?"
)
_FEEDBACK_MSG = (
    "Before you leave, we'd love to hear your feedback!\n\n"
    "How was your experience with Axis Direct today? Your input helps us improve."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_business_hours() -> bool:
    now_ist = datetime.now(timezone.utc) + _IST_OFFSET
    return (
        now_ist.weekday() in _BIZ_DAYS
        and _BIZ_START <= now_ist.hour < _BIZ_END
    )

def _is_yes(text: str) -> bool:
    t = text.strip().lower()
    return any(w in t for w in ("yes", "yeah", "yep", "sure", "ok", "okay",
                                "connect", "agent", "human", "live"))

def _is_no(text: str) -> bool:
    t = text.strip().lower()
    return any(w in t for w in ("no", "nope", "not", "cancel", "back",
                                "never", "don't", "dont"))

def _elapsed(collected_data: dict) -> float:
    sent_at = collected_data.get("confirm_sent_at")
    if sent_at is None:
        return 0.0
    return time.time() - float(sent_at)

def _save(conv_id, state):
    save_session(conv_id, state)
    return state

# ── Handler ───────────────────────────────────────────────────────────────────

def handle_need_more_help(
    state: SessionState,
    customer_message: str,
    event: str = "Incoming message",
) -> tuple[InternalMessageResponse, SessionState]:

    fs = state.flow_state

    # ── ENTRY ─────────────────────────────────────────────────────────────────
    if fs in ("start", "need_more_help"):

        if not _is_business_hours():
            # Out of hours — show ticket deeplink first
            hist = state.history + [
                {"role": "user",      "content": customer_message},
                {"role": "assistant", "content": _TICKET_MSG},
            ]
            ns = _save(state.conversation_id, state.model_copy(update={
                "flow":       "need_more_help",
                "flow_state": "out_of_hours_thankyou",
                "history":    hist,
            }))
            logger.info("[NEED_MORE_HELP] conv=%s out of hours → ticket deeplink",
                        state.conversation_id)
            return (
                InternalMessageResponse(
                    reply_message=_TICKET_MSG,
                    quick_reply_options=["Go back to main menu", "End Chat"],
                    flow_state="out_of_hours_thankyou",
                    status="ok",
                    eventid="1001",
                ),
                ns,
            )

        # Within hours — ask confirmation + store timestamp
        now_ts = time.time()
        hist = state.history + [
            {"role": "user",      "content": customer_message},
            {"role": "assistant", "content": _CONFIRM_MSG},
        ]
        ns = _save(state.conversation_id, state.model_copy(update={
            "flow":       "need_more_help",
            "flow_state": "confirm_agent",
            "history":    hist,
            "collected_data": {
                **state.collected_data,
                "confirm_sent_at":   now_ts,
                "no_response_count": 0,
            },
        }))
        logger.info("[NEED_MORE_HELP] conv=%s within hours → confirmation prompt sent",
                    state.conversation_id)
        return (
            InternalMessageResponse(
                reply_message=_CONFIRM_MSG,
                quick_reply_options=["Yes", "No"],
                flow_state="confirm_agent",
                status="ok",
                eventid="1001",
            ),
            ns,
        )

    # ── OUT OF HOURS: Thank you + main menu options ───────────────────────────
    if fs == "out_of_hours_thankyou":
        msg_lower = customer_message.strip().lower()

        # "Go back to main menu"
        if any(s in msg_lower for s in ("main menu", "back", "menu", "go back")):
            ns = _save(state.conversation_id, state.model_copy(update={
                "flow":          None,
                "flow_state":    "main_menu",
                "collected_data": {},
            }))
            logger.info("[NEED_MORE_HELP] conv=%s out-of-hours → back to main menu",
                        state.conversation_id)
            return (
                InternalMessageResponse(
                    reply_message="Taking you back to the main menu. How can I help you?",
                    quick_reply_options=[],
                    flow_state="main_menu",
                    status="route_to_entry",
                ),
                ns,
            )

        # "End Chat" → show feedback message first
        if any(s in msg_lower for s in ("end chat", "end", "bye", "exit", "done")):
            hist = state.history + [
                {"role": "user",      "content": customer_message},
                {"role": "assistant", "content": _FEEDBACK_MSG},
            ]
            ns = _save(state.conversation_id, state.model_copy(update={
                "flow_state": "feedback",
                "history":    hist,
            }))
            logger.info("[NEED_MORE_HELP] conv=%s out-of-hours → feedback prompt",
                        state.conversation_id)
            return (
                InternalMessageResponse(
                    reply_message=_FEEDBACK_MSG,
                    quick_reply_options=[],
                    flow_state="feedback",
                    status="ok",
                    eventid="1001",
                ),
                ns,
            )

        # No recognised option — show the thank-you with options
        hist = state.history + [
            {"role": "user",      "content": customer_message},
            {"role": "assistant", "content": _OUT_OF_HOURS_THANKYOU_MSG},
        ]
        ns = _save(state.conversation_id, state.model_copy(update={
            "history": hist,
        }))
        return (
            InternalMessageResponse(
                reply_message=_OUT_OF_HOURS_THANKYOU_MSG,
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state="out_of_hours_thankyou",
                status="ok",
                eventid="1001",
            ),
            ns,
        )

    # ── FEEDBACK state: customer sent something after feedback prompt ──────────
    if fs == "feedback":
        # Any response after feedback → end session
        from src.core.session_store import clear_session
        clear_session(state.conversation_id)
        logger.info("[NEED_MORE_HELP] conv=%s feedback received → session cleared",
                    state.conversation_id)
        return (
            InternalMessageResponse(
                reply_message="Thank you for your feedback! Have a great day. 😊",
                quick_reply_options=[],
                flow_state="ended",
                status="end",
                eventid="1001",
            ),
            state.model_copy(update={"flow_state": "ended"}),
        )

    # ── CONFIRM AGENT (within hours) ──────────────────────────────────────────
    if fs == "confirm_agent":
        no_response_count = state.collected_data.get("no_response_count", 0)
        elapsed_s         = _elapsed(state.collected_data)

        logger.info(
            "[NEED_MORE_HELP] conv=%s confirm_agent elapsed=%.1fs count=%d",
            state.conversation_id, elapsed_s, no_response_count,
        )

        # ── Timeout: first miss (>60s, count=0) ──────────────────────────────
        if elapsed_s >= _TIMEOUT_SECS and no_response_count == 0:
            now_ts = time.time()
            hist   = state.history + [
                {"role": "user",      "content": customer_message},
                {"role": "assistant", "content": _TIMEOUT_REPROMPT_MSG},
            ]
            ns = _save(state.conversation_id, state.model_copy(update={
                "history":        hist,
                "collected_data": {
                    **state.collected_data,
                    "confirm_sent_at":   now_ts,
                    "no_response_count": 1,
                },
            }))
            logger.info("[NEED_MORE_HELP] conv=%s timeout first miss → re-prompt",
                        state.conversation_id)
            return (
                InternalMessageResponse(
                    reply_message=_TIMEOUT_REPROMPT_MSG,
                    quick_reply_options=["Yes", "No"],
                    flow_state="confirm_agent",
                    status="ok",
                    eventid="1001",
                ),
                ns,
            )

        # ── Timeout: second miss (>60s, count≥1) → Thank You + end ───────────
        if elapsed_s >= _TIMEOUT_SECS and no_response_count >= 1:
            hist = state.history + [
                {"role": "user",      "content": customer_message},
                {"role": "assistant", "content": _TIMEOUT_END_MSG},
            ]
            ns = _save(state.conversation_id, state.model_copy(update={
                "flow_state": "session_end_response",
                "history":    hist,
            }))
            logger.info("[NEED_MORE_HELP] conv=%s timeout second miss → session end",
                        state.conversation_id)
            return (
                InternalMessageResponse(
                    reply_message=_TIMEOUT_END_MSG,
                    quick_reply_options=[],
                    flow_state="session_end_response",
                    status="end",
                    eventid="1001",
                ),
                ns,
            )

        # ── Within window: YES → escalate ─────────────────────────────────────
        if _is_yes(customer_message):
            hist = state.history + [
                {"role": "user",      "content": customer_message},
                {"role": "assistant", "content": _LIVE_AGENT_MSG},
            ]
            ns = _save(state.conversation_id, state.model_copy(update={
                "flow_state": "escalated",
                "escalate":   True,
                "history":    hist,
            }))
            logger.info("[NEED_MORE_HELP] conv=%s → YES → escalating (eventid 1002)",
                        state.conversation_id)
            return (
                InternalMessageResponse(
                    reply_message=_LIVE_AGENT_MSG,
                    quick_reply_options=[],
                    flow_state="escalated",
                    status="escalate",
                    eventid="1002",
                ),
                ns,
            )

        # ── Within window: NO → Thank You ─────────────────────────────────────
        if _is_no(customer_message):
            hist = state.history + [
                {"role": "user",      "content": customer_message},
                {"role": "assistant", "content": _NO_AGENT_MSG},
            ]
            ns = _save(state.conversation_id, state.model_copy(update={
                "flow_state": "session_end_response",
                "history":    hist,
            }))
            logger.info("[NEED_MORE_HELP] conv=%s → NO → session end",
                        state.conversation_id)
            return (
                InternalMessageResponse(
                    reply_message=_NO_AGENT_MSG,
                    quick_reply_options=["Go back to main menu", "End Chat"],
                    flow_state="session_end_response",
                    status="ok",
                    eventid="1001",
                ),
                ns,
            )

        # ── Ambiguous → re-ask ────────────────────────────────────────────────
        hist = state.history + [
            {"role": "user",      "content": customer_message},
            {"role": "assistant", "content": _CONFIRM_MSG},
        ]
        ns = _save(state.conversation_id, state.model_copy(update={"history": hist}))
        return (
            InternalMessageResponse(
                reply_message=_CONFIRM_MSG,
                quick_reply_options=["Yes", "No"],
                flow_state="confirm_agent",
                status="reprompt",
                eventid="1001",
            ),
            ns,
        )

    # ── SESSION END ───────────────────────────────────────────────────────────
    if fs == "session_end_response":
        return handle_session_end(state, customer_message)

    logger.warning("[NEED_MORE_HELP] unknown flow_state %r — reset", fs)
    return handle_need_more_help(
        state.model_copy(update={"flow_state": "start"}),
        customer_message,
        event,
    )
