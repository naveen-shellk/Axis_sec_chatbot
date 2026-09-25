"""
chatbot_web/entry/handler.py
------------------------------
Single entry point — routes every conversation turn to the correct flow.

No-auth flows (no sub_account_id needed):
  bank_query, how_to_trade, need_more_help, edit_profile

Auth-required flows (sub_account_id required):
  statement, ipo, account_details, brokerage, login_query, order_status

Auth gate:
  If an auth-required intent is detected and sub_account_id is absent from session
  → return status="auth_required" (Cisco/Simcomm shows auth prompt)
  → Assumption: for now, a test sub_account_id is injected via InternalMessageRequest
    so auth-required flows can be tested end-to-end without real auth.

Intent classification:
  All messages — including button taps — are classified by the Haiku LLM.
  Last 5 conversation turns are passed as context for accurate mid-flow routing.
  Confidence threshold: 0.60 — below this, falls back to main menu reprompt.
"""

from __future__ import annotations

import logging
import os

from src.core.conversation import run_conversation_turn
from src.core.llm import call_intent_llm
from src.core.session_store import get_or_create_session, save_session, clear_session

# No-auth flows
from src.flows.bank_query.handler    import handle_bank_query
from src.flows.how_to_trade.handler  import handle_how_to_trade
from src.flows.need_more_help.handler import handle_need_more_help
from src.flows.edit_profile.handler  import handle_edit_profile

# Auth-required flows
from src.flows.statement.handler       import handle_statement
from src.flows.ipo.handler             import handle_ipo
from src.flows.account_details.handler import handle_account_details
from src.flows.brokerage.handler       import handle_brokerage
from src.flows.login_query.handler     import handle_login_query
from src.flows.order_status.handler    import handle_order_status

# Intent-only flows (not in main menu — triggered by LLM classification only)
from src.flows.closure.handler         import handle_closure

from src.shared.session_end import _MAIN_MENU_OPTIONS
from models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

_TEST_SUB_ACCOUNT_ID = os.getenv("TEST_SUB_ACCOUNT_ID", "7032318")  # used when no real auth

# ── Greeting ──────────────────────────────────────────────────────────────────
_GREETING_MSG = (
    "👋 Welcome to Axis Direct! I'm your virtual assistant.\n\n"
    "How can I help you today? Please choose one of the options below "
    "or type your question."
)

_FULL_MENU = [
    # No-auth
    "Bank Query", "How To Trade", "Need More Help", "Edit Profile",
    # Auth-required
    "Statement", "IPO", "Account Details", "Brokerage and Charges",
    "Login Query", "Order Status",
]

# ── Flow maps ─────────────────────────────────────────────────────────────────

_NO_AUTH_FLOWS = {
    "bank_query":     handle_bank_query,
    "how_to_trade":   handle_how_to_trade,
    "need_more_help": handle_need_more_help,
    "edit_profile":   handle_edit_profile,
}

_AUTH_FLOWS = {
    "statement":       handle_statement,
    "ipo":             handle_ipo,
    "account_details": handle_account_details,
    "brokerage":       handle_brokerage,
    "login_query":     handle_login_query,
    "order_status":    handle_order_status,
    "closure":         handle_closure,   # intent-only — not in main menu
}

# ── Auth required response ────────────────────────────────────────────────────
_AUTH_REQUIRED_MSG = (
    "To access this feature, you need to verify your identity first.\n\n"
    "Please complete the authentication process to continue.\n"
    "(Authentication will be available shortly.)"
)

# ── Intent classification ─────────────────────────────────────────────────────
_INTENT_SYS = """\
You are a routing assistant for the Axis Direct web chatbot.

Classify the customer's message into ONE OR MORE intents.
Use the last few conversation turns for context — the customer may be
mid-flow or referring back to something said earlier.

Intents:
  greeting         — customer is greeting or opening the conversation ("hi", "hello", "good morning", "hey", "good evening", "hii" etc.)
  bank_query       — query about Axis Bank products (loan, credit card, savings account, branch)
  how_to_trade     — wants to learn how to place a trade or use the trading platform
  need_more_help   — EXPLICITLY wants a live agent or human support ("connect me to agent", "talk to someone", "I need a human")
  edit_profile     — wants to update profile details (email, mobile, address)
  statement        — wants a statement or report (tax, ledger, contract notes etc.)
  ipo              — wants to apply for IPO or check IPO status
  account_details  — wants to see their account info (demat number, trading ID etc.)
  brokerage        — asks about brokerage charges, DP charges, AMC etc.
  login_query      — can't login, forgot password, FTL, account locked
  order_status     — wants to check order/trade status or history
  closure          — wants to close their demat/trading account ("close my account", "account closure", "I want to close")
  escalate_to_human — query is clearly related to Axis Direct / Axis Securities but is too complex, sensitive, or specific for the bot to handle — e.g. fraud complaints, unauthorized transactions, regulatory grievances, complaints against the company, account disputes, margin call disputes, queries about specific corporate actions. The bot cannot resolve this and a human agent is needed.
  unknown          — general questions (support hours, contact info, anything not clearly one of the above)

IMPORTANT:
- "greeting" is ONLY for pure greetings with no other intent embedded ("hi", "hello", "good morning", "hey there").
- If a greeting contains an intent ("hi, I want my statement") → classify as the embedded intent, not greeting.
- "need_more_help" is ONLY for customers explicitly requesting a live agent or human.
- "escalate_to_human" is for Axis Direct-related queries that are too advanced/complex/sensitive for the bot.
- DO NOT use "escalate_to_human" for queries unrelated to Axis Direct — those are "unknown".
- Questions like "what are your support hours?", "how can I contact you?" → unknown
- Queries completely unrelated to Axis Direct → unknown
- Multi-intent: if the message clearly asks about MORE THAN ONE topic, list all detected intents (max 3).
- Single-intent: if the message is about one topic, return a single-element array.
- When in doubt → ["unknown"]

Return valid JSON only — no markdown fences:
{
  "intents":        ["<intent1>", "<intent2>"],
  "confidence":     <0.0 to 1.0>,
  "reasoning":      "<one sentence>",
  "sub_account_id": "<numeric ID if present in message, else null>"
}

NOTE on sub_account_id:
- Extract ONLY if a numeric ID (typically 5–10 digits) appears in the message that looks like a trading/customer account ID
- Examples: "my id is 7032318", "account 4433009", "Statement 7032318" → extract the number
- If no such ID is present → null
"""

_THRESHOLD = 0.60
_NO_MATCH_MSG = (
    "I'm here to help with your Axis Direct account! "
    "For support, you can reach us Monday–Friday, 9:00 AM – 6:00 PM IST.\n\n"
    "Please choose what you need help with:"
)

# ── Greeting system prompt (Qwen) ─────────────────────────────────────────────
_GREETING_SYS = """\
You are a warm, professional virtual assistant for Axis Direct (Axis Securities Limited),
embedded in the customer's website via a live chat widget.

The customer has just greeted you. Respond with:
1. A warm, brief greeting back — match their tone (e.g. if they say "Good morning", say it back).
2. Introduce yourself as the Axis Direct virtual assistant in one short sentence.
3. Ask how you can help and mention the available options.

The available options are:
  Bank Query | How To Trade | Need More Help | Edit Profile |
  Statement | IPO | Account Details | Brokerage and Charges | Login Query | Order Status

Rules:
- Keep it to 2–3 sentences maximum. Do not be verbose.
- Do not ask for sensitive information.
- Return ONLY valid JSON, no markdown fences:
{
  "message": "<warm greeting + intro + how can I help>",
  "quick_replies": [],
  "flow_action": "show_menu",
  "reasoning": ""
}
"""


# ── Public entry point ────────────────────────────────────────────────────────

def handle_message(
    conversation_id: str,
    raw_input: str,
    input_type: str = "free_text",
    sub_account_id: str | None = None,
    event: str = "Incoming message",
) -> InternalMessageResponse:
    """
    Route one conversation turn to the correct flow handler.
    Every message — including the first — goes directly to intent resolution.
    No automatic greeting is sent.
    """
    state = get_or_create_session(conversation_id)

    # Inject sub_account_id if provided (from auth or test)
    if sub_account_id and state.sub_account_id != sub_account_id:
        state = state.model_copy(update={"sub_account_id": sub_account_id})
        save_session(conversation_id, state)

    # ── Fully-agentic mode (AGENTIC_MODE=true) ────────────────────────────────
    # When enabled, the LLM agent decides which tools to call and in what order.
    # The deterministic flows below are bypassed. Auth is still enforced: an
    # account-specific request without a sub_account_id is asked for inline.
    if os.getenv("AGENTIC_MODE", "false").lower() == "true":
        return _handle_agentic(state, raw_input, conversation_id)

    # ── Global "End Chat" — works from ANY flow state ─────────────────────────
    # A customer can end the conversation at any point (e.g. mid-flow at the FY
    # picker), not only from the session_end node. Scoped to explicit end
    # phrases so it never hijacks legitimate flow input.
    _GLOBAL_END_PHRASES = {"end chat", "endchat", "end the chat", "end conversation"}
    if raw_input.strip().lower() in _GLOBAL_END_PHRASES:
        clear_session(conversation_id)
        logger.info("[ENTRY] conv=%s → global end chat", conversation_id)
        return InternalMessageResponse(
            reply_message=(
                "Thank you for reaching out to Axis Direct! "
                "Have a great day. If you need help again, we're always here."
            ),
            quick_reply_options=[],
            flow_state="ended",
            status="end",
            eventid="1001",
        )

    # ── Awaiting sub_account_id (inline ID collection) ────────────────────────
    if state.flow == "awaiting_sub_account_id":
        import re as _re
        # Extract numeric ID from customer's reply
        match = _re.search(r'\b(\d{5,10})\b', raw_input)
        if match:
            extracted_id = match.group(1)
            pending_intent = state.collected_data.get("pending_intent")
            logger.info("[ENTRY] conv=%s sub_account_id=%s provided, resuming intent=%r",
                        conversation_id, extracted_id, pending_intent)
            # Set sub_account_id and clear awaiting state
            new_state = state.model_copy(update={
                "sub_account_id": extracted_id,
                "flow":           pending_intent,
                "flow_state":     "start",
                "collected_data": {
                    k: v for k, v in state.collected_data.items()
                    if k != "pending_intent"
                },
            })
            save_session(conversation_id, new_state)
            # Continue to flow dispatch with updated state
            state = new_state
        else:
            # Couldn't extract ID — ask again
            _ASK_AGAIN = (
                "I couldn't find a valid Sub-Account ID in your message.\n\n"
                "Please enter your numeric Sub-Account ID:"
            )
            hist = state.history + [
                {"role": "user",      "content": raw_input},
                {"role": "assistant", "content": _ASK_AGAIN},
            ]
            save_session(conversation_id, state.model_copy(update={"history": hist}))
            return InternalMessageResponse(
                reply_message=_ASK_AGAIN,
                quick_reply_options=[],
                flow_state="awaiting_sub_account_id",
                status="ok",
                eventid="1001",
            )

    # ── Active flow continuation ───────────────────────────────────────────────
    if state.flow in _NO_AUTH_FLOWS:
        # Pass event to need_more_help for no-response detection
        if state.flow == "need_more_help":
            resp, new_state = handle_need_more_help(state, raw_input, event)
        else:
            resp, new_state = _NO_AUTH_FLOWS[state.flow](state, raw_input)
        if resp.status == "route_to_entry" and not resp.reply_message:
            # Check for pending intents from a multi-intent session
            pending = new_state.collected_data.get("pending_intents", [])
            if pending:
                return _dispatch_pending_intent(new_state, raw_input, pending)
            return _resolve_and_dispatch(new_state, raw_input, input_type)
        return resp
    if state.flow in _AUTH_FLOWS:
        if not state.sub_account_id:
            return _auth_required_response()
        resp, new_state = _AUTH_FLOWS[state.flow](state, raw_input)
        if resp.status == "route_to_entry" and not resp.reply_message:
            # Check for pending intents from a multi-intent session
            pending = new_state.collected_data.get("pending_intents", [])
            if pending:
                return _dispatch_pending_intent(new_state, raw_input, pending)
            return _resolve_and_dispatch(new_state, raw_input, input_type)
        return resp

    # ── At main menu — resolve intent ─────────────────────────────────────────
    return _resolve_and_dispatch(state, raw_input, input_type)


def _resolve_and_dispatch(state: SessionState, raw_input: str, input_type: str) -> InternalMessageResponse:
    """Classify intent via LLM (always), gate auth-required flows, dispatch."""

    # ── Exact button-label fast-path (bypasses LLM for known button taps) ────
    # These are the exact quick-reply labels shown in the UI — no ambiguity.
    _BUTTON_EXACT: dict[str, str] = {
        "need more help":        "need_more_help",
        "bank query":            "bank_query",
        "how to trade":          "how_to_trade",
        "edit profile":          "edit_profile",
        "statement":             "statement",
        "ipo":                   "ipo",
        "account details":       "account_details",
        "brokerage and charges": "brokerage",
        "brokerage":             "brokerage",
        "login query":           "login_query",
        "order status":          "order_status",
    }
    raw_lower  = raw_input.strip().lower()
    fast_intent = _BUTTON_EXACT.get(raw_lower)

    # default — overridden by LLM path if multi-intent is detected
    is_multi = False
    intents  = []

    # ── Greeting fast-path (skip Haiku for obvious greetings) ────────────────
    # These are unambiguous — no need to spend a Haiku call to classify them.
    _GREETING_WORDS = {
        "hi", "hello", "hey", "hii", "hiii", "helo", "hai",
        "good morning", "good afternoon", "good evening", "good night",
        "howdy", "greetings", "sup", "yo",
    }
    if not fast_intent and raw_lower in _GREETING_WORDS:
        fast_intent = "greeting"
        logger.info("[ENTRY] conv=%s greeting fast-path %r", state.conversation_id, raw_lower)

    if fast_intent:
        logger.info("[ENTRY] conv=%s button match %r → %s", state.conversation_id, raw_lower, fast_intent)
        intent   = fast_intent
        intents  = [fast_intent]
        is_multi = False
    else:
        # ── LLM classification for free-text ─────────────────────────────────
        # Intent classification only needs last 3 messages for context — 
        # reducing from 6 saves ~0.5s on Haiku call.
        recent_history = state.history[-3:]   # last 3 messages = ~1-2 turns
        messages = []
        for h in recent_history:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": [{"text": h["content"]}]})

        # Bedrock Converse requires messages to start with a user role.
        while messages and messages[0]["role"] != "user":
            messages.pop(0)

        messages.append({"role": "user", "content": [{"text": raw_input}]})

        import time as _t
        _t0 = _t.perf_counter()
        result     = call_intent_llm(_INTENT_SYS, messages)
        logger.info("[TIMING] intent_classify_ms=%d model=%s in=%d out=%d",
                    int((_t.perf_counter() - _t0) * 1000),
                    result.get("model_id", "?"),
                    result.get("input_tokens", 0), result.get("output_tokens", 0))
        parsed     = result.get("parsed") or {}

        # Support both old single-intent {"intent": ...} and new multi-intent {"intents": [...]}
        raw_intents = parsed.get("intents") or []
        if not raw_intents:
            # fallback: single intent field
            single = parsed.get("intent", "unknown")
            raw_intents = [single] if single else ["unknown"]

        confidence = float(parsed.get("confidence", 0.0))
        reasoning  = parsed.get("reasoning", "")

        # Accumulate intent LLM tokens into the turn accumulator — split by model
        from src.core.conversation import _turn_tokens as _tt
        _tt["input_tokens"]              += result.get("input_tokens",  0)
        _tt["output_tokens"]             += result.get("output_tokens", 0)
        _tt["llm_call_count"]            += 1
        _tt["intent_input_tokens"]       += result.get("input_tokens",  0)
        _tt["intent_output_tokens"]      += result.get("output_tokens", 0)

        # Extract sub_account_id if provided in the message
        extracted_sub_id = parsed.get("sub_account_id")
        if extracted_sub_id and str(extracted_sub_id).strip().isdigit():
            extracted_sub_id = str(extracted_sub_id).strip()
            if state.sub_account_id != extracted_sub_id:
                state = state.model_copy(update={"sub_account_id": extracted_sub_id})
                save_session(state.conversation_id, state)
                logger.info("[ENTRY] conv=%s sub_account_id extracted from message: %s",
                            state.conversation_id, extracted_sub_id)

        # Filter valid intents above confidence threshold
        _valid_set = {*_NO_AUTH_FLOWS, *_AUTH_FLOWS, "greeting", "escalate_to_human", "unknown"}
        if confidence < _THRESHOLD:
            intents = ["unknown"]
        else:
            intents = [i for i in raw_intents if i in _valid_set] or ["unknown"]

        # Deduplicate while preserving order
        seen = set()
        intents = [i for i in intents if not (i in seen or seen.add(i))]

        intent = intents[0]  # primary intent for single-intent path
        is_multi = len(intents) > 1

        logger.info(
            "[ENTRY] conv=%s intents=%r confidence=%.2f reasoning=%r multi=%s",
            state.conversation_id, intents, confidence, reasoning, is_multi,
        )

        if confidence < _THRESHOLD or intent not in (*_NO_AUTH_FLOWS, *_AUTH_FLOWS, "greeting", "escalate_to_human"):
            if intent not in ("greeting", "escalate_to_human"):
                intent = "unknown"
                intents = ["unknown"]
                is_multi = False

    # ── Auth gate for requires-login flows ───────────────────────────────────
    # If sub_account_id is missing, ask the customer to provide it inline.
    # No auth system yet — customer can pass their ID directly in the message.
    if intent in _AUTH_FLOWS and not state.sub_account_id:
        _ASK_ID_MSG = (
            "To access this feature I'll need your Axis Direct Sub-Account ID.\n\n"
            "Please share your Sub-Account ID and I'll continue with your request."
        )
        # Store the pending intent so we can resume once they provide the ID
        pending_flow_state = state.model_copy(update={
            "collected_data": {
                **state.collected_data,
                "pending_intent": intent,
            },
            "flow":       "awaiting_sub_account_id",
            "flow_state": "awaiting_sub_account_id",
        })
        save_session(state.conversation_id, pending_flow_state)
        hist = state.history + [
            {"role": "user",      "content": raw_input},
            {"role": "assistant", "content": _ASK_ID_MSG},
        ]
        save_session(state.conversation_id, pending_flow_state.model_copy(update={"history": hist}))
        logger.info("[ENTRY] conv=%s requires-login flow %r but no sub_account_id — asking inline",
                    state.conversation_id, intent)
        return InternalMessageResponse(
            reply_message=_ASK_ID_MSG,
            quick_reply_options=[],
            flow_state="awaiting_sub_account_id",
            status="ok",
            eventid="1001",
        )

    # ── Multi-intent handling ────────────────────────────────────────────────
    # Single-shot flows — complete in one step, no follow-up needed.
    # These are always handled FIRST in multi-intent so the customer gets
    # immediate value before being asked questions for multi-step flows.
    _SINGLE_SHOT_FLOWS = {
        "bank_query",       # static redirect — 0 API calls
        "edit_profile",     # static deeplink — 0 API calls
        "ipo",              # 1 API call, 1 Qwen call, done
        "account_details",  # 1 API call, 1 Qwen call, done
        "login_query",      # 1 API call, 1 Qwen call, done
    }

    if is_multi:
        # Separate single-shot and multi-step intents
        single_shot = [i for i in intents if i in _SINGLE_SHOT_FLOWS]
        multi_step  = [i for i in intents if i in {**_NO_AUTH_FLOWS, **_AUTH_FLOWS} and i not in _SINGLE_SHOT_FLOWS]

        # Auth gate: if any requires-login flow is in the list and no sub_account_id → ask inline
        if any(i in _AUTH_FLOWS for i in intents) and not state.sub_account_id:
            logger.info("[ENTRY] conv=%s multi-intent has requires-login flow but no sub_account_id — asking inline",
                        state.conversation_id)
            # Store all the intents as pending so we can resume after ID is provided
            pending_flow_state = state.model_copy(update={
                "collected_data": {
                    **state.collected_data,
                    "pending_intent": intents[0],
                    "pending_intents": intents[1:],
                },
                "flow":       "awaiting_sub_account_id",
                "flow_state": "awaiting_sub_account_id",
            })
            _ASK_ID_MSG = (
                "To access this feature I'll need your Axis Direct Sub-Account ID.\n\n"
                "Please share your Sub-Account ID and I'll continue with your request."
            )
            hist = state.history + [
                {"role": "user",      "content": raw_input},
                {"role": "assistant", "content": _ASK_ID_MSG},
            ]
            save_session(state.conversation_id, pending_flow_state.model_copy(update={"history": hist}))
            return InternalMessageResponse(
                reply_message=_ASK_ID_MSG,
                quick_reply_options=[],
                flow_state="awaiting_sub_account_id",
                status="ok",
                eventid="1001",
            )

        logger.info(
            "[ENTRY] conv=%s multi-intent single_shot=%r multi_step=%r",
            state.conversation_id, single_shot, multi_step,
        )

        all_flows = {**_NO_AUTH_FLOWS, **_AUTH_FLOWS}
        combined_parts: list[str] = []
        combined_quick_replies: list[str] = []
        updated_history = state.history + [{"role": "user", "content": raw_input}]

        # Run all single-shot flows and collect responses
        for ss_intent in single_shot:
            ss_state = state.model_copy(update={
                "flow": ss_intent, "flow_state": "start", "history": updated_history,
            })
            ss_resp, ss_state_out = all_flows[ss_intent](ss_state, raw_input)
            combined_parts.append(ss_resp.reply_message)
            updated_history = ss_state_out.history

        # For multi-step flows: handle the first one now, queue the rest
        pending = list(multi_step)
        primary_intent = None
        if pending:
            primary_intent = pending[0]
            pending_queue  = pending[1:]  # queue the rest

            primary_state = state.model_copy(update={
                "flow":          primary_intent,
                "flow_state":    "start",
                "history":       updated_history,
                "sub_account_id": state.sub_account_id,  # always preserve
                "collected_data": {
                    **state.collected_data,
                    "pending_intents": pending_queue,
                },
            })
            save_session(state.conversation_id, primary_state)
            primary_resp, primary_state_out = all_flows[primary_intent](primary_state, raw_input)

            if combined_parts:
                # Prepend single-shot responses before multi-step
                combined_parts.append(primary_resp.reply_message)
                combined_msg = "\n\n---\n\n".join(combined_parts)
                combined_quick_replies = primary_resp.quick_reply_options

                if pending_queue:
                    combined_quick_replies = list(combined_quick_replies)  # copy
                    combined_msg += f"\n\n_(I'll also help with {', '.join(pending_queue)} after this.)_"

                final_hist = primary_state_out.history[:-1] + [
                    {"role": "assistant", "content": combined_msg}
                ]
                # Preserve sub_account_id from original state in case flow handlers didn't carry it
                final_state = primary_state_out.model_copy(update={
                    "history": final_hist,
                    "sub_account_id": primary_state_out.sub_account_id or state.sub_account_id,
                })
                save_session(state.conversation_id, final_state)
                return InternalMessageResponse(
                    reply_message=combined_msg,
                    quick_reply_options=combined_quick_replies,
                    flow_state=primary_resp.flow_state,
                    status=primary_resp.status,
                    eventid=primary_resp.eventid,
                )
            else:
                # Only multi-step, no single-shot to prepend
                if pending_queue:
                    # Inform customer about the pending intents
                    pending_note = f"\n\n_(I'll also help with {', '.join(i.replace('_', ' ').title() for i in pending_queue)} after this.)_"
                    modified_reply = primary_resp.reply_message + pending_note
                    return InternalMessageResponse(
                        reply_message=modified_reply,
                        quick_reply_options=primary_resp.quick_reply_options,
                        flow_state=primary_resp.flow_state,
                        status=primary_resp.status,
                        eventid=primary_resp.eventid,
                    )
                return primary_resp

        # Only single-shot intents (no multi-step)
        if combined_parts:
            combined_msg = "\n\n---\n\n".join(combined_parts)
            final_hist = updated_history + [{"role": "assistant", "content": combined_msg}]
            save_session(state.conversation_id, state.model_copy(update={
                "flow": None, "flow_state": "session_end_response", "history": final_hist,
            }))
            return InternalMessageResponse(
                reply_message=combined_msg,
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state="session_end_response",
                status="ok",
            )

    # ── Single-intent dispatch ───────────────────────────────────────────────
    all_flows = {**_NO_AUTH_FLOWS, **_AUTH_FLOWS}
    if intent in all_flows:
        new_state = state.model_copy(update={"flow": intent, "flow_state": "start"})
        save_session(state.conversation_id, new_state)
        import time as _t
        _t0 = _t.perf_counter()
        resp, _ = all_flows[intent](new_state, raw_input)
        logger.info("[TIMING] flow_dispatch_ms=%d flow=%s (API + any flow LLM)",
                    int((_t.perf_counter() - _t0) * 1000), intent)
        return resp

    # ── Greeting — static message + menu (no LLM call) ───────────────────────
    # The greeting is fixed and the menu buttons carry the value, so there's no
    # need to spend an LLM round-trip (~2s) generating a "personalised" hello.
    if intent == "greeting":
        greeting_reply = _GREETING_MSG
        hist = state.history + [
            {"role": "user",      "content": raw_input},
            {"role": "assistant", "content": greeting_reply},
        ]
        save_session(state.conversation_id, state.model_copy(update={"history": hist}))
        logger.info("[ENTRY] conv=%s greeting → static response", state.conversation_id)
        return InternalMessageResponse(
            reply_message=greeting_reply,
            quick_reply_options=_FULL_MENU,
            flow_state="main_menu",
            status="ok",
        )

    # ── Escalate to human — Axis-related query too complex for bot ──────────
    # Direct escalation — no confirmation prompt. Bot determined it cannot
    # handle the query. Within hours → eventid 1002. Outside hours → ticket.
    if intent == "escalate_to_human":
        from src.flows.need_more_help.handler import _is_business_hours
        if _is_business_hours():
            _ESCALATE_MSG = (
                "I understand your query requires specialised assistance that goes beyond "
                "what I'm able to help with right now.\n\n"
                "Let me connect you to one of our customer service representatives "
                "who will be able to assist you further. Please hold on."
            )
            hist = state.history + [
                {"role": "user",      "content": raw_input},
                {"role": "assistant", "content": _ESCALATE_MSG},
            ]
            save_session(state.conversation_id, state.model_copy(update={
                "escalate":   True,
                "flow_state": "escalated",
                "history":    hist,
            }))
            logger.info("[ENTRY] conv=%s escalate_to_human (within hours) → eventid 1002",
                        state.conversation_id)
            return InternalMessageResponse(
                reply_message=_ESCALATE_MSG,
                quick_reply_options=[],
                flow_state="escalated",
                status="escalate",
                eventid="1002",
            )
        else:
            # Outside hours — show ticket link
            import os as _os
            _ticket = _os.getenv(
                "SUPPORT_TICKET_URL",
                "https://simplehai.axisdirect.in/portal/index.php/supportPortal/raise-query",
            )
            _OOH_MSG = (
                "Our live agents are available Monday to Friday, 9:00 AM – 6:00 PM IST.\n\n"
                "For your query, please raise a support ticket and our team will get back to you:\n"
                f"🎫 Create ticket: {_ticket}\n"
            )
            hist = state.history + [
                {"role": "user",      "content": raw_input},
                {"role": "assistant", "content": _OOH_MSG},
            ]
            save_session(state.conversation_id, state.model_copy(update={"history": hist}))
            logger.info("[ENTRY] conv=%s escalate_to_human (out of hours) → ticket",
                        state.conversation_id)
            return InternalMessageResponse(
                reply_message=_OOH_MSG,
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state="main_menu",
                status="ok",
            )

    # ── Unknown — route through need_more_help (confirmation + biz hours) ────
    # Customer sent something Axis-related but unclassifiable — ask if they
    # want a live agent (with business hours check and timestamp timeout).
    if intent == "unknown":
        logger.info("[ENTRY] conv=%s unknown intent → routing to need_more_help flow",
                    state.conversation_id)
        new_state = state.model_copy(update={
            "flow":       "need_more_help",
            "flow_state": "start",
        })
        save_session(state.conversation_id, new_state)
        resp, _ = handle_need_more_help(new_state, raw_input)
        return resp


def _handle_agentic(state: SessionState, raw_input: str, conversation_id: str) -> InternalMessageResponse:
    """
    Fully-agentic turn: the LLM agent decides which tools to call and when.
    Bypasses the deterministic state machines. Auth is still enforced in code —
    if an account-specific request has no sub_account_id, we ask for it inline
    (reusing the same awaiting_sub_account_id mechanism).
    """
    from src.core.strands_agent import run_agent_turn

    # Build a context block for the agent (recent history + known account id).
    recent = state.history[-6:]
    convo  = "\n".join(
        f"{'Customer' if h.get('role')=='user' else 'Assistant'}: {h.get('content','')}"
        for h in recent if h.get("content")
    )
    sub_line = (
        f"Customer Sub-Account ID: {state.sub_account_id}"
        if state.sub_account_id else
        "Customer Sub-Account ID: (not provided — if you need it for an account "
        "action, ask the customer to share their numeric Sub-Account ID)"
    )
    prompt = (
        f"{sub_line}\n\n"
        f"Conversation so far:\n{convo or '(none)'}\n\n"
        f"Customer's latest message: {raw_input}\n\n"
        f"Decide what to do (call tools as needed) and reply to the customer."
    )

    result   = run_agent_turn(prompt)
    message  = result.get("message") or "I'm sorry, I couldn't process that. Please try again."
    escalate = bool(result.get("escalate"))

    hist = state.history + [
        {"role": "user",      "content": raw_input},
        {"role": "assistant", "content": message},
    ]
    save_session(conversation_id, state.model_copy(update={"history": hist}))
    logger.info("[ENTRY:agentic] conv=%s escalate=%s reply_len=%d",
                conversation_id, escalate, len(message))

    if escalate:
        return InternalMessageResponse(
            reply_message=message,
            quick_reply_options=[],
            flow_state="escalated",
            status="escalate",
            eventid="1002",
        )
    return InternalMessageResponse(
        reply_message=message,
        quick_reply_options=[],
        flow_state="agentic",
        status="ok",
        eventid="1001",
    )


def _auth_required_response() -> InternalMessageResponse:
    """Return auth_required signal when an auth-required flow is requested without auth."""
    return InternalMessageResponse(
        reply_message=_AUTH_REQUIRED_MSG,
        quick_reply_options=[],
        flow_state="auth_required",
        status="auth_required",
        eventid="1001",
    )


def _dispatch_pending_intent(
    state: SessionState,
    raw_input: str,
    pending: list[str],
) -> InternalMessageResponse:
    """
    Dispatch the next pending intent from a multi-intent session.
    Called when the current flow completes (route_to_entry) and there are
    queued intents remaining from the original multi-intent message.
    """
    all_flows = {**_NO_AUTH_FLOWS, **_AUTH_FLOWS}
    next_intent = pending[0]
    remaining   = pending[1:]

    logger.info(
        "[ENTRY] conv=%s pending_intents: dispatching %r, remaining=%r",
        state.conversation_id, next_intent, remaining,
    )

    new_state = state.model_copy(update={
        "flow":       next_intent,
        "flow_state": "start",
        "collected_data": {
            **{k: v for k, v in state.collected_data.items() if k != "pending_intents"},
            "pending_intents": remaining,
        },
    })
    save_session(state.conversation_id, new_state)

    if next_intent in all_flows:
        resp, _ = all_flows[next_intent](new_state, raw_input)
        # Prepend a transition message so the customer knows we're moving to the next topic
        transition = f"Now helping you with **{next_intent.replace('_', ' ').title()}**:\n\n"
        return InternalMessageResponse(
            reply_message=transition + resp.reply_message,
            quick_reply_options=resp.quick_reply_options,
            flow_state=resp.flow_state,
            status=resp.status,
            eventid=resp.eventid,
        )

    # Fallback: clear pending and go to main menu
    save_session(state.conversation_id, state.model_copy(update={
        "flow": None, "flow_state": "main_menu", "collected_data": {},
    }))
    return InternalMessageResponse(
        reply_message="How else can I help you?",
        quick_reply_options=_FULL_MENU,
        flow_state="main_menu",
        status="ok",
    )

