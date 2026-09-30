"""
chatbot_langchain/entry/handler.py
----------------------------------
Single entry point — HYBRID router.

Default (AGENTIC_MODE=false): deterministic flow state-machines handle every
turn — greeting menu, intent classification, and the per-flow step logic
(statement date pickers, closure confirmation, order segments, etc.) with their
static messages and quick replies. This matches the documented flowcharts.

Opt-in (AGENTIC_MODE=true): the whole turn is handed to the LLM agent
(_handle_agentic), which decides tools itself. Kept for experimentation.

Authentication (both modes): account-required flows are gated behind a secure
mock OTP phone flow. When such a flow is requested and the session is not yet
authenticated, we ask for the registered mobile number → send OTP → verify →
the backend resolves the Sub-Account ID from the number → the pending flow
resumes. General flows need no auth.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from src.core.llm import call_intent_llm
from src.core.session_store import get_or_create_session, save_session, clear_session
from src.auth.otp_service import send_otp, verify_otp

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

# Intent-only flow (not in main menu — LLM classification only)
from src.flows.closure.handler         import handle_closure

from models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

# ── Menus / quick-reply sets ──────────────────────────────────────────────────
_FULL_MENU = [
    "Bank Query", "How To Trade", "Need More Help", "Edit Profile",
    "Statement", "IPO", "Account Details", "Brokerage and Charges",
    "Login Query", "Order Status",
]
_FOLLOWUP_REPLIES = ["Main Menu", "End Chat"]

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

_GREETING_MSG = (
    "👋 Welcome to Axis Direct! I'm your virtual assistant.\n\n"
    "How can I help you today? Please choose one of the options below "
    "or type your question."
)

# Greeting words that show the full menu.
_GREETING_WORDS = {
    "hi", "hello", "hey", "hii", "hiii", "helo", "hai",
    "good morning", "good afternoon", "good evening", "good night",
    "howdy", "greetings", "sup", "yo", "start", "menu", "main menu",
}

# ── Intent classification (Haiku) ─────────────────────────────────────────────
_INTENT_SYS = """\
You are a routing assistant for the Axis Direct web chatbot.

Classify the customer's message into ONE OR MORE intents.
Use the last few conversation turns for context — the customer may be
mid-flow or referring back to something said earlier.

Intents:
  greeting         — pure greeting/opening ("hi", "hello", "good morning")
  bank_query       — query about Axis Bank products (loan, credit card, savings, branch)
  how_to_trade     — wants to learn how to place a trade / use the platform
  need_more_help   — EXPLICITLY wants a live agent/human ("connect me to agent", "talk to someone")
  edit_profile     — wants to update profile details (email, mobile, address)
  statement        — wants a statement or report (tax, ledger, contract notes etc.)
  ipo              — wants to apply for IPO or check IPO status
  account_details  — wants account info (demat number, trading ID etc.)
  brokerage        — asks about brokerage charges, DP charges, AMC etc.
  login_query      — can't login, forgot password, FTL, account locked
  order_status     — wants to check order/trade status or history
  closure          — wants to close their demat/trading account
  escalate_to_human — Axis Direct-related query too complex/sensitive for the bot (fraud, disputes, grievances)
  unknown          — general questions (support hours, contact info, anything not clearly above)

IMPORTANT:
- "greeting" is ONLY for pure greetings with no embedded intent.
- If a greeting contains an intent ("hi, I want my statement") → classify the embedded intent.
- "need_more_help" is ONLY for explicit live-agent/human requests.
- "escalate_to_human" is for Axis Direct queries too advanced for the bot; unrelated queries → "unknown".
- Multi-intent: if the message clearly asks about MORE THAN ONE topic, list all (max 3). Else single-element array.
- When in doubt → ["unknown"].

Return valid JSON only — no markdown fences:
{
  "intents":        ["<intent1>", "<intent2>"],
  "confidence":     <0.0 to 1.0>,
  "reasoning":      "<one sentence>",
  "sub_account_id": "<numeric ID if present in message, else null>"
}

NOTE on sub_account_id:
- Extract ONLY if a numeric ID (5–10 digits) that looks like a trading/customer account ID appears.
- If no such ID → null
"""

_THRESHOLD = 0.60

# Exact button-label fast-path (bypasses the intent LLM for known button taps).
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


# ── Public entry point ────────────────────────────────────────────────────────

def handle_message(
    conversation_id: str,
    raw_input: str,
    input_type: str = "free_text",
    sub_account_id: str | None = None,
    event: str = "Incoming message",
) -> InternalMessageResponse:
    """Route one conversation turn to the correct flow handler."""
    state = get_or_create_session(conversation_id)

    # Inject sub_account_id if explicitly provided (internal test path) — treated
    # as already-authenticated so auth flows can be exercised without OTP.
    if sub_account_id and state.sub_account_id != sub_account_id:
        state = state.model_copy(update={
            "sub_account_id":      sub_account_id,
            "authenticated":       True,
            "auth_sub_account_id": sub_account_id,
        })
        save_session(conversation_id, state)

    # ── Opt-in fully-agentic mode ─────────────────────────────────────────────
    if os.getenv("AGENTIC_MODE", "false").lower() == "true":
        return _handle_agentic(state, raw_input, conversation_id)

    # ── Global "End Chat" — works from ANY flow state ─────────────────────────
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

    # ── Awaiting phone (collect registered mobile → send OTP) ─────────────────
    if state.flow in ("awaiting_phone", "awaiting_sub_account_id"):
        import re as _re
        digits = "".join(ch for ch in raw_input if ch.isdigit())
        if len(digits) >= 10:
            phone = digits[-10:]
            send_otp(phone)
            logger.info("[ENTRY] conv=%s phone=***%s → OTP sent", conversation_id, phone[-4:])
            _ASK_OTP = (
                f"An OTP has been sent to your registered mobile number "
                f"ending in {phone[-4:]}.\n\n"
                "Please enter the 6-digit OTP to verify your identity."
            )
            otp_state = state.model_copy(update={
                "flow":       "awaiting_otp",
                "flow_state": "awaiting_otp",
                "collected_data": {**state.collected_data, "otp_phone": phone},
                "history": state.history + [
                    {"role": "user",      "content": raw_input},
                    {"role": "assistant", "content": _ASK_OTP},
                ],
            })
            save_session(conversation_id, otp_state)
            return InternalMessageResponse(
                reply_message=_ASK_OTP, quick_reply_options=[],
                flow_state="awaiting_otp", status="ok", eventid="1001",
            )
        _ASK_AGAIN = (
            "I couldn't find a valid mobile number in your message.\n\n"
            "Please enter your 10-digit registered mobile number:"
        )
        save_session(conversation_id, state.model_copy(update={
            "flow": "awaiting_phone", "flow_state": "awaiting_phone",
            "history": state.history + [
                {"role": "user", "content": raw_input},
                {"role": "assistant", "content": _ASK_AGAIN},
            ],
        }))
        return InternalMessageResponse(
            reply_message=_ASK_AGAIN, quick_reply_options=[],
            flow_state="awaiting_phone", status="ok", eventid="1001",
        )

    # ── Awaiting OTP (verify → resolve sub-account → resume PENDING flow) ──────
    if state.flow == "awaiting_otp":
        import re as _re
        otp_phone = state.collected_data.get("otp_phone")
        match = _re.search(r'\b(\d{4,8})\b', raw_input)
        entered_otp = match.group(1) if match else raw_input.strip()

        result = verify_otp(otp_phone or "", entered_otp)
        if result.get("verified"):
            resolved_sub_id = result.get("sub_account_id")
            pending_intent  = state.collected_data.get("pending_intent")
            # Other intents captured before auth (multi-intent case).
            queued_intents  = list(state.collected_data.get("pending_intents", []) or [])
            logger.info("[ENTRY] conv=%s OTP verified → sub_account_id=%s, resuming intent=%r queued=%r",
                        conversation_id, resolved_sub_id, pending_intent, queued_intents)
            new_state = state.model_copy(update={
                "authenticated":       True,
                "sub_account_id":      resolved_sub_id,
                "auth_sub_account_id": resolved_sub_id,
                "auth_phone":          otp_phone,
                "flow":                pending_intent,
                "flow_state":          "start",
                "collected_data": {
                    k: v for k, v in state.collected_data.items()
                    if k not in ("pending_intent", "pending_intents", "otp_phone")
                },
            })
            save_session(conversation_id, new_state)

            # ── Multi-intent resume: if OTP gated a multi-intent turn, run the
            # FULL set (resumed + queued) through the parallel path so both
            # flows' first steps fire together and merge into one reply.
            if queued_intents and pending_intent:
                all_flows = {**_NO_AUTH_FLOWS, **_AUTH_FLOWS}
                run_intents = [i for i in ([pending_intent] + queued_intents) if i in all_flows]
                seen = set()
                run_intents = [i for i in run_intents if not (i in seen or seen.add(i))]
                if len(run_intents) > 1:
                    _SINGLE_SHOT = {"bank_query", "edit_profile", "ipo", "account_details", "login_query"}
                    return _run_multi_intent_parallel(new_state, raw_input, run_intents,
                                                      all_flows, _SINGLE_SHOT)

            # Resume the single pending auth flow deterministically at its start step.
            if pending_intent in _AUTH_FLOWS:
                resp, _ = _AUTH_FLOWS[pending_intent](new_state, raw_input)
                return resp
            # No pending flow (e.g. authed proactively) → show the menu.
            return _greeting_response(new_state, raw_input, conversation_id)
        _OTP_INVALID = "OTP invalid. Please check the code and enter the 6-digit OTP again."
        save_session(conversation_id, state.model_copy(update={
            "history": state.history + [
                {"role": "user", "content": raw_input},
                {"role": "assistant", "content": _OTP_INVALID},
            ],
        }))
        logger.info("[ENTRY] conv=%s OTP verify failed", conversation_id)
        return InternalMessageResponse(
            reply_message=_OTP_INVALID, quick_reply_options=[],
            flow_state="awaiting_otp", status="ok", eventid="1001",
        )

    # ── Greeting → static welcome + full menu ─────────────────────────────────
    if raw_input.strip().lower() in _GREETING_WORDS:
        return _greeting_response(state, raw_input, conversation_id)

    # ── Active flow continuation ──────────────────────────────────────────────
    if state.flow in _NO_AUTH_FLOWS:
        if state.flow == "need_more_help":
            resp, new_state = handle_need_more_help(state, raw_input, event)
        else:
            resp, new_state = _NO_AUTH_FLOWS[state.flow](state, raw_input)
        if resp.status == "route_to_entry" and not resp.reply_message:
            pending = new_state.collected_data.get("pending_intents", [])
            if pending:
                return _dispatch_pending_intent(new_state, raw_input, pending)
            return _resolve_and_dispatch(new_state, raw_input, input_type)
        return resp
    if state.flow in _AUTH_FLOWS:
        if not state.authenticated:
            return _start_phone_auth(state, raw_input, conversation_id, pending_intent=state.flow)
        resp, new_state = _AUTH_FLOWS[state.flow](state, raw_input)
        if resp.status == "route_to_entry" and not resp.reply_message:
            pending = new_state.collected_data.get("pending_intents", [])
            if pending:
                return _dispatch_pending_intent(new_state, raw_input, pending)
            return _resolve_and_dispatch(new_state, raw_input, input_type)
        return resp

    # ── At main menu — classify intent + dispatch ─────────────────────────────
    return _resolve_and_dispatch(state, raw_input, input_type)


def _greeting_response(state, raw_input, conversation_id) -> InternalMessageResponse:
    save_session(conversation_id, state.model_copy(update={
        "flow": None, "flow_state": "greeting",
        "history": state.history + [
            {"role": "user",      "content": raw_input},
            {"role": "assistant", "content": _GREETING_MSG},
        ],
    }))
    logger.info("[ENTRY] conv=%s greeting → full menu", conversation_id)
    return InternalMessageResponse(
        reply_message=_GREETING_MSG, quick_reply_options=_FULL_MENU,
        flow_state="greeting", status="ok", eventid="1001",
    )


def _start_phone_auth(state, raw_input, conversation_id, pending_intent) -> InternalMessageResponse:
    """Begin the OTP phone flow, remembering which flow to resume after verify."""
    _ASK_PHONE_MSG = (
        "To access this feature I'll need to verify your identity.\n\n"
        "Please share your registered mobile number to receive an OTP."
    )
    save_session(conversation_id, state.model_copy(update={
        "flow":       "awaiting_phone",
        "flow_state": "awaiting_phone",
        "collected_data": {**state.collected_data, "pending_intent": pending_intent},
        "history": state.history + [
            {"role": "user",      "content": raw_input},
            {"role": "assistant", "content": _ASK_PHONE_MSG},
        ],
    }))
    logger.info("[ENTRY] conv=%s auth-required flow %r → phone flow",
                conversation_id, pending_intent)
    return InternalMessageResponse(
        reply_message=_ASK_PHONE_MSG, quick_reply_options=[],
        flow_state="awaiting_phone", status="ok", eventid="1001",
    )


def _resolve_and_dispatch(state: SessionState, raw_input: str, input_type: str) -> InternalMessageResponse:
    """Classify intent (button fast-path or Haiku), gate auth flows, dispatch."""
    raw_lower   = raw_input.strip().lower()
    fast_intent = _BUTTON_EXACT.get(raw_lower)
    is_multi = False
    intents  = []

    if fast_intent:
        logger.info("[ENTRY] conv=%s button match %r → %s", state.conversation_id, raw_lower, fast_intent)
        intent  = fast_intent
        intents = [fast_intent]
    else:
        # ── AGENT is the decision-maker (Option a) ────────────────────────────
        # The agent decides the intent/flow + control action for this turn (it
        # does NOT write the reply — the chosen deterministic flow emits the
        # hardcoded messages). Context-aware: recent history + auth + active flow.
        from src.core.langchain_agent import run_router_decision
        import time as _t
        _t0 = _t.perf_counter()
        result = run_router_decision(
            customer_message=raw_input,
            recent_history=state.history[-4:],
            authenticated=bool(state.authenticated),
            active_flow=state.flow,
        )
        logger.info("[TIMING] agent_router_ms=%d in=%d out=%d",
                    int((_t.perf_counter() - _t0) * 1000),
                    result.get("input_tokens", 0), result.get("output_tokens", 0))
        parsed = result  # run_router_decision already returns the parsed dict

        raw_intents = parsed.get("intents") or []
        if not raw_intents:
            single = parsed.get("intent", "unknown")
            raw_intents = [single] if single else ["unknown"]
        confidence = float(parsed.get("confidence", 0.0))
        reasoning  = parsed.get("reasoning", "")

        # Token accounting into the per-turn accumulator.
        try:
            from src.core.conversation import _turn_tokens as _tt
            _tt["input_tokens"]         += result.get("input_tokens",  0)
            _tt["output_tokens"]        += result.get("output_tokens", 0)
            _tt["llm_call_count"]       += 1
            _tt["intent_input_tokens"]  += result.get("input_tokens",  0)
            _tt["intent_output_tokens"] += result.get("output_tokens", 0)
        except Exception:
            pass

        # Extract sub_account_id if the message carried one (does NOT authenticate).
        extracted_sub_id = parsed.get("sub_account_id")
        if extracted_sub_id and str(extracted_sub_id).strip().isdigit():
            extracted_sub_id = str(extracted_sub_id).strip()
            if state.sub_account_id != extracted_sub_id:
                state = state.model_copy(update={"sub_account_id": extracted_sub_id})
                save_session(state.conversation_id, state)

        _valid_set = {*_NO_AUTH_FLOWS, *_AUTH_FLOWS, "greeting", "escalate_to_human", "unknown"}
        if confidence < _THRESHOLD:
            intents = ["unknown"]
        else:
            intents = [i for i in raw_intents if i in _valid_set] or ["unknown"]
        seen = set()
        intents = [i for i in intents if not (i in seen or seen.add(i))]
        intent = intents[0]
        is_multi = len(intents) > 1
        logger.info("[ENTRY] conv=%s intents=%r confidence=%.2f reasoning=%r multi=%s",
                    state.conversation_id, intents, confidence, reasoning, is_multi)

        if confidence < _THRESHOLD or intent not in (*_NO_AUTH_FLOWS, *_AUTH_FLOWS, "greeting", "escalate_to_human"):
            if intent not in ("greeting", "escalate_to_human"):
                intent = "unknown"; intents = ["unknown"]; is_multi = False

    # ── Auth gate: auth-required flow but not authenticated → phone flow ──────
    # (Single-intent only. For MULTI-intent, the block below handles auth so it
    # can stash ALL the other intents in pending_intents and resume them after
    # OTP — otherwise the extra intents would be silently dropped.)
    if not is_multi and intent in _AUTH_FLOWS and not state.authenticated:
        return _start_phone_auth(state, raw_input, state.conversation_id, pending_intent=intent)

    # ── Multi-intent (PARALLEL first-step execution) ──────────────────────────
    _SINGLE_SHOT_FLOWS = {"bank_query", "edit_profile", "ipo", "account_details", "login_query"}
    if is_multi:
        # If any auth flow is present and not authenticated → auth first.
        if any(i in _AUTH_FLOWS for i in intents) and not state.authenticated:
            first_auth = next(i for i in intents if i in _AUTH_FLOWS)
            st = state.model_copy(update={"collected_data": {
                **state.collected_data, "pending_intents": [i for i in intents if i != first_auth],
            }})
            return _start_phone_auth(st, raw_input, state.conversation_id, pending_intent=first_auth)

        all_flows = {**_NO_AUTH_FLOWS, **_AUTH_FLOWS}
        # Keep only real, dispatchable flow intents, preserving classifier order.
        run_intents = [i for i in intents if i in all_flows]
        if run_intents:
            return _run_multi_intent_parallel(state, raw_input, run_intents, all_flows,
                                              _SINGLE_SHOT_FLOWS)

    # ── Single-intent dispatch ────────────────────────────────────────────────
    all_flows = {**_NO_AUTH_FLOWS, **_AUTH_FLOWS}
    if intent in all_flows:
        new_state = state.model_copy(update={"flow": intent, "flow_state": "start"})
        save_session(state.conversation_id, new_state)
        import time as _t
        _t0 = _t.perf_counter()
        resp, _ = all_flows[intent](new_state, raw_input)
        logger.info("[TIMING] flow_dispatch_ms=%d flow=%s", int((_t.perf_counter() - _t0) * 1000), intent)
        return resp

    if intent == "greeting":
        return _greeting_response(state, raw_input, state.conversation_id)

    if intent == "escalate_to_human":
        new_state = state.model_copy(update={"flow": "need_more_help", "flow_state": "start"})
        save_session(state.conversation_id, new_state)
        resp, _ = handle_need_more_help(new_state, raw_input, "Incoming message")
        return resp

    # ── Unknown / out-of-scope → helpful fallback + main menu ─────────────────
    # A general or off-topic question ("what is 2+2", "hello there", small talk)
    # must NOT jump to live-agent escalation. Show a hardcoded scope message and
    # the main menu so the customer can pick a real service. (Explicit human
    # requests are handled above via need_more_help / escalate_to_human.)
    logger.info("[ENTRY] conv=%s unknown intent → scope fallback + menu", state.conversation_id)
    _UNKNOWN_MSG = (
        "I'm the Axis Direct assistant — I can help with statements, order status, "
        "account details, brokerage & charges, IPO, login help, closure, editing "
        "your profile, and how-to-trade queries.\n\n"
        "Please choose an option below, or rephrase your question."
    )
    save_session(state.conversation_id, state.model_copy(update={
        "flow": None, "flow_state": "main_menu",
        "history": state.history + [
            {"role": "user",      "content": raw_input},
            {"role": "assistant", "content": _UNKNOWN_MSG},
        ],
    }))
    return InternalMessageResponse(
        reply_message=_UNKNOWN_MSG, quick_reply_options=_FULL_MENU,
        flow_state="main_menu", status="ok", eventid="1001",
    )


def _run_multi_intent_parallel(
    state: SessionState,
    raw_input: str,
    run_intents: list[str],
    all_flows: dict,
    single_shot_flows: set,
) -> InternalMessageResponse:
    """
    Execute the FIRST step of every requested intent's flow CONCURRENTLY.

    Each flow runs on its OWN isolated copy of the session state (no shared
    mutation) inside a thread pool — so the first API/LLM hit of e.g. Statement
    and Brokerage fire in parallel rather than one-after-the-other. Results are
    then merged deterministically in the classifier's intent order.

    Turn ownership:
      - The first MULTI-STEP flow (needs follow-up: statement/order/brokerage/
        closure/how_to_trade/need_more_help) becomes the "primary": its
        quick-replies + flow_state drive the next customer input, and its state
        is persisted as the active session. Any further multi-step flows are
        queued (pending_intents) to resume on later turns.
      - SINGLE-SHOT flows (bank_query/edit_profile/ipo/account_details/
        login_query) complete in one step; their replies are merged inline.
      - If there is NO multi-step flow, all replies are merged and the turn ends.
    """
    base_history = state.history + [{"role": "user", "content": raw_input}]

    # Build an isolated starting state for each intent (shared read-only fields:
    # auth, sub_account_id, customer_profile — but a private flow/flow_state).
    def _make_state(intent: str) -> SessionState:
        return state.model_copy(update={
            "flow": intent, "flow_state": "start", "history": base_history,
        })

    # Fire every intent's first step in parallel. Preserve input order on read.
    results: dict[str, tuple] = {}

    def _run(intent: str):
        return intent, all_flows[intent](_make_state(intent), raw_input)

    with ThreadPoolExecutor(max_workers=min(len(run_intents), 4)) as pool:
        for intent, out in pool.map(_run, run_intents):
            results[intent] = out  # out = (InternalMessageResponse, SessionState)

    logger.info("[ENTRY] conv=%s multi-intent parallel ran %r",
                state.conversation_id, run_intents)

    # ── Classify each flow's FIRST-STEP result at runtime (#2 + #3) ───────────
    # TERMINAL   → the flow completed this turn and needs NO more input
    #              (its reply already carries the full answer/data). Detected by
    #              flow_state == "session_end_response" or status in end/escalate.
    # INTERACTIVE→ the flow asked a question and is waiting for the customer's
    #              next choice (statement/order/brokerage pickers, etc.).
    # Rationale: TERMINAL flows (account details, deactivated notices, one-shot
    # confirmations) can ALL be answered together in this single reply. Only
    # ONE INTERACTIVE flow can own the turn (its buttons), so the rest queue.
    def _is_terminal(resp) -> bool:
        return (resp.flow_state == "session_end_response"
                or resp.status in ("end", "escalate")
                or not resp.quick_reply_options and resp.flow_state in ("session_end_response", "ended"))

    terminal    = [i for i in run_intents if _is_terminal(results[i][0])]
    interactive = [i for i in run_intents if i not in terminal]
    primary_intent = interactive[0] if interactive else None
    pending_queue  = interactive[1:] if interactive else []

    # Merged reply shows every TERMINAL result (all done now) + the PRIMARY
    # interactive flow's question. Queued interactive flows are NOT shown (their
    # buttons can't render this turn) — only named in the deferral note.
    from src.core.langchain_agent import run_multi_intent_merge
    show_order = terminal + ([primary_intent] if primary_intent else [])
    show_order = [i for i in run_intents if i in show_order]  # preserve order
    sections = [
        {"intent": i, "message": results[i][0].reply_message}
        for i in show_order if results[i][0].reply_message
    ]
    combined_msg = run_multi_intent_merge(sections)

    if primary_intent:
        primary_resp, primary_state_out = results[primary_intent]
        if pending_queue:
            combined_msg += ("\n\n_(I'll also help with "
                             + ", ".join(i.replace('_', ' ').title() for i in pending_queue)
                             + " after this.)_")
        # Persist the primary (interactive) flow's state as the active session;
        # carry only the remaining INTERACTIVE flows in the queue.
        final_state = primary_state_out.model_copy(update={
            "history": primary_state_out.history[:-1] + [{"role": "assistant", "content": combined_msg}]
            if primary_state_out.history else base_history + [{"role": "assistant", "content": combined_msg}],
            "sub_account_id": primary_state_out.sub_account_id or state.sub_account_id,
            "collected_data": {**primary_state_out.collected_data, "pending_intents": pending_queue},
        })
        save_session(state.conversation_id, final_state)
        return InternalMessageResponse(
            reply_message=combined_msg,
            quick_reply_options=primary_resp.quick_reply_options,
            flow_state=primary_resp.flow_state,
            status=primary_resp.status,
            eventid=primary_resp.eventid,
        )

    # All intents were TERMINAL → every answer is in the merged reply; end turn.
    save_session(state.conversation_id, state.model_copy(update={
        "flow": None, "flow_state": "session_end_response",
        "history": base_history + [{"role": "assistant", "content": combined_msg}],
    }))
    return InternalMessageResponse(
        reply_message=combined_msg, quick_reply_options=_FOLLOWUP_REPLIES,
        flow_state="session_end_response", status="ok", eventid="1001",
    )


def _dispatch_pending_intent(state: SessionState, raw_input: str, pending: list[str]) -> InternalMessageResponse:
    """Dispatch the next queued intent from a multi-intent session."""
    all_flows = {**_NO_AUTH_FLOWS, **_AUTH_FLOWS}
    next_intent = pending[0]
    remaining   = pending[1:]
    logger.info("[ENTRY] conv=%s pending: dispatching %r remaining=%r",
                state.conversation_id, next_intent, remaining)
    new_state = state.model_copy(update={
        "flow": next_intent, "flow_state": "start",
        "collected_data": {
            **{k: v for k, v in state.collected_data.items() if k != "pending_intents"},
            "pending_intents": remaining,
        },
    })
    save_session(state.conversation_id, new_state)
    if next_intent in all_flows:
        resp, _ = all_flows[next_intent](new_state, raw_input)
        transition = f"Now helping you with **{next_intent.replace('_', ' ').title()}**:\n\n"
        return InternalMessageResponse(
            reply_message=transition + resp.reply_message, quick_reply_options=resp.quick_reply_options,
            flow_state=resp.flow_state, status=resp.status, eventid=resp.eventid,
        )
    save_session(state.conversation_id, state.model_copy(update={
        "flow": None, "flow_state": "main_menu", "collected_data": {},
    }))
    return InternalMessageResponse(
        reply_message="How else can I help you?", quick_reply_options=_FULL_MENU,
        flow_state="main_menu", status="ok",
    )


# ── Opt-in agentic path (AGENTIC_MODE=true) ───────────────────────────────────

def _handle_agentic(state: SessionState, raw_input: str, conversation_id: str) -> InternalMessageResponse:
    """Fully-agentic turn: the LLM agent decides which tools to call. Opt-in."""
    from src.core.langchain_agent import run_agent_turn

    recent = state.history[-6:]
    convo  = "\n".join(
        f"{'Customer' if h.get('role')=='user' else 'Assistant'}: {h.get('content','')}"
        for h in recent if h.get("content")
    )
    if state.authenticated and state.sub_account_id:
        sub_line = (f"AUTH STATUS: verified. Customer Sub-Account ID: {state.sub_account_id} "
                    f"(use this for account actions; never invent one).")
    else:
        sub_line = ("AUTH STATUS: NOT verified — no Sub-Account ID. For any account-specific "
                    "action, call request_authentication (do NOT ask for the number, do NOT call account tools).")
    prompt = (f"{sub_line}\n\nConversation so far:\n{convo or '(none)'}\n\n"
              f"Customer's latest message: {raw_input}\n\n"
              f"Decide what to do (call tools as needed) and reply to the customer.")

    result = run_agent_turn(prompt)
    if result.get("needs_auth"):
        st = state.model_copy(update={"collected_data": {**state.collected_data, "pending_request": raw_input}})
        return _start_phone_auth(st, raw_input, conversation_id, pending_intent=None)

    message  = result.get("message") or "I'm sorry, I couldn't process that. Please try again."
    escalate = bool(result.get("escalate"))
    qset = (result.get("quick_reply_set") or "").strip()
    _sets = {"main_menu": _FULL_MENU, "session_end": _FOLLOWUP_REPLIES}
    quick_replies = _sets.get(qset, list(_FOLLOWUP_REPLIES))

    save_session(conversation_id, state.model_copy(update={
        "history": state.history + [
            {"role": "user", "content": raw_input},
            {"role": "assistant", "content": message},
        ],
    }))
    if escalate:
        return InternalMessageResponse(
            reply_message=message, quick_reply_options=[],
            flow_state="escalated", status="escalate", eventid="1002",
        )
    return InternalMessageResponse(
        reply_message=message, quick_reply_options=quick_replies,
        flow_state="agentic", status="ok", eventid="1001",
    )
