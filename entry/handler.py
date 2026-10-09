"""
chatbot_langchain/entry/handler.py
----------------------------------
Single entry point — deterministic flow router.

Deterministic flow state-machines handle every turn — greeting menu, intent
classification, and the per-flow step logic (statement date pickers, closure
confirmation, order segments, etc.) with their static messages and quick
replies. This matches the documented flowcharts.

Authentication: account-required flows need a Client ID (Sub-Account ID), which
the client supplies in the request payload (handled in app/web routes → passed
as sub_account_id, which marks the session authenticated). We NEVER ask the
customer for it in chat. If an account flow is reached without a Client ID, we
show a sign-in notice + the main menu instead. General flows need no identity.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

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
# Pre-login (no Client ID) menu — only the flows that work WITHOUT an account
# identity. Shown when the customer is not authenticated so we never offer an
# option that would immediately be blocked by the auth gate.
_PRELOGIN_MENU = [
    "Bank Query", "How To Trade", "Edit Profile", "Need More Help",
]
_FOLLOWUP_REPLIES = ["Main Menu", "End Chat"]

# ── Complaint handling ────────────────────────────────────────────────────────
# When the router flags a genuine grievance/dispute (complaint=true) AND a
# serviceable flow handled the request, we still show the self-service data but
# append an offer to connect to a live agent. The customer can tap the option to
# enter the need_more_help escalation flow.
_COMPLAINT_OFFER = (
    "\n\nIf this still looks wrong, I can connect you to a live agent to "
    "raise a complaint."
)
_TALK_TO_AGENT = "Talk to a live agent"


def _append_complaint_offer(resp: "InternalMessageResponse", complaint: bool):
    """Append the live-agent offer + quick-reply to a serviceable flow's reply
    when the turn was flagged as a genuine complaint. No-op otherwise, and skips
    turns that are themselves already escalations/auth prompts."""
    if not complaint or resp is None:
        return resp
    if resp.status in ("escalate", "auth_required") or resp.eventid == "1002":
        return resp
    if _COMPLAINT_OFFER.strip()[:20] in (resp.reply_message or ""):
        return resp  # already appended
    opts = list(resp.quick_reply_options or [])
    if _TALK_TO_AGENT not in opts:
        opts = [_TALK_TO_AGENT] + opts
    return resp.model_copy(update={
        "reply_message": (resp.reply_message or "") + _COMPLAINT_OFFER,
        "quick_reply_options": opts,
    })

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

# Friendly, customer-facing labels for each intent (used in multi-intent
# deferral notes and queued-flow transitions — avoids raw names like
# "Order Status" when the request was specifically order history).
_INTENT_LABELS = {
    "statement":       "your statement request",
    "order_status":    "your order request",
    "brokerage":       "your charges request",
    "account_details": "your account details",
    "ipo":             "your IPO request",
    "login_query":     "your login query",
    "closure":         "your account closure request",
    "bank_query":      "your bank query",
    "edit_profile":    "your profile update",
    "how_to_trade":    "your how-to-trade query",
    "need_more_help":  "connecting you to an agent",
}


def _intent_label(intent: str) -> str:
    return _INTENT_LABELS.get(intent, intent.replace("_", " ").title())


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
    "talk to a live agent":  "need_more_help",   # complaint-offer quick reply
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

    # ── Global "Talk to a live agent" — works from ANY flow state ─────────────
    # This is the quick-reply shown with the complaint offer. It must escalate
    # regardless of the active flow, so it short-circuits into need_more_help
    # before the active-flow continuation consumes it as flow input.
    if raw_input.strip().lower() in {"talk to a live agent", "talk to live agent"}:
        new_state = state.model_copy(update={"flow": "need_more_help", "flow_state": "start"})
        save_session(conversation_id, new_state)
        logger.info("[ENTRY] conv=%s → global 'talk to a live agent' → need_more_help", conversation_id)
        resp, _ = handle_need_more_help(new_state, raw_input, "Incoming message")
        return resp

    # ── Global "Go back to main menu" — works from ANY state ──────────────────
    # The navigation quick-reply shown after FAQ answers and flow endings. Reset
    # to a clean session and show the welcome + full menu.
    if raw_input.strip().lower() in {"go back to main menu", "back to main menu", "main menu"}:
        clean = state.model_copy(update={"flow": None, "flow_state": "main_menu",
                                         "collected_data": {}})
        logger.info("[ENTRY] conv=%s → global 'go back to main menu'", conversation_id)
        return _greeting_response(clean, raw_input, conversation_id)

    # ── Awaiting phone (collect registered mobile → send OTP) ─────────────────
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
        # Flow finished with a terminal reply but still has queued intents →
        # resume the next one now and merge, so it isn't stranded.
        if _is_terminal_resp(resp) and new_state.collected_data.get("pending_intents"):
            return _resume_pending_after_terminal(resp, new_state, raw_input, input_type)
        return resp
    if state.flow in _AUTH_FLOWS:
        if not state.authenticated:
            return _auth_unavailable(state, raw_input, conversation_id, pending_intent=state.flow)
        resp, new_state = _AUTH_FLOWS[state.flow](state, raw_input)
        if resp.status == "route_to_entry" and not resp.reply_message:
            pending = new_state.collected_data.get("pending_intents", [])
            if pending:
                return _dispatch_pending_intent(new_state, raw_input, pending)
            return _resolve_and_dispatch(new_state, raw_input, input_type)
        if _is_terminal_resp(resp) and new_state.collected_data.get("pending_intents"):
            return _resume_pending_after_terminal(resp, new_state, raw_input, input_type)
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
    # Authenticated (Client ID present) → full menu. Otherwise only pre-login
    # options, so we never offer a flow that would be blocked by the auth gate.
    menu = _FULL_MENU if state.authenticated else _PRELOGIN_MENU
    logger.info("[ENTRY] conv=%s greeting → %s menu", conversation_id,
                "full" if state.authenticated else "pre-login")
    return InternalMessageResponse(
        reply_message=_GREETING_MSG, quick_reply_options=menu,
        flow_state="greeting", status="ok", eventid="1001",
    )


def _auth_unavailable(state, raw_input, conversation_id, pending_intent) -> InternalMessageResponse:
    """Account (post-login) flow requested but no Client ID (Sub-Account ID) is
    on the session.

    Client ID is OPTIONAL for general flows but MANDATORY for account flows, and
    it is supplied by the client in the request payload — we NEVER ask the
    customer for it in chat. If it's missing we can't identify the account, so we
    show a brief notice and the main menu instead of running the flow or
    prompting for the ID.
    """
    _MSG = (
        "This feature needs you to be signed in to your Axis Direct account. "
        "Please access it from your logged-in session.\n\n"
        "In the meantime, here's what I can help you with:"
    )
    save_session(conversation_id, state.model_copy(update={
        "flow": None, "flow_state": "main_menu", "collected_data": {},
        "history": state.history + [
            {"role": "user",      "content": raw_input},
            {"role": "assistant", "content": _MSG},
        ],
    }))
    logger.info("[ENTRY] conv=%s auth-required flow %r → no Client ID, menu fallback",
                conversation_id, pending_intent)
    return InternalMessageResponse(
        reply_message=_MSG, quick_reply_options=_PRELOGIN_MENU,
        flow_state="main_menu", status="auth_required", eventid="1001",
    )


def _try_faq_answer(state: SessionState, raw_input: str):
    """Try to answer a general/FAQ question from the knowledge base
    (retrieve + LLM compose). Returns an InternalMessageResponse with the KB
    answer when confident, else None so the caller shows the menu fallback.
    Never raises."""
    try:
        from src.core.faq import answer_faq
        faq = answer_faq(raw_input)
    except Exception as exc:
        logger.warning("[ENTRY] FAQ attempt failed: %s — menu fallback", exc)
        return None

    if not (faq.get("found") and faq.get("answer")):
        return None

    faq_reply = faq["answer"]
    # After an FAQ answer, offer only navigation — not the full service menu.
    # flow_state session_end_response so the shared session-end node handles
    # the "Go back to main menu" / "End Chat" taps (and any new free-text query
    # re-routes to the classifier).
    save_session(state.conversation_id, state.model_copy(update={
        "flow": None, "flow_state": "session_end_response",
        "history": state.history + [
            {"role": "user",      "content": raw_input},
            {"role": "assistant", "content": faq_reply},
        ],
    }))
    logger.info("[ENTRY] conv=%s answered from FAQ KB (score=%.3f)",
                state.conversation_id, faq.get("score", 0.0))
    return InternalMessageResponse(
        reply_message=faq_reply,
        quick_reply_options=["Go back to main menu", "End Chat"],
        flow_state="session_end_response", status="ok", eventid="1001",
        debug_info={
            "kb_retrieval": {
                "query":    raw_input,
                "top_score": round(float(faq.get("score", 0.0) or 0.0), 4),
                "chunks":   faq.get("retrieved", []),
            }
        },
    )


def _resolve_and_dispatch(state: SessionState, raw_input: str, input_type: str) -> InternalMessageResponse:
    """Classify intent (button fast-path or Haiku), gate auth flows, dispatch."""
    raw_lower   = raw_input.strip().lower()
    fast_intent = _BUTTON_EXACT.get(raw_lower)
    is_multi = False
    intents  = []
    complaint = False   # set by the router for genuine grievances/disputes

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
        complaint  = bool(parsed.get("complaint", False))

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

        _valid_set = {*_NO_AUTH_FLOWS, *_AUTH_FLOWS, "greeting", "escalate_to_human", "faq", "unknown"}
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

        if confidence < _THRESHOLD or intent not in (*_NO_AUTH_FLOWS, *_AUTH_FLOWS, "greeting", "escalate_to_human", "faq"):
            if intent not in ("greeting", "escalate_to_human", "faq"):
                intent = "unknown"; intents = ["unknown"]; is_multi = False

    # ── Auth gate: auth-required flow but not authenticated → phone flow ──────
    # (Single-intent only. For MULTI-intent, the block below handles auth so it
    # can stash ALL the other intents in pending_intents and resume them after
    # OTP — otherwise the extra intents would be silently dropped.)
    if not is_multi and intent in _AUTH_FLOWS and not state.authenticated:
        # Client ID (Sub-Account ID) is mandatory for account flows and comes in
        # the request payload. Without it we can't identify the customer. We do
        # NOT prompt for it — show a sign-in notice + menu instead.
        return _auth_unavailable(state, raw_input, state.conversation_id, pending_intent=intent)

    # ── Multi-intent (PARALLEL first-step execution) ──────────────────────────
    _SINGLE_SHOT_FLOWS = {"bank_query", "edit_profile", "ipo", "account_details", "login_query"}
    if is_multi:
        # If any auth flow is present and not authenticated → no Client ID.
        if any(i in _AUTH_FLOWS for i in intents) and not state.authenticated:
            first_auth = next(i for i in intents if i in _AUTH_FLOWS)
            return _auth_unavailable(state, raw_input, state.conversation_id, pending_intent=first_auth)

        all_flows = {**_NO_AUTH_FLOWS, **_AUTH_FLOWS}
        # Keep only real, dispatchable flow intents, preserving classifier order.
        run_intents = [i for i in intents if i in all_flows]
        if run_intents:
            resp = _run_multi_intent_parallel(state, raw_input, run_intents, all_flows,
                                              _SINGLE_SHOT_FLOWS)
            return _append_complaint_offer(resp, complaint)

    # ── Single-intent dispatch ────────────────────────────────────────────────
    all_flows = {**_NO_AUTH_FLOWS, **_AUTH_FLOWS}
    if intent in all_flows:
        new_state = state.model_copy(update={"flow": intent, "flow_state": "start"})
        save_session(state.conversation_id, new_state)
        import time as _t
        _t0 = _t.perf_counter()
        resp, _ = all_flows[intent](new_state, raw_input)
        logger.info("[TIMING] flow_dispatch_ms=%d flow=%s", int((_t.perf_counter() - _t0) * 1000), intent)
        return _append_complaint_offer(resp, complaint)

    if intent == "greeting":
        return _greeting_response(state, raw_input, state.conversation_id)

    if intent == "escalate_to_human":
        new_state = state.model_copy(update={"flow": "need_more_help", "flow_state": "start"})
        save_session(state.conversation_id, new_state)
        resp, _ = handle_need_more_help(new_state, raw_input, "Incoming message")
        return resp

    # ── FAQ intent → answer from the KB ───────────────────────────────────────
    # The router explicitly classified this as an informational/FAQ question
    # ('what is X', 'how does X work', 'are X allowed', etc.). Answer from the
    # knowledge base; if the KB has no confident match, show the menu.
    if intent == "faq":
        faq_resp = _try_faq_answer(state, raw_input)
        if faq_resp is not None:
            return faq_resp
        # no confident KB answer → fall through to the menu below

    # ── Unknown / out-of-scope → FAQ attempt, then helpful fallback + menu ────
    # A general question not matching a flow. Try the FAQ KB first (catches
    # informational questions the router left as 'unknown'); if no match, show
    # the scope message + menu. (Explicit human requests handled above.)
    if intent == "unknown":
        faq_resp = _try_faq_answer(state, raw_input)
        if faq_resp is not None:
            return faq_resp

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
    menu = _FULL_MENU if state.authenticated else _PRELOGIN_MENU
    return InternalMessageResponse(
        reply_message=_UNKNOWN_MSG, quick_reply_options=menu,
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
                             + ", ".join(_intent_label(i) for i in pending_queue)
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


def _is_terminal_resp(resp: InternalMessageResponse) -> bool:
    """True when a flow turn has finished (nothing more expected for it)."""
    return (resp.flow_state in ("session_end_response", "ended")
            or resp.status in ("end", "escalate"))


def _resume_pending_after_terminal(active_resp, new_state, raw_input, input_type):
    """A flow just produced a TERMINAL reply (e.g. statement emailed). If there
    are still queued multi-intent flows, start the next one NOW and merge its
    first-step reply beneath the terminal reply, so the second request isn't
    stranded. Returns a merged response, or the original if nothing is queued."""
    pending = list(new_state.collected_data.get("pending_intents", []) or [])
    if not pending:
        return active_resp

    nxt = _dispatch_pending_intent(new_state, raw_input, pending)

    # Merge: show the completed flow's reply, then the next flow's prompt.
    merged_msg = (active_resp.reply_message or "").rstrip()
    if nxt.reply_message:
        merged_msg = f"{merged_msg}\n\n---\n\n{nxt.reply_message}" if merged_msg else nxt.reply_message
    return nxt.model_copy(update={"reply_message": merged_msg})


def _dispatch_pending_intent(state: SessionState, raw_input: str, pending: list[str]) -> InternalMessageResponse:
    """Dispatch the next queued intent from a multi-intent session."""
    all_flows = {**_NO_AUTH_FLOWS, **_AUTH_FLOWS}
    next_intent = pending[0]
    remaining   = pending[1:]
    # Prefer the ORIGINAL multi-intent message (stashed as pending_raw_input) so
    # the queued flow can slot-fill the parts meant for it (e.g. "order history
    # for last month"), instead of the current input which may be "Generate".
    dispatch_input = state.collected_data.get("pending_raw_input") or raw_input
    logger.info("[ENTRY] conv=%s pending: dispatching %r remaining=%r (input=%r)",
                state.conversation_id, next_intent, remaining, dispatch_input[:60])
    # Start the queued flow with a CLEAN slate — do NOT inherit the previous
    # flow's slots (report_name, ranges, segment, charges_type, etc.), which
    # would cross-contaminate (e.g. the statement's FY range leaking into the
    # order-history request). Only carry control keys forward.
    _cd = {"pending_intents": remaining}
    # Keep the original message available so this flow (and any still queued)
    # can slot-fill the parts meant for it.
    if remaining:
        _cd["pending_raw_input"] = state.collected_data.get("pending_raw_input") or raw_input
    new_state = state.model_copy(update={
        "flow": next_intent, "flow_state": "start",
        "collected_data": _cd,
    })
    save_session(state.conversation_id, new_state)
    if next_intent in all_flows:
        resp, _ = all_flows[next_intent](new_state, dispatch_input)
        transition = f"Now helping you with **{_intent_label(next_intent)}**:\n\n"
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
