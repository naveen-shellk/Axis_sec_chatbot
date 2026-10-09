"""
chatbot_web/src/flows/how_to_trade/handler.py
-----------------------------------------------
How To Trade — pre-login flow, no tools needed (static instructions).

State machine (identical to chatbot/src/flows/how_to_trade/handler.py):
  start
    → LLM asks trade type    → trade_type_selection
  trade_type_selection
    → Cash & ETF / E-Margin  → LLM asks BUY/SELL → buy_sell_selection
    → Derivatives            → LLM asks sub-type  → derivative_sub_selection
    → Other                  → LLM asks sub-type  → other_sub_selection
    → Stop Loss              → skip to app_type_selection
  buy_sell_selection / derivative_sub_selection / other_sub_selection
    → build full key → LLM asks which app → app_type_selection
  app_type_selection
    → look up static instructions → show steps → session_end_response
  session_end_response
    → shared handle_session_end()
"""

from __future__ import annotations

import logging

from src.core.conversation import run_conversation_turn
from src.core.llm import call_intent_llm
from src.core.session_store import save_session
from src.flows.how_to_trade.instructions import (
    APP_TYPES, APP_TYPES_NO_INVESTORS,
    BUY_SELL_TRADES, BUY_SELL_TYPES,
    DERIVATIVE_TYPES, OTHER_TYPES, TRADE_TYPES,
    get_available_apps, get_instructions,
)
from src.shared.session_end import handle_session_end
from models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

# ── Free-text slot extraction ─────────────────────────────────────────────────
# When the customer's FIRST message already names the trade type / sub-type /
# direction / app (e.g. "how to sell holdings using Encash"), extract those and
# skip the step-by-step questions. Mirrors the slot-fill in the other flows.
_EXTRACT_SLOTS_SYS = """\
You extract 'how to trade' details from an Axis Direct customer's message.

Top-level trade types: Cash & ETF, E-Margin & Intraday, Stop Loss, Derivatives, Other
Derivative sub-types:  Futures, Options, FNO Sell
"Other" special order sub-types: Encash, GTDT, Intersettlement, Cover
Apps: Traders App, Investors App, Swift Trade

Return JSON ONLY:
{
  "trade_type": "<one of the top-level types OR a sub-type (Futures/Options/FNO Sell/Encash/GTDT/Intersettlement/Cover), or null>",
  "direction":  "BUY | SELL | null",
  "app":        "Traders App | Investors App | Swift Trade | null"
}

Keyword hints (map the customer's wording to the trade_type):
- Cash & ETF        : buy/sell stocks, shares, ETF, cash, delivery, demat holdings
- E-Margin & Intraday: e-margin, MTF, margin, intraday, leverage
- Stop Loss         : stop loss, SL order, trailing stop, stop-loss trigger
- Derivatives       : futures, options, FNO, F&O (then sub-type Futures/Options/FNO Sell)
- Encash            : encash, instant payout sell, encash product
- GTDT              : GTDT, good till date, GTC, extended-validity order
- Intersettlement   : intersettlement, T1/T+1 sell, unsettled holdings
- Cover             : cover order, CO order, intraday with mandatory stop-loss

Rules:
- Map the customer's wording to the CLOSEST known value and return the EXACT label.
  e.g. "sell my holdings using Encash" -> trade_type="Encash", direction="SELL".
  "buy shares in cash" -> trade_type="Cash & ETF", direction="BUY".
- A named sub-type (Encash/GTDT/Intersettlement/Cover/Futures/Options/FNO Sell)
  IS the trade_type — do NOT return "Other" or "Derivatives" when the specific
  sub-type is clear.
- direction (BUY/SELL) applies ONLY to Cash & ETF and E-Margin & Intraday; it is
  null for every other type (Encash/GTDT/Stop Loss/etc. have no BUY/SELL step).
- Only fill a field the customer actually implied; else null.
- Do NOT invent values.
"""


def _extract_slots(message: str) -> dict:
    """One Haiku call -> whatever how-to-trade slots the free text names."""
    result = call_intent_llm(
        _EXTRACT_SLOTS_SYS,
        [{"role": "user", "content": [{"text": f"Customer message: {message}"}]}],
    )
    parsed = result.get("parsed") or {}
    out: dict = {}

    tt = parsed.get("trade_type")
    if tt in TRADE_TYPES or tt in DERIVATIVE_TYPES or tt in OTHER_TYPES:
        out["trade_type"] = tt

    d = (parsed.get("direction") or "").upper()
    if d in BUY_SELL_TYPES:
        out["direction"] = d

    app = parsed.get("app")
    if app in APP_TYPES:
        out["app"] = app

    return out


def _looks_like_free_text(message: str, step_options: list[str]) -> bool:
    """True when the message is genuine free text, not an exact button tap."""
    msg = (message or "").strip()
    if not msg:
        return False
    low = msg.lower()
    if any(opt.lower() == low for opt in (step_options or [])):
        return False
    if low in {"go back to main menu", "main menu", "end chat", "yes", "no"}:
        return False
    return True

# ── LLM system prompts ────────────────────────────────────────────────────────

_TRADE_TYPE_SYS = """\
You are helping an Axis Direct customer learn how to place a trade.
Ask which type of trade they want guidance on.
Return JSON:
{"message": "<ask which trade type>",
 "quick_replies": ["Cash & ETF", "E-Margin & Intraday", "Stop Loss", "Derivatives", "Other"],
 "flow_action": "reprompt", "reasoning": ""}
"""

_BUY_SELL_SYS = """\
You are asking an Axis Direct customer BUY or SELL for {trade_type}.
Return JSON:
{{"message": "<ask BUY or SELL>",
 "quick_replies": ["BUY", "SELL"],
 "flow_action": "reprompt", "reasoning": ""}}
"""

_DERIVATIVE_SUB_SYS = """\
You are asking which derivative type they want to learn about on Axis Direct.
Return JSON:
{"message": "<ask Futures, Options or FNO Sell>",
 "quick_replies": ["Futures", "Options", "FNO Sell"],
 "flow_action": "reprompt", "reasoning": ""}
"""

_OTHER_SUB_SYS = """\
You are asking which special order type they want to learn about on Axis Direct.
Return JSON:
{"message": "<ask Encash, GTDT, Intersettlement or Cover>",
 "quick_replies": ["Encash", "GTDT", "Intersettlement", "Cover"],
 "flow_action": "reprompt", "reasoning": ""}
"""

_APP_SYS = """\
You are asking which Axis Direct trading app the customer uses.
Return JSON:
{"message": "<ask Traders App, Investors App, or Swift Trade>",
 "quick_replies": ["Traders App", "Investors App", "Swift Trade"],
 "flow_action": "reprompt", "reasoning": ""}
"""

_APP_NO_INVESTORS_SYS = """\
You are asking which Axis Direct trading app the customer uses.
Note: Cover orders are NOT available on Investors App.
Return JSON:
{"message": "<ask Traders App or Swift Trade only>",
 "quick_replies": ["Traders App", "Swift Trade"],
 "flow_action": "reprompt", "reasoning": ""}
"""

_INSTRUCTIONS_SYS = """\
You are providing step-by-step trading instructions to an Axis Direct customer.
Use EXACTLY the instructions from backend_data["instructions"]. Do not change a word.
Return JSON:
{{"message": "<exact instructions>",
 "quick_replies": ["Go back to main menu", "End Chat"],
 "flow_action": "session_end", "reasoning": ""}}
"""

_UNSUPPORTED_SYS = """\
You are informing an Axis Direct customer that the combination they selected is not supported.
Be polite and suggest they pick a different option.
Return JSON:
{"message": "<polite not-supported message>",
 "quick_replies": ["Go back to main menu", "End Chat"],
 "flow_action": "session_end", "reasoning": ""}
"""


# ── LLM wrapper ───────────────────────────────────────────────────────────────

def _llm(state: SessionState, msg: str, bd: dict, system: str, next_fs: str) -> tuple[dict, list]:
    resp = run_conversation_turn(
        state_data={"flow": "how_to_trade", "flow_state": next_fs},
        history=state.history,
        customer_message=msg,
        backend_data=bd,
        system_override=system,
    )
    hist = state.history + [
        {"role": "user",      "content": msg},
        {"role": "assistant", "content": resp["message"]},
    ]
    return resp, hist


def _match(text: str, options: list[str]) -> str | None:
    # EXACT match only (button tap / verbatim option). Non-exact typed input is
    # left for the LLM (_llm via run_conversation_turn) — no substring guessing,
    # per the rule "exact predefined match → programmatic; everything else → LLM".
    tl = text.strip().lower()
    for opt in options:
        if opt.lower() == tl:
            return opt
    return None


def _ask_app(state, customer_message, trade_type, collected_data):
    available = get_available_apps(trade_type) or APP_TYPES
    sys = _APP_NO_INVESTORS_SYS if trade_type == "Cover" else _APP_SYS
    resp, hist = _llm(state, f"(system: selected {trade_type})", {}, sys, "app_type_selection")
    new_state = state.model_copy(update={
        "flow_state": "app_type_selection", "collected_data": collected_data, "history": hist,
    })
    save_session(state.conversation_id, new_state)
    return (
        InternalMessageResponse(
            reply_message=resp["message"],
            quick_reply_options=available,
            flow_state="app_type_selection",
            status="ok",
        ),
        new_state,
    )


def _serve_instructions(state, trade_type, app, customer_message, cd):
    """Serve the static steps for a fully-resolved trade_type + app, or an
    unsupported message if the combination has no instructions."""
    steps = get_instructions(trade_type, app)
    if not steps:
        resp, hist = _llm(state, customer_message, {}, _UNSUPPORTED_SYS, "session_end_response")
        ns = state.model_copy(update={"flow": "how_to_trade",
                                      "flow_state": "session_end_response", "history": hist})
        save_session(state.conversation_id, ns)
        return (InternalMessageResponse(reply_message=resp["message"],
                                        quick_reply_options=["Go back to main menu", "End Chat"],
                                        flow_state="session_end_response", status="ok"), ns)
    reply = f"Here's how to place a *{trade_type}* order on *{app}*:\n\n{steps}"
    hist = state.history + [
        {"role": "user", "content": customer_message},
        {"role": "assistant", "content": reply},
    ]
    next_fs = "account_status_check" if state.sub_account_id else "session_end_response"
    ns = state.model_copy(update={
        "flow": "how_to_trade", "flow_state": next_fs,
        "collected_data": {**cd, "trade_type": trade_type, "app": app}, "history": hist,
    })
    save_session(state.conversation_id, ns)
    logger.info("[HOW_TO_TRADE] conv=%s slot-fill served trade=%r app=%r",
                state.conversation_id, trade_type, app)
    return (InternalMessageResponse(reply_message=reply,
                                    quick_reply_options=["Go back to main menu", "End Chat"],
                                    flow_state=next_fs, status="ok"), ns)


def _try_slot_fill(state: SessionState, customer_message: str):
    """Free-text pre-pass. Resolve as far as the message allows and jump to the
    right step: full instructions, or app selection, or BUY/SELL. Returns a
    response tuple to short-circuit, or None to fall back to the normal start."""
    slots = _extract_slots(customer_message)
    tt = slots.get("trade_type")
    if not tt:
        return None  # nothing useful named → normal step-by-step

    cd = {**state.collected_data, "trade_type": tt}
    needs_dir = tt in BUY_SELL_TRADES
    direction = slots.get("direction")
    app = slots.get("app")

    # BUY/SELL types: fold direction into the key when known.
    if needs_dir:
        if not direction:
            # Trade type known but direction missing → ask BUY/SELL.
            sys = _BUY_SELL_SYS.format(trade_type=tt)
            resp, hist = _llm(state, f"(selected {tt})", {"trade_type": tt},
                              sys, "buy_sell_selection")
            ns = state.model_copy(update={"flow": "how_to_trade",
                                          "flow_state": "buy_sell_selection",
                                          "collected_data": cd, "history": hist})
            save_session(state.conversation_id, ns)
            return (InternalMessageResponse(reply_message=resp["message"],
                                            quick_reply_options=BUY_SELL_TYPES,
                                            flow_state="buy_sell_selection", status="ok"), ns)
        tt = f"{tt} {direction}"
        cd["trade_type"] = tt

    # trade_type (and direction if needed) resolved. App known → serve now.
    if app:
        return _serve_instructions(state, tt, app, customer_message, cd)

    # App missing → jump straight to the app-selection step.
    return _ask_app(state, customer_message, tt, cd)


# ── Main handler ──────────────────────────────────────────────────────────────

def handle_how_to_trade(
    state: SessionState,
    customer_message: str,
) -> tuple[InternalMessageResponse, SessionState]:

    fs = state.flow_state

    # ── Step 1: entry ─────────────────────────────────────────────────────────
    if fs == "start":
        # Free-text slot-fill: the first message may already name the trade
        # type / sub-type / direction / app → skip the questions it answered.
        if _looks_like_free_text(customer_message, []):
            short = _try_slot_fill(state, customer_message)
            if short is not None:
                return short

        resp, hist = _llm(state, customer_message, {}, _TRADE_TYPE_SYS, "trade_type_selection")
        new_state = state.model_copy(update={
            "flow": "how_to_trade", "flow_state": "trade_type_selection", "history": hist,
        })
        save_session(state.conversation_id, new_state)
        return (
            InternalMessageResponse(
                reply_message=resp["message"],
                quick_reply_options=TRADE_TYPES,
                flow_state="trade_type_selection",
                status="ok",
            ),
            new_state,
        )

    # ── Step 2: trade type ────────────────────────────────────────────────────
    if fs == "trade_type_selection":
        trade_type = _match(customer_message, TRADE_TYPES)
        if not trade_type:
            resp, hist = _llm(state, customer_message, {}, _TRADE_TYPE_SYS, "trade_type_selection")
            new_state = state.model_copy(update={"history": hist})
            save_session(state.conversation_id, new_state)
            return (
                InternalMessageResponse(reply_message=resp["message"],
                                        quick_reply_options=TRADE_TYPES,
                                        flow_state="trade_type_selection", status="reprompt"),
                new_state,
            )

        cd = {**state.collected_data, "trade_type": trade_type}

        if trade_type in BUY_SELL_TRADES:
            sys = _BUY_SELL_SYS.format(trade_type=trade_type)
            resp, hist = _llm(state, f"(selected {trade_type})", {"trade_type": trade_type},
                              sys, "buy_sell_selection")
            new_state = state.model_copy(update={
                "flow_state": "buy_sell_selection", "collected_data": cd, "history": hist,
            })
            save_session(state.conversation_id, new_state)
            return (
                InternalMessageResponse(reply_message=resp["message"],
                                        quick_reply_options=BUY_SELL_TYPES,
                                        flow_state="buy_sell_selection", status="ok"),
                new_state,
            )

        if trade_type == "Derivatives":
            resp, hist = _llm(state, "(selected Derivatives)", {}, _DERIVATIVE_SUB_SYS,
                              "derivative_sub_selection")
            new_state = state.model_copy(update={
                "flow_state": "derivative_sub_selection", "collected_data": cd, "history": hist,
            })
            save_session(state.conversation_id, new_state)
            return (
                InternalMessageResponse(reply_message=resp["message"],
                                        quick_reply_options=DERIVATIVE_TYPES,
                                        flow_state="derivative_sub_selection", status="ok"),
                new_state,
            )

        if trade_type == "Other":
            resp, hist = _llm(state, "(selected Other)", {}, _OTHER_SUB_SYS, "other_sub_selection")
            new_state = state.model_copy(update={
                "flow_state": "other_sub_selection", "collected_data": cd, "history": hist,
            })
            save_session(state.conversation_id, new_state)
            return (
                InternalMessageResponse(reply_message=resp["message"],
                                        quick_reply_options=OTHER_TYPES,
                                        flow_state="other_sub_selection", status="ok"),
                new_state,
            )

        # Stop Loss — skip to app
        return _ask_app(state, customer_message, trade_type, cd)

    # ── Step 2a: BUY/SELL ────────────────────────────────────────────────────
    if fs == "buy_sell_selection":
        trade_type = state.collected_data.get("trade_type", "Cash & ETF")
        direction  = _match(customer_message, BUY_SELL_TYPES)
        if not direction:
            sys = _BUY_SELL_SYS.format(trade_type=trade_type)
            resp, hist = _llm(state, customer_message, {}, sys, "buy_sell_selection")
            new_state = state.model_copy(update={"history": hist})
            save_session(state.conversation_id, new_state)
            return (
                InternalMessageResponse(reply_message=resp["message"],
                                        quick_reply_options=BUY_SELL_TYPES,
                                        flow_state="buy_sell_selection", status="reprompt"),
                new_state,
            )
        full_key = f"{trade_type} {direction}"
        return _ask_app(state, customer_message, full_key,
                        {**state.collected_data, "trade_type": full_key})

    # ── Step 2b: Derivatives sub-type ─────────────────────────────────────────
    if fs == "derivative_sub_selection":
        sub = _match(customer_message, DERIVATIVE_TYPES)
        if not sub:
            resp, hist = _llm(state, customer_message, {}, _DERIVATIVE_SUB_SYS,
                              "derivative_sub_selection")
            new_state = state.model_copy(update={"history": hist})
            save_session(state.conversation_id, new_state)
            return (
                InternalMessageResponse(reply_message=resp["message"],
                                        quick_reply_options=DERIVATIVE_TYPES,
                                        flow_state="derivative_sub_selection", status="reprompt"),
                new_state,
            )
        return _ask_app(state, customer_message, sub,
                        {**state.collected_data, "trade_type": sub})

    # ── Step 2c: Other sub-type ───────────────────────────────────────────────
    if fs == "other_sub_selection":
        sub = _match(customer_message, OTHER_TYPES)
        if not sub:
            resp, hist = _llm(state, customer_message, {}, _OTHER_SUB_SYS, "other_sub_selection")
            new_state = state.model_copy(update={"history": hist})
            save_session(state.conversation_id, new_state)
            return (
                InternalMessageResponse(reply_message=resp["message"],
                                        quick_reply_options=OTHER_TYPES,
                                        flow_state="other_sub_selection", status="reprompt"),
                new_state,
            )
        return _ask_app(state, customer_message, sub,
                        {**state.collected_data, "trade_type": sub})

    # ── Step 3: app selection ─────────────────────────────────────────────────
    if fs == "app_type_selection":
        trade_type = state.collected_data.get("trade_type", "")
        available  = get_available_apps(trade_type) or APP_TYPES
        app        = _match(customer_message, available)

        if not app:
            sys = _APP_NO_INVESTORS_SYS if trade_type == "Cover" else _APP_SYS
            resp, hist = _llm(state, customer_message, {}, sys, "app_type_selection")
            new_state = state.model_copy(update={"history": hist})
            save_session(state.conversation_id, new_state)
            return (
                InternalMessageResponse(reply_message=resp["message"],
                                        quick_reply_options=available,
                                        flow_state="app_type_selection", status="reprompt"),
                new_state,
            )

        steps = get_instructions(trade_type, app)

        if not steps:
            resp, hist = _llm(state, customer_message, {}, _UNSUPPORTED_SYS,
                              "session_end_response")
            new_state = state.model_copy(update={
                "flow_state": "session_end_response", "history": hist,
            })
            save_session(state.conversation_id, new_state)
            return (
                InternalMessageResponse(reply_message=resp["message"],
                                        quick_reply_options=["Go back to main menu", "End Chat"],
                                        flow_state="session_end_response", status="ok"),
                new_state,
            )

        # Show static instructions — do NOT pass through LLM to avoid mutation
        reply = f"Here's how to place a *{trade_type}* order on *{app}*:\n\n{steps}"
        hist = state.history + [
            {"role": "user",      "content": customer_message},
            {"role": "assistant", "content": reply},
        ]

        # ── Account status check (spec: after instructions, check active/deactive) ──
        # Only if sub_account_id is available (post-login context)
        next_fs = "account_status_check" if state.sub_account_id else "session_end_response"

        new_state = state.model_copy(update={
            "flow_state": next_fs,
            "collected_data": {**state.collected_data, "app": app},
            "history": hist,
        })
        save_session(state.conversation_id, new_state)
        logger.info("[HOW_TO_TRADE] conv=%s trade=%r app=%r instructions served",
                    state.conversation_id, trade_type, app)
        return (
            InternalMessageResponse(
                reply_message=reply,
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state=next_fs,
                status="ok",
            ),
            new_state,
        )

    # ── Account status check after instructions (post-login only) ─────────────
    if fs == "account_status_check":
        # The buttons shown after trade instructions are "Go back to main menu" /
        # "End Chat". Honour those via the shared session-end node (robust
        # word-boundary matching), instead of running an account-status check
        # that would hijack the customer's navigation choice.
        return handle_session_end(state, customer_message)

    # ── Session end ───────────────────────────────────────────────────────────
    if fs == "session_end_response":
        return handle_session_end(state, customer_message)

    logger.warning("[HOW_TO_TRADE] unknown flow_state %r — reset", fs)
    return handle_how_to_trade(state.model_copy(update={"flow_state": "start"}), customer_message)
