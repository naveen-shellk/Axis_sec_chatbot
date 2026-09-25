"""
chatbot_web/src/flows/order_status/handler.py
----------------------------------------------
Order Status flow — post-login.

Optimised pattern:
  - LLM handles customer INPUT (extracts segment, order type, date, scrip name)
  - Response messages are HARDCODED
  - LLM only called when exact/partial match fails

State machine:
  start → segment_selection → order_type_selection
    → Today Order Status → today_order_ask → fetch & display / send email
    → Order History      → order_date_selection → send history email
  session_end_response → shared
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from src.core.llm import call_intent_llm
from src.core.session_store import save_session
from src.shared.session_end import handle_session_end
from models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

SEGMENTS    = ["Equity", "Commodity", "Derivatives", "Mutual Funds"]
ORDER_TYPES = ["Today Order Status", "Order History"]


def _last_30_days() -> list[str]:
    today = date.today()
    return [(today - timedelta(days=i)).strftime("%d-%m-%Y") for i in range(1, 31)]


# ── Hardcoded messages ────────────────────────────────────────────────────────

def _msg_segment() -> str:
    return "Please select the market segment:"

def _msg_order_type(segment: str) -> str:
    return f"What would you like to check for {segment}?"

def _msg_today_ask() -> str:
    return "Would you like to check the status of a specific order placed today?"

def _msg_scrip_ask() -> str:
    return "Please enter the stock/scrip name:"

def _msg_order_found(o: dict, sub_id: str) -> str:
    return (
        f"Trading ID {sub_id}: Order {o.get('omsOrderId','N/A')} to "
        f"{o.get('transactionType','N/A')} {o.get('tradeQty',0)} "
        f"{o.get('symbol','N/A')} @ ₹{o.get('tradePrice',0.0):.2f} "
        f"{o.get('product','N/A')} on {o.get('exchange','NSE')}"
    )

def _msg_no_orders(segment: str) -> str:
    return f"There are no orders placed today in {segment}."

def _msg_orders_sent() -> str:
    return "We have sent you the order book for your orders today on your registered E-mail ID."

def _msg_history_sent() -> str:
    return "We have sent you the order book for the selected period on your registered E-mail ID."

def _msg_date_picker() -> str:
    return "Please select the date for order history (last 30 days):"

def _msg_deactivated() -> str:
    return (
        "Your account is currently deactivated. Please contact our support team "
        "to reactivate your account.\n\n📞 Customer Care: 022-40508080 / 022-61480808"
    )

def _msg_reprompt(options: list[str]) -> str:
    return f"I didn't catch that. Please select from the options:"

def _msg_scrip_reprompt() -> str:
    return "Please enter a valid stock/scrip name to search your orders:"


# ── LLM extraction (input only) ───────────────────────────────────────────────

_EXTRACT_SEGMENT_SYS = """\
The customer is selecting a market segment on Axis Direct.
Options: Equity, Commodity, Derivatives, Mutual Funds
Extract which segment the customer meant.
Return JSON only:
{"selected": "<exact option or null>", "reasoning": "<one line>"}
"""

_EXTRACT_ORDER_TYPE_SYS = """\
The customer is choosing between "Today Order Status" and "Order History" on Axis Direct.
Extract their choice from the message.
Return JSON only:
{"selected": "<Today Order Status|Order History|null>", "reasoning": "<one line>"}
"""

_EXTRACT_DATE_SYS = """\
The customer is selecting a date (DD-MM-YYYY format) for order history on Axis Direct.
Available dates are in context. Extract the date they meant.
Return JSON only:
{"selected": "<DD-MM-YYYY or null>", "reasoning": "<one line>"}
"""

_EXTRACT_SCRIP_SYS = """\
The customer is entering a stock/scrip name to look up their order on Axis Direct.
Extract the stock or company name from their message.
Return JSON only:
{"selected": "<stock name or null>", "reasoning": "<one line>"}
"""


def _extract(system: str, message: str, context: dict | None = None) -> dict:
    ctx = ""
    if context:
        ctx = "\n".join(f"{k}: {v}" for k, v in context.items())
    full = f"{ctx}\n\nCustomer: {message}" if ctx else f"Customer: {message}"
    result = call_intent_llm(system, [{"role": "user", "content": [{"text": full}]}])
    return result.get("parsed") or {}


def _match(text: str, options: list[str]) -> str | None:
    tl = text.strip().lower()
    for opt in options:
        if opt.lower() == tl:
            return opt
    for opt in options:
        if opt.lower() in tl or tl in opt.lower():
            return opt
    return None


def _hist(state: SessionState, msg: str, reply: str) -> list:
    return state.history + [
        {"role": "user",      "content": msg},
        {"role": "assistant", "content": reply},
    ]


# ── Main handler ──────────────────────────────────────────────────────────────

def handle_order_status(
    state: SessionState,
    customer_message: str,
) -> tuple[InternalMessageResponse, SessionState]:

    fs = state.flow_state

    # ── START: account check ──────────────────────────────────────────────────
    if fs == "start":
        from src.gateways.customer_api import get_customer_profile
        try:
            profile = get_customer_profile(state.sub_account_id or "")
            status  = profile.account_status
        except Exception:
            status = "active"

        if status in ("deactivated", "purged"):
            reply = _msg_deactivated()
            ns = state.model_copy(update={
                "flow": "order_status", "flow_state": "session_end_response",
                "history": _hist(state, customer_message, reply),
            })
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply,
                    quick_reply_options=["Go back to main menu", "End Chat"],
                    flow_state="session_end_response", status="end",
                ), ns,
            )

        reply = _msg_segment()
        ns = state.model_copy(update={
            "flow": "order_status", "flow_state": "segment_selection",
            "history": _hist(state, customer_message, reply),
        })
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=reply,
                quick_reply_options=SEGMENTS,
                flow_state="segment_selection", status="ok",
            ), ns,
        )

    # ── SEGMENT SELECTION ─────────────────────────────────────────────────────
    if fs == "segment_selection":
        segment = _match(customer_message, SEGMENTS)
        if not segment:
            parsed  = _extract(_EXTRACT_SEGMENT_SYS, customer_message)
            segment = _match(parsed.get("selected", ""), SEGMENTS)

        if not segment:
            reply = _msg_reprompt(SEGMENTS)
            ns = state.model_copy(update={"history": _hist(state, customer_message, reply)})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply, quick_reply_options=SEGMENTS,
                    flow_state="segment_selection", status="reprompt",
                ), ns,
            )

        reply = _msg_order_type(segment)
        ns = state.model_copy(update={
            "flow_state": "order_type_selection",
            "collected_data": {**state.collected_data, "segment": segment},
            "history": _hist(state, customer_message, reply),
        })
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=reply, quick_reply_options=ORDER_TYPES,
                flow_state="order_type_selection", status="ok",
            ), ns,
        )

    # ── ORDER TYPE SELECTION ──────────────────────────────────────────────────
    if fs == "order_type_selection":
        segment = state.collected_data.get("segment", "Equity")
        order_type = _match(customer_message, ORDER_TYPES)
        if not order_type:
            parsed     = _extract(_EXTRACT_ORDER_TYPE_SYS, customer_message)
            order_type = _match(parsed.get("selected", ""), ORDER_TYPES)

        if not order_type:
            reply = _msg_reprompt(ORDER_TYPES)
            ns = state.model_copy(update={"history": _hist(state, customer_message, reply)})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply, quick_reply_options=ORDER_TYPES,
                    flow_state="order_type_selection", status="reprompt",
                ), ns,
            )

        if order_type == "Today Order Status":
            reply = _msg_today_ask()
            ns = state.model_copy(update={
                "flow_state": "today_order_ask",
                "collected_data": {**state.collected_data, "order_type": "today"},
                "history": _hist(state, customer_message, reply),
            })
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply, quick_reply_options=["Yes", "No"],
                    flow_state="today_order_ask", status="ok",
                ), ns,
            )
        else:
            dates = _last_30_days()[:10]
            reply = _msg_date_picker()
            ns = state.model_copy(update={
                "flow_state": "order_date_selection",
                "collected_data": {**state.collected_data, "order_type": "history"},
                "history": _hist(state, customer_message, reply),
            })
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply, quick_reply_options=dates,
                    flow_state="order_date_selection", status="ok",
                ), ns,
            )

    # ── TODAY ORDER: specific scrip? ──────────────────────────────────────────
    if fs == "today_order_ask":
        segment   = state.collected_data.get("segment", "Equity")
        msg_lower = customer_message.strip().lower()
        wants_specific = "yes" in msg_lower or msg_lower == "y"

        if wants_specific:
            reply = _msg_scrip_ask()
            ns = state.model_copy(update={
                "flow_state": "today_order_scrip",
                "history": _hist(state, customer_message, reply),
            })
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply, quick_reply_options=[],
                    flow_state="today_order_scrip", status="ok",
                ), ns,
            )
        else:
            from src.gateways.order_api import send_order_history_email
            today_str = date.today().strftime("%d-%m-%Y")
            result = send_order_history_email(state.sub_account_id or "", segment, today_str)
            if not result.get("success", False):
                _err = (
                    "We were unable to send your order book at this time.\n\n"
                    "Please try again later or contact support: 📞 022-40508080 / 022-61480808"
                )
                ns = state.model_copy(update={
                    "flow_state": "session_end_response",
                    "history": _hist(state, customer_message, _err),
                })
                save_session(state.conversation_id, ns)
                return (
                    InternalMessageResponse(
                        reply_message=_err,
                        quick_reply_options=["Go back to main menu", "End Chat"],
                        flow_state="session_end_response", status="error",
                    ), ns,
                )
            reply = _msg_orders_sent()
            ns = state.model_copy(update={
                "flow_state": "session_end_response",
                "history": _hist(state, customer_message, reply),
            })
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply,
                    quick_reply_options=["Go back to main menu", "End Chat"],
                    flow_state="session_end_response", status="ok",
                ), ns,
            )

    # ── TODAY ORDER: scrip name ───────────────────────────────────────────────
    if fs == "today_order_scrip":
        segment = state.collected_data.get("segment", "Equity")

        # LLM extracts scrip name from free text
        parsed     = _extract(_EXTRACT_SCRIP_SYS, customer_message)
        scrip_name = parsed.get("selected") or customer_message.strip()

        if not scrip_name:
            reply = _msg_scrip_reprompt()
            ns = state.model_copy(update={"history": _hist(state, customer_message, reply)})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply, quick_reply_options=[],
                    flow_state="today_order_scrip", status="reprompt",
                ), ns,
            )

        from src.gateways.order_api import get_todays_orders
        order_data   = get_todays_orders(state.sub_account_id or "", segment)

        if not order_data.get("found", False) and order_data.get("error"):
            # API failed — not just empty results
            _err = (
                "We were unable to fetch your order details at this time.\n\n"
                "Please try again later or contact support: 📞 022-40508080 / 022-61480808"
            )
            ns = state.model_copy(update={
                "flow_state": "session_end_response",
                "history": _hist(state, customer_message, _err),
            })
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=_err,
                    quick_reply_options=["Go back to main menu", "End Chat"],
                    flow_state="session_end_response", status="error",
                ), ns,
            )

        orders       = order_data.get("orders", [])
        scrip_lower  = scrip_name.lower()
        matched      = [o for o in orders if scrip_lower in str(o.get("symbol","")).lower()]

        if matched:
            reply = _msg_order_found(matched[0], state.sub_account_id or "")
        else:
            reply = _msg_no_orders(segment)

        ns = state.model_copy(update={
            "flow_state": "session_end_response",
            "history": _hist(state, customer_message, reply),
        })
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=reply,
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state="session_end_response", status="ok",
            ), ns,
        )

    # ── ORDER HISTORY: date ───────────────────────────────────────────────────
    if fs == "order_date_selection":
        segment = state.collected_data.get("segment", "Equity")
        dates   = _last_30_days()[:10]

        # Exact match first
        selected_date = _match(customer_message, dates)
        if not selected_date:
            parsed        = _extract(_EXTRACT_DATE_SYS, customer_message, {"dates": ", ".join(dates)})
            selected_date = _match(parsed.get("selected", ""), dates) or parsed.get("selected")

        if not selected_date:
            reply = _msg_reprompt(dates)
            ns = state.model_copy(update={"history": _hist(state, customer_message, reply)})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply, quick_reply_options=dates,
                    flow_state="order_date_selection", status="reprompt",
                ), ns,
            )

        from src.gateways.order_api import send_order_history_email
        result = send_order_history_email(state.sub_account_id or "", segment, selected_date)
        if not result.get("success", False):
            _err = (
                "We were unable to send your order history at this time.\n\n"
                "Please try again later or contact support: 📞 022-40508080 / 022-61480808"
            )
            ns = state.model_copy(update={
                "flow_state": "session_end_response",
                "history": _hist(state, customer_message, _err),
            })
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=_err,
                    quick_reply_options=["Go back to main menu", "End Chat"],
                    flow_state="session_end_response", status="error",
                ), ns,
            )
        reply = _msg_history_sent()
        ns = state.model_copy(update={
            "flow_state": "session_end_response",
            "history": _hist(state, customer_message, reply),
        })
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=reply,
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state="session_end_response", status="ok",
            ), ns,
        )

    if fs == "session_end_response":
        return handle_session_end(state, customer_message)

    logger.warning("[ORDER_STATUS] unknown flow_state %r — reset", fs)
    return handle_order_status(state.model_copy(update={"flow_state": "start"}), customer_message)
