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
from datetime import date, datetime, timedelta

from src.core.llm import call_intent_llm
from src.core.session_store import save_session
from src.shared.session_end import handle_session_end
from models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

SEGMENTS    = ["Equity", "Commodity", "Derivatives", "Currency", "Mutual Funds"]
ORDER_TYPES = ["Today Order Status", "Order History"]

# Segment → profile segmentsEnabled key (thor-aligned). Used to gate the
# trade-book call: if the segment is not active on the customer's account, we
# don't call the API (which would 400 / return no data) — we tell them directly.
_SEGMENT_PROFILE_KEY = {
    "Equity":       "nseCash",
    "Commodity":    "mcxCmx",
    "Derivatives":  "nseFO",
    "Currency":     "nseCdx",
    "Mutual Funds": "bseMF",
}


import re


def _last_30_days() -> list[str]:
    today = date.today()
    return [(today - timedelta(days=i)).strftime("%d-%m-%Y") for i in range(1, 31)]


def _parse_range(text: str) -> tuple[date, date] | None:
    """Extract two DD-MM-YYYY dates (start, end) from the calendar picker's
    'DD-MM-YYYY to DD-MM-YYYY' post, or from free text."""
    dates = re.findall(r"\b(\d{2})[-/](\d{2})[-/](\d{4})\b", text or "")
    if len(dates) >= 2:
        try:
            d1 = date(int(dates[0][2]), int(dates[0][1]), int(dates[0][0]))
            d2 = date(int(dates[1][2]), int(dates[1][1]), int(dates[1][0]))
            return (d1, d2) if d1 <= d2 else (d2, d1)
        except ValueError:
            return None
    return None


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
    return ("Please select a date range within 30 days from the calendar "
            "(DD-MM-YYYY to DD-MM-YYYY) for your order history.")

def _msg_range_wrong() -> str:
    return "Please choose a date range within the last 30 days."

def _msg_deactivated() -> str:
    return (
        "Your account is currently deactivated. Please contact our support team "
        "to reactivate your account.\n\n📞 Customer Care: 022-40508080 / 022-61480808"
    )

def _msg_reprompt(options: list[str]) -> str:
    return f"I didn't catch that. Please select from the options:"

def _msg_segment_inactive(segment: str) -> str:
    return (f"The {segment} segment is not active on your account, so there "
            f"are no {segment} orders to show.\n\nYou can activate it from the "
            f"Axis Direct portal, or choose a different segment.")


def _is_segment_active(state: SessionState, segment: str) -> bool:
    """Thor-aligned gate: is this segment enabled on the customer's profile?
    If the profile can't be read, default to True (don't block) so a transient
    profile hiccup doesn't wrongly deny a valid request."""
    try:
        from src.core.langchain_agent import get_profile
        prof = get_profile(state.sub_account_id or "")
        seg_enabled = getattr(prof, "segments_enabled", {}) or {}
        if not seg_enabled:
            return True  # no data → don't block
        key = _SEGMENT_PROFILE_KEY.get(segment)
        if not key:
            return True
        return bool(seg_enabled.get(key, False))
    except Exception as exc:
        logger.warning("[ORDER_STATUS] segment-active check failed: %s — allowing", exc)
        return True

def _msg_scrip_reprompt() -> str:
    return "Please enter a valid stock/scrip name to search your orders:"


# ── LLM extraction (input only) ───────────────────────────────────────────────

_EXTRACT_SEGMENT_SYS = """\
The customer is selecting a market segment on Axis Direct.
Options: Equity, Commodity, Derivatives, Currency, Mutual Funds
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


# ── Slot-filling: pull EVERY order-status slot from one free-text message ──────
_EXTRACT_SLOTS_SYS = """\
You extract order-status request details from an Axis Direct customer's message.

Market segments: Equity, Commodity, Derivatives, Currency, Mutual Funds
Order type: "Today Order Status" (orders placed today) or "Order History" (past date range)

Return JSON ONLY:
{
  "segment": "<segment or null>",
  "order_type": "Today Order Status | Order History | null",
  "scrip": "<stock/scrip name if they named one, else null>",
  "ranges": [ {"start": "DD-MM-YYYY", "end": "DD-MM-YYYY"} ]
}

Rules:
- Convert ANY date to DD-MM-YYYY. Relative dates ("last month", "past 30 days")
  are resolved RELATIVE TO TODAY (today's date is given in the message context).
- SCOPE: extract ONLY the date period(s) the customer tied to their ORDER
  HISTORY. If the message ALSO asks about other topics (statements, charges,
  holdings), IGNORE the dates that belong to those other topics — do NOT put
  them in "ranges". Example: "ledger statement for FY 2024-25 and my order
  history for last month" -> ranges=[last month only]; the FY 2024-25 range is
  for the statement, NOT order history, so EXCLUDE it.
- If they gave an order-history date period, order_type is "Order History".
- "ranges" may hold multiple periods only for the SAME order-history request; [] if none.
- NEVER HALLUCINATE DATES. Put a range in "ranges" ONLY when the customer gave
  an EXPLICIT period (e.g. "01-01-2026 to 20-01-2026", "last month"). If no
  period was given, or it is vague/unresolvable, return ranges = []. When
  unsure, prefer [] over a guessed date — the flow will ask for the dates.
- Only fill a field the customer actually implied; else null / [].
- Do NOT invent values (dates, segments, scrip names).
"""


def _norm_ddmmyyyy(s: str) -> str | None:
    s = (s or "").strip()
    m = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", s)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1))).strftime("%d-%m-%Y")
        except ValueError:
            return None
    return None


def _extract_slots(message: str) -> dict:
    _today = date.today().strftime("%d-%m-%Y")
    parsed = _extract(_EXTRACT_SLOTS_SYS, message, {"today": _today}) or {}
    out: dict = {}
    if parsed.get("segment") in SEGMENTS:
        out["segment"] = parsed["segment"]
    if parsed.get("order_type") in ORDER_TYPES:
        out["order_type"] = parsed["order_type"]
    if parsed.get("scrip"):
        out["scrip"] = str(parsed["scrip"]).strip()
    ranges = []
    for r in (parsed.get("ranges") or []):
        s = _norm_ddmmyyyy(str(r.get("start", "")))
        e = _norm_ddmmyyyy(str(r.get("end", "")))
        if s and e:
            d1 = datetime.strptime(s, "%d-%m-%Y").date()
            d2 = datetime.strptime(e, "%d-%m-%Y").date()
            if d1 > d2:
                s, e = e, s
            ranges.append({"start": s, "end": e})
    if ranges:
        out["ranges"] = ranges
    return out


def _looks_like_free_text(message: str, step_options: list[str]) -> bool:
    msg = (message or "").strip()
    if not msg:
        return False
    low = msg.lower()
    if any(opt.lower() == low for opt in (step_options or [])):
        return False
    if low in {"generate", "change", "yes", "no", "y", "n",
               "go back to main menu", "main menu", "end chat"}:
        return False
    residue = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", "", msg)
    residue = re.sub(r"[\s,./\-]|(?:\bto\b)|(?:\band\b)", "", residue, flags=re.IGNORECASE)
    if _parse_range(msg) and not residue.strip():
        return False
    return True


def _match(text: str, options: list[str]) -> str | None:
    tl = text.strip().lower()
    for opt in options:
        if opt.lower() == tl:
            return opt
    # EXACT match only. Non-exact (typed/free-text) input is left for the LLM
    # (Haiku) extractor — no substring guessing, per the routing rule
    # "exact predefined match → programmatic; everything else → Haiku".
    return None


def _hist(state: SessionState, msg: str, reply: str) -> list:
    return state.history + [
        {"role": "user",      "content": msg},
        {"role": "assistant", "content": reply},
    ]


def _resp(state, reply, qr, flow_state, customer_message, status="ok", update=None):
    ns = state.model_copy(update={**(update or {}), "flow_state": flow_state,
                                  "history": _hist(state, customer_message, reply)})
    save_session(state.conversation_id, ns)
    return (InternalMessageResponse(reply_message=reply, quick_reply_options=qr,
                                    flow_state=flow_state, status=status), ns)


def _options_for_state(fs: str, cd: dict) -> list[str]:
    if fs == "segment_selection":
        return SEGMENTS
    if fs == "order_type_selection":
        return ORDER_TYPES
    if fs == "today_order_ask":
        return ["Yes", "No"]
    return []


def _msg_confirm_history(segment: str, ranges: list[dict]) -> str:
    if len(ranges) == 1:
        return (f"I'll email your {segment} order history for {ranges[0]['start']} "
                f"to {ranges[0]['end']}.\n\nShall I generate it?")
    body = "\n".join(f"• {r['start']} to {r['end']}" for r in ranges)
    return (f"I'll email your {segment} order history for these periods:\n\n"
            f"{body}\n\nShall I generate them?")


def _send_history_ranges(state, segment, ranges, customer_message):
    """Order History across MULTIPLE date ranges → one email per range."""
    from src.gateways.order_api import send_order_history_email
    ok, no_data, errored = [], [], []
    for r in ranges:
        start, end = r["start"], r["end"]
        result = None
        try:
            from src.core.langchain_agent import run_tool
            result = run_tool("send_order_history_email",
                              sub_account_id=state.sub_account_id or "",
                              segment=segment, date_str=start, end_date=end)
        except Exception as exc:
            logger.warning("[ORDER_STATUS] multi-range tool path failed: %s — direct", exc)
        if result is None:
            result = send_order_history_email(state.sub_account_id or "", segment, start, end)
        label = f"{start} to {end}"
        if result.get("success"):
            ok.append(label)
        elif result.get("no_data"):
            no_data.append(label)
        else:
            errored.append(label)
    lines = []
    if ok:
        lines.append("We've emailed your order history for:\n" + "\n".join(f"• {l}" for l in ok))
    if no_data:
        lines.append("No order history found for:\n" + "\n".join(f"• {l}" for l in no_data))
    if errored:
        lines.append("We couldn't process these period(s) now (please retry):\n"
                     + "\n".join(f"• {l}" for l in errored))
    reply = "\n\n".join(lines) if lines else (
        "We were unable to send your order history at this time. "
        "Please try again later or contact support: 📞 022-40508080 / 022-61480808")
    status = "ok" if ok or no_data else "error"
    return _resp(state, reply, ["Go back to main menu", "End Chat"],
                 "session_end_response", customer_message, status=status)


def _try_slot_fill(state: SessionState, customer_message: str, step_options: list[str]):
    """Free-text pre-pass for order status. Fills segment/order_type/scrip/ranges
    and jumps ahead. Gated to genuine free text (buttons/picker posts skip)."""
    if not _looks_like_free_text(customer_message, step_options):
        return None
    slots = _extract_slots(customer_message)
    if not slots:
        return None
    cd = dict(state.collected_data)
    if "segment" not in cd and slots.get("segment"):
        cd["segment"] = slots["segment"]
    if "order_type" not in cd and slots.get("order_type"):
        cd["order_type"] = "history" if slots["order_type"] == "Order History" else "today"
    if "scrip" not in cd and slots.get("scrip"):
        cd["scrip"] = slots["scrip"]
    if "ranges" not in cd and slots.get("ranges"):
        cd["ranges"] = slots["ranges"]

    seg = cd.get("segment")
    # Segment-active gate (thor-aligned): if the named segment isn't enabled on
    # the account, inform the customer instead of proceeding to the API.
    if seg and not _is_segment_active(state, seg):
        return _resp(state, _msg_segment_inactive(seg),
                     ["Go back to main menu", "End Chat"], "session_end_response",
                     customer_message, update={"flow": "order_status", "collected_data": cd})
    # Order History + ranges + segment → confirm & generate (multi-range).
    if cd.get("order_type") == "history" and cd.get("ranges") and seg:
        return _resp(state, _msg_confirm_history(seg, cd["ranges"]), ["Generate", "Change"],
                     "confirm_generate", customer_message,
                     update={"flow": "order_status", "collected_data": cd})
    # Order History + ranges but no segment → ask segment first.
    if cd.get("order_type") == "history" and cd.get("ranges") and not seg:
        return _resp(state, _msg_segment(), SEGMENTS, "segment_selection",
                     customer_message, update={"flow": "order_status", "collected_data": cd})
    # Order History + segment but NO ranges → go straight to the date picker.
    if cd.get("order_type") == "history" and seg and not cd.get("ranges"):
        return _resp(state, _msg_date_picker(), [], "order_date_selection",
                     customer_message, update={"flow": "order_status", "collected_data": cd})

    # ── Today Order Status branches ───────────────────────────────────────────
    # Today + segment + scrip → run the specific-order lookup now.
    if cd.get("order_type") == "today" and seg and cd.get("scrip"):
        st = state.model_copy(update={"flow": "order_status",
                                      "flow_state": "today_order_scrip",
                                      "collected_data": cd})
        save_session(state.conversation_id, st)
        return handle_order_status(st, cd["scrip"])
    # Today + segment, no scrip → ask whether a specific order.
    if cd.get("order_type") == "today" and seg and not cd.get("scrip"):
        return _resp(state, _msg_today_ask(), ["Yes", "No"], "today_order_ask",
                     customer_message, update={"flow": "order_status", "collected_data": cd})
    # Today/History known but NO segment → ask segment (type/scrip carried fwd).
    if cd.get("order_type") and not seg:
        return _resp(state, _msg_segment(), SEGMENTS, "segment_selection",
                     customer_message, update={"flow": "order_status", "collected_data": cd})

    # Segment known but no order type → ask order type.
    if seg and not cd.get("order_type"):
        return _resp(state, _msg_order_type(seg), ORDER_TYPES, "order_type_selection",
                     customer_message, update={"flow": "order_status", "collected_data": cd})
    return None


# ── Main handler ──────────────────────────────────────────────────────────────

def handle_order_status(
    state: SessionState,
    customer_message: str,
) -> tuple[InternalMessageResponse, SessionState]:

    fs = state.flow_state
    cd = state.collected_data

    # ── CONFIRM & GENERATE (Order History, multi-range) ───────────────────────
    if fs == "confirm_generate":
        low = customer_message.strip().lower()
        if low in ("change", "no", "edit"):
            return _resp(state, _msg_date_picker(), [], "order_date_selection", customer_message)
        seg = cd.get("segment", "Equity")
        ranges = cd.get("ranges", [])
        if not ranges:
            return _resp(state, _msg_date_picker(), [], "order_date_selection", customer_message)
        return _send_history_ranges(state, seg, ranges, customer_message)

    # ── Free-text slot-fill pre-pass (gated to free text) ─────────────────────
    if fs not in ("start", "session_end_response", "confirm_generate", "today_order_scrip"):
        short = _try_slot_fill(state, customer_message, _options_for_state(fs, cd))
        if short is not None:
            return short

    # ── START: account check ──────────────────────────────────────────────────
    if fs == "start":
        from src.core.langchain_agent import get_profile
        try:
            profile = get_profile(state.sub_account_id or "")
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

        # Free-text slot-fill at entry: first message may already name segment /
        # order type / scrip / date ranges → skip straight ahead.
        short = _try_slot_fill(
            state.model_copy(update={"flow": "order_status"}),
            customer_message, step_options=[])
        if short is not None:
            return short

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

        cd2 = {**state.collected_data, "segment": segment}

        # ── Segment-active gate (thor-aligned) ────────────────────────────────
        # If this segment isn't enabled on the customer's account, they can't
        # have orders in it — don't call the trade-book API (it would 400 / 500).
        if not _is_segment_active(state, segment):
            reply = _msg_segment_inactive(segment)
            ns = state.model_copy(update={
                "flow_state": "session_end_response",
                "collected_data": cd2,
                "history": _hist(state, customer_message, reply),
            })
            save_session(state.conversation_id, ns)
            logger.info("[ORDER_STATUS] conv=%s segment=%s not active → informing customer",
                        state.conversation_id, segment)
            return (InternalMessageResponse(
                reply_message=reply,
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state="session_end_response", status="ok"), ns)

        # If the customer ALREADY told us the order type (e.g. "order history
        # for last month" captured by slot-fill), do NOT re-ask it. Resume at
        # the right next step for that order type.
        known_type = cd2.get("order_type")
        if known_type == "history":
            ranges = cd2.get("ranges") or []
            if ranges:
                # We also have the period → confirm & generate directly.
                reply = _msg_confirm_history(segment, ranges)
                ns = state.model_copy(update={
                    "flow_state": "confirm_generate", "collected_data": cd2,
                    "history": _hist(state, customer_message, reply),
                })
                save_session(state.conversation_id, ns)
                return (InternalMessageResponse(reply_message=reply,
                        quick_reply_options=["Generate", "Change"],
                        flow_state="confirm_generate", status="ok"), ns)
            # History but no period yet → go straight to the date picker.
            reply = _msg_date_picker()
            ns = state.model_copy(update={
                "flow_state": "order_date_selection", "collected_data": cd2,
                "history": _hist(state, customer_message, reply),
            })
            save_session(state.conversation_id, ns)
            return (InternalMessageResponse(reply_message=reply, quick_reply_options=[],
                    flow_state="order_date_selection", status="ok"), ns)
        if known_type == "today":
            reply = _msg_today_ask()
            ns = state.model_copy(update={
                "flow_state": "today_order_ask", "collected_data": cd2,
                "history": _hist(state, customer_message, reply),
            })
            save_session(state.conversation_id, ns)
            return (InternalMessageResponse(reply_message=reply, quick_reply_options=["Yes", "No"],
                    flow_state="today_order_ask", status="ok"), ns)

        # Order type not yet known → ask it.
        reply = _msg_order_type(segment)
        ns = state.model_copy(update={
            "flow_state": "order_type_selection",
            "collected_data": cd2,
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
            # 30-day calendar range picker (UI keys off flow_state, no chips).
            reply = _msg_date_picker()
            ns = state.model_copy(update={
                "flow_state": "order_date_selection",
                "collected_data": {**state.collected_data, "order_type": "history"},
                "history": _hist(state, customer_message, reply),
            })
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply, quick_reply_options=[],
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
            result = None
            try:
                from src.core.langchain_agent import run_tool
                result = run_tool("send_order_history_email",
                                  sub_account_id=state.sub_account_id or "",
                                  segment=segment, date_str=today_str)
            except Exception as exc:
                logger.warning("[ORDER_STATUS] agent tool path failed: %s — direct fallback", exc)
            if result is None:
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
        # Agent-orchestrated tool call; direct fallback if the agent path fails.
        order_data = None
        try:
            from src.core.langchain_agent import run_tool
            order_data = run_tool("get_todays_orders",
                                  sub_account_id=state.sub_account_id or "", segment=segment)
        except Exception as exc:
            logger.warning("[ORDER_STATUS] agent tool path failed: %s — direct fallback", exc)
        if order_data is None:
            order_data = get_todays_orders(state.sub_account_id or "", segment)

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

    # ── ORDER HISTORY: 30-day calendar range ──────────────────────────────────
    if fs == "order_date_selection":
        segment = state.collected_data.get("segment", "Equity")

        rng = _parse_range(customer_message)
        if not rng:
            reply = _msg_date_picker()
            ns = state.model_copy(update={"history": _hist(state, customer_message, reply)})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply, quick_reply_options=[],
                    flow_state="order_date_selection", status="reprompt",
                ), ns,
            )
        d1, d2 = rng
        if (d2 - d1).days > 30 or d2 > date.today():
            reply = _msg_range_wrong()
            ns = state.model_copy(update={"history": _hist(state, customer_message, reply)})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply, quick_reply_options=[],
                    flow_state="order_date_selection", status="reprompt",
                ), ns,
            )
        start_date = d1.strftime("%d-%m-%Y")
        end_date   = d2.strftime("%d-%m-%Y")

        from src.gateways.order_api import send_order_history_email
        result = None
        try:
            from src.core.langchain_agent import run_tool
            result = run_tool("send_order_history_email",
                              sub_account_id=state.sub_account_id or "",
                              segment=segment, date_str=start_date, end_date=end_date)
        except Exception as exc:
            logger.warning("[ORDER_STATUS] agent tool path failed: %s — direct fallback", exc)
        if result is None:
            result = send_order_history_email(state.sub_account_id or "", segment, start_date, end_date)
        if not result.get("success", False):
            if result.get("no_data"):
                _err = ("No order history was found for the selected date range. "
                        "Please try a different period.")
                _status = "ok"
            else:
                _err = (
                    "We were unable to send your order history at this time.\n\n"
                    "Please try again later or contact support: 📞 022-40508080 / 022-61480808"
                )
                _status = "error"
            ns = state.model_copy(update={
                "flow_state": "session_end_response",
                "history": _hist(state, customer_message, _err),
            })
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=_err,
                    quick_reply_options=["Go back to main menu", "End Chat"],
                    flow_state="session_end_response", status=_status,
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
