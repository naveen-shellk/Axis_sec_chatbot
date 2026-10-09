"""
chatbot_web/src/flows/brokerage/handler.py
-------------------------------------------
Brokerage & Charges flow — post-login.

Optimised pattern:
  - LLM handles customer INPUT (extracts charges type, date from free text)
  - Response messages are HARDCODED
  - LLM only called when exact/partial match fails

State machine:
  start → account check → charges_type_selection
    → Trading Charges → fetch ledger → hardcoded charges display
    → DP Charges      → dp_date_selection → send DP bill → hardcoded confirm
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

_DP_CHARGES_LINK = "https://www.axisdirect.in/charges"
CHARGES_TYPES    = ["Trading Charges", "DP Charges"]


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

def _msg_charges_type() -> str:
    return "Which type of charges information do you need?"

def _msg_trading_charges(closing: str, opening: str, masked_email: str) -> str:
    balance = float(closing) if closing else 0.0
    if balance > 0:
        return (
            f"Your current ledger balance:\n\n"
            f"• Opening Balance: ₹{opening}\n"
            f"• Closing Balance: ₹{closing}\n\n"
            f"You have outstanding charges. A detailed statement has been "
            f"sent to {masked_email}."
        )
    return (
        f"Your current ledger balance:\n\n"
        f"• Opening Balance: ₹{opening}\n"
        f"• Closing Balance: ₹{closing}\n\n"
        f"No outstanding charges at this time."
    )

def _msg_dp_date() -> str:
    return ("Please select a date range within 30 days from the calendar "
            "(DD-MM-YYYY to DD-MM-YYYY) for your DP charges statement.")

def _msg_dp_range_wrong() -> str:
    return "Please choose a date range within the last 30 days."

def _msg_dp_sent(masked_email: str) -> str:
    return (
        f"Your DP charges statement has been sent to {masked_email}.\n\n"
        f"You can also view the complete DP charges schedule here:\n"
        f"🔗 {_DP_CHARGES_LINK}"
    )

def _msg_deactivated() -> str:
    return (
        "Your account is currently deactivated. Please contact our support team "
        "to reactivate.\n\n📞 Customer Care: 022-40508080 / 022-61480808"
    )

def _msg_reprompt() -> str:
    return "I didn't catch that. Please select from the options:"


# ── LLM extraction (input only) ───────────────────────────────────────────────

_EXTRACT_CHARGES_SYS = """\
The customer is choosing between "Trading Charges" and "DP Charges" on Axis Direct.
Extract their choice from the message.
Return JSON only:
{"selected": "<Trading Charges|DP Charges|null>", "reasoning": "<one line>"}
"""

_EXTRACT_DATE_SYS = """\
The customer is selecting a date (DD-MM-YYYY format) for a DP charges statement on Axis Direct.
Available dates are in context. Extract the date they meant.
Return JSON only:
{"selected": "<DD-MM-YYYY or null>", "reasoning": "<one line>"}
"""


def _extract(system: str, message: str, context: dict | None = None) -> dict:
    ctx  = "\n".join(f"{k}: {v}" for k, v in (context or {}).items())
    full = f"{ctx}\n\nCustomer: {message}" if ctx else f"Customer: {message}"
    result = call_intent_llm(system, [{"role": "user", "content": [{"text": full}]}])
    return result.get("parsed") or {}


# ── Slot-filling: pull charges type + date range(s) from one free-text message ─
_EXTRACT_SLOTS_SYS = """\
You extract charges-request details from an Axis Direct customer's message.

Charges types:
  "Trading Charges" — brokerage / ledger balance / outstanding / why debited (NO date needed)
  "DP Charges"      — demat / DP / AMC / depository charges bill (uses a date range)

Return JSON ONLY:
{
  "charges_type": "Trading Charges | DP Charges | null",
  "ranges": [ {"start": "DD-MM-YYYY", "end": "DD-MM-YYYY"} ]
}

Rules:
- Convert ANY date to DD-MM-YYYY. Relative dates ("last month", "past 30 days")
  are resolved RELATIVE TO TODAY (today's date is given in the message context).
- SCOPE: extract ONLY the date period(s) the customer tied to their DP CHARGES.
  If the message ALSO asks about other topics (statements, order history),
  IGNORE the dates that belong to those other topics — do NOT put them in
  "ranges".
- Date range(s) only apply to DP Charges; "ranges" may hold multiple for the
  SAME DP-charges request; [] if none.
- NEVER HALLUCINATE DATES. Put a range in "ranges" ONLY when the customer gave
  an EXPLICIT period (e.g. "01-01-2026 to 25-01-2026", "last month"). If no
  period was given, or it is vague/unresolvable, return ranges = []. When
  unsure, prefer [] over a guessed date — the flow will ask for the dates.
- Only fill a field the customer actually implied; else null / [].
- Do NOT invent values.
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
    if parsed.get("charges_type") in CHARGES_TYPES:
        out["charges_type"] = parsed["charges_type"]
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


def _is_generic_brokerage_trigger(message: str) -> bool:
    """True when the FIRST message is only a generic entry phrase for this flow
    ('Brokerage and Charges' button, 'brokerage', 'charges', 'show my charges',
    etc.) with no specific Trading/DP/AMC/date signal. Such messages must land
    on the Trading-vs-DP question, NOT auto-select a type.

    Returns False as soon as the message carries a SPECIFIC signal:
      - names DP / demat / depository / AMC / annual maintenance  (→ DP path)
      - names trading / ledger / outstanding / brokerage amount    (specific)
      - contains a date (DP date range)
    so a genuinely specific first message can still skip the question.
    """
    low = (message or "").strip().lower()
    if not low:
        return False
    # Any explicit sub-type / date signal → NOT generic (let slot-fill run).
    specific_signals = (
        "dp", "demat", "depository", "amc", "annual maintenance",
        "ledger", "outstanding", "debited", "trading charge",
    )
    if any(sig in low for sig in specific_signals):
        return False
    if re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", low):
        return False
    # Otherwise, treat it as a generic trigger if it's essentially just the
    # brokerage/charges entry wording (button label or a short "show my charges"
    # style phrase). Strip common filler and check what's left.
    stripped = re.sub(r"\b(and|my|the|me|show|see|check|view|get|want|to|please|i|of)\b",
                      "", low)
    stripped = re.sub(r"[^a-z]", "", stripped)
    return stripped in {"brokerage", "charges", "brokeragecharges", "chargesbrokerage"}


def _looks_like_free_text(message: str, step_options: list[str]) -> bool:
    msg = (message or "").strip()
    if not msg:
        return False
    low = msg.lower()
    if any(opt.lower() == low for opt in (step_options or [])):
        return False
    if low in {"generate", "change", "yes", "no",
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
    # (Haiku) extractor / agentic decision — no substring guessing, per the rule
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


def _msg_confirm_dp(ranges: list[dict]) -> str:
    if len(ranges) == 1:
        return (f"I'll email your DP charges statement for {ranges[0]['start']} "
                f"to {ranges[0]['end']}.\n\nShall I generate it?")
    body = "\n".join(f"• {r['start']} to {r['end']}" for r in ranges)
    return (f"I'll email your DP charges statements for these periods:\n\n"
            f"{body}\n\nShall I generate them?")


def _send_dp_ranges(state, ranges, customer_message):
    """DP charges bill across MULTIPLE date ranges → one email per range."""
    from src.gateways.statement_api import send_dp_bill
    from src.gateways.customer_api import mask_email

    masked = "your registered email"
    ok, no_data, errored = [], [], []
    for r in ranges:
        start, end = r["start"], r["end"]
        result = None
        try:
            from src.core.langchain_agent import run_tool
            result = run_tool("send_dp_bill", sub_account_id=state.sub_account_id or "",
                              start_date=start, end_date=end)
        except Exception as exc:
            logger.warning("[BROKERAGE] multi-range DP tool path failed: %s — direct", exc)
        if result is None:
            result = send_dp_bill(state.sub_account_id or "", start, end)
        if result.get("masked_email"):
            masked = result["masked_email"]
        label = f"{start} to {end}"
        if result.get("success"):
            ok.append(label)
        elif result.get("no_data"):
            no_data.append(label)
        else:
            errored.append(label)
    try:
        from src.core.langchain_agent import get_profile
        em = mask_email(get_profile(state.sub_account_id or "").registered_email)
        if em:
            masked = em
    except Exception:
        pass
    lines = []
    if ok:
        lines.append(f"We've emailed your DP charges statement(s) to {masked} for:\n"
                     + "\n".join(f"• {l}" for l in ok))
    if no_data:
        lines.append("No DP charges found for:\n" + "\n".join(f"• {l}" for l in no_data))
    if errored:
        lines.append("We couldn't process these period(s) now (please retry):\n"
                     + "\n".join(f"• {l}" for l in errored))
    reply = "\n\n".join(lines) if lines else (
        "We were unable to send your DP charges statement at this time. "
        "Please try again later or contact support: 📞 022-40508080 / 022-61480808")
    status = "ok" if ok or no_data else "error"
    return _resp(state, reply, ["Go back to main menu", "End Chat"],
                 "session_end_response", customer_message, status=status)


def _try_slot_fill(state: SessionState, customer_message: str, step_options: list[str]):
    """Free-text pre-pass for brokerage. Resolves charges_type (and DP ranges)
    from free text; routes to confirm (DP+ranges) or into the existing
    charges_type logic (Trading, or DP without dates). Gated to free text."""
    if not _looks_like_free_text(customer_message, step_options):
        return None
    slots = _extract_slots(customer_message)
    if not slots:
        return None
    cd = dict(state.collected_data)
    if "charges_type" not in cd and slots.get("charges_type"):
        cd["charges_type"] = slots["charges_type"]
    if "ranges" not in cd and slots.get("ranges"):
        cd["ranges"] = slots["ranges"]

    ct = cd.get("charges_type")
    if ct == "DP Charges" and cd.get("ranges"):
        return _resp(state, _msg_confirm_dp(cd["ranges"]), ["Generate", "Change"],
                     "confirm_generate", customer_message,
                     update={"flow": "brokerage", "collected_data": cd})
    if ct:
        # Type known (Trading, or DP without a range yet) → run the existing
        # charges_type_selection logic by re-entering with the resolved type.
        st = state.model_copy(update={"flow": "brokerage",
                                      "flow_state": "charges_type_selection",
                                      "collected_data": cd})
        return handle_brokerage(st, ct)
    return None


# ── Main handler ──────────────────────────────────────────────────────────────

def handle_brokerage(
    state: SessionState,
    customer_message: str,
) -> tuple[InternalMessageResponse, SessionState]:

    fs = state.flow_state
    cd = state.collected_data

    # ── CONFIRM & GENERATE (DP Charges, multi-range) ──────────────────────────
    if fs == "confirm_generate":
        low = customer_message.strip().lower()
        if low in ("change", "no", "edit"):
            return _resp(state, _msg_dp_date(), [], "dp_date_selection", customer_message)
        ranges = cd.get("ranges", [])
        if not ranges:
            return _resp(state, _msg_dp_date(), [], "dp_date_selection", customer_message)
        return _send_dp_ranges(state, ranges, customer_message)

    # ── Free-text slot-fill pre-pass (gated to free text) ─────────────────────
    if fs not in ("start", "session_end_response", "confirm_generate"):
        _opts = CHARGES_TYPES if fs == "charges_type_selection" else []
        short = _try_slot_fill(state, customer_message, _opts)
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
                "flow": "brokerage", "flow_state": "session_end_response",
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

        # Free-text slot-fill at entry: a FIRST message that already names a
        # SPECIFIC charges type (e.g. "my DP charges for last month") may skip
        # the type question. BUT the generic trigger phrases that merely OPEN
        # this flow ("Brokerage and Charges" button, "brokerage", "charges",
        # "show my charges") must NOT auto-pick Trading Charges — per the flow
        # the customer must be shown the Trading vs DP choice. So we only run
        # the entry slot-fill when the message is MORE than a bare trigger.
        if not _is_generic_brokerage_trigger(customer_message):
            short = _try_slot_fill(
                state.model_copy(update={"flow": "brokerage"}),
                customer_message, step_options=[])
            if short is not None:
                return short

        reply = _msg_charges_type()
        ns = state.model_copy(update={
            "flow": "brokerage", "flow_state": "charges_type_selection",
            "history": _hist(state, customer_message, reply),
        })
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=reply,
                quick_reply_options=CHARGES_TYPES,
                flow_state="charges_type_selection", status="ok",
            ), ns,
        )

    # ── CHARGES TYPE SELECTION ────────────────────────────────────────────────
    if fs == "charges_type_selection":
        # 1. Exact/partial match (button tap)
        charges_type = _match(customer_message, CHARGES_TYPES)

        # 2. LLM extraction for free text
        if not charges_type:
            parsed       = _extract(_EXTRACT_CHARGES_SYS, customer_message)
            charges_type = _match(parsed.get("selected", ""), CHARGES_TYPES)

        # 3. AGENTIC API DECISION — this is a genuine MULTI-API choice point:
        #    "Trading Charges" → ledger API, "DP Charges" → DP bill API. When the
        #    customer's free text is ambiguous and steps 1-2 couldn't resolve it,
        #    let the agent pick which API path to take. (Deterministic fallback:
        #    the first candidate if the agent can't decide.)
        if not charges_type:
            try:
                from src.core.langchain_agent import run_api_decision
                decision = run_api_decision(
                    flow="charges",
                    request_summary=customer_message,
                    candidates=[
                        {"id": "Trading Charges",
                         "when": "brokerage / trading / ledger balance / outstanding "
                                 "amount / why was I debited / account balance charges"},
                        {"id": "DP Charges",
                         "when": "demat / DP / AMC / annual maintenance / depository "
                                 "charges / demat account charges bill"},
                    ],
                )
                charges_type = _match(decision.get("choice", ""), CHARGES_TYPES)
                if charges_type:
                    logger.info("[BROKERAGE] agentic API decision → %s", charges_type)
            except Exception as exc:
                logger.warning("[BROKERAGE] agentic API decision failed: %s", exc)

        if not charges_type:
            reply = _msg_reprompt()
            ns = state.model_copy(update={"history": _hist(state, customer_message, reply)})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply, quick_reply_options=CHARGES_TYPES,
                    flow_state="charges_type_selection", status="reprompt",
                ), ns,
            )

        # ── Trading Charges: fetch ledger → hardcoded display ─────────────────
        if charges_type == "Trading Charges":
            from src.gateways.statement_api import get_ledger
            from src.gateways.customer_api import get_customer_profile, mask_email

            today_str = date.today().strftime("%d-%m-%Y")
            ledger    = None
            try:
                from src.core.langchain_agent import run_tool
                ledger = run_tool("get_ledger_balance",
                                  sub_account_id=state.sub_account_id or "",
                                  start_date=today_str, end_date=today_str)
            except Exception as exc:
                logger.warning("[BROKERAGE] agent tool path failed: %s — direct fallback", exc)
            if ledger is None:
                ledger = get_ledger(state.sub_account_id or "", today_str, today_str)

            if not ledger.get("success", False):
                _err = (
                    "We were unable to retrieve your charges information at this time.\n\n"
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

            try:
                from src.core.langchain_agent import get_profile
                profile = get_profile(state.sub_account_id or "")
                masked  = mask_email(profile.registered_email) or "your registered email"
            except Exception:
                masked = "your registered email"

            reply = _msg_trading_charges(
                closing=str(ledger.get("closing_balance", "0")),
                opening=str(ledger.get("opening_balance", "0")),
                masked_email=masked,
            )
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

        # ── DP Charges: 30-day calendar range picker (UI keys off flow_state) ──
        reply = _msg_dp_date()
        ns = state.model_copy(update={
            "flow_state": "dp_date_selection",
            "history": _hist(state, customer_message, reply),
        })
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=reply, quick_reply_options=[],
                flow_state="dp_date_selection", status="ok",
            ), ns,
        )

    # ── DP DATE SELECTION (30-day calendar range) ─────────────────────────────
    if fs == "dp_date_selection":
        rng = _parse_range(customer_message)
        if not rng:
            reply = _msg_dp_date()
            ns = state.model_copy(update={"history": _hist(state, customer_message, reply)})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply, quick_reply_options=[],
                    flow_state="dp_date_selection", status="reprompt",
                ), ns,
            )
        d1, d2 = rng
        if (d2 - d1).days > 30 or d2 > date.today():
            reply = _msg_dp_range_wrong()
            ns = state.model_copy(update={"history": _hist(state, customer_message, reply)})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply, quick_reply_options=[],
                    flow_state="dp_date_selection", status="reprompt",
                ), ns,
            )
        start_date = d1.strftime("%d-%m-%Y")
        end_date   = d2.strftime("%d-%m-%Y")

        from src.gateways.statement_api import send_dp_bill
        from src.gateways.customer_api import get_customer_profile, mask_email

        result = None
        try:
            from src.core.langchain_agent import run_tool
            result = run_tool("send_dp_bill",
                              sub_account_id=state.sub_account_id or "",
                              start_date=start_date, end_date=end_date)
        except Exception as exc:
            logger.warning("[BROKERAGE] agent tool path failed: %s — direct fallback", exc)
        if result is None:
            result = send_dp_bill(state.sub_account_id or "", start_date, end_date)

        if not result.get("success", False):
            # "No documents found" for the period is a data condition, not an
            # outage — show a clear no-data message instead of a system error.
            if result.get("no_data"):
                _err = ("No DP charges statement was found for the selected date "
                        "range. Please try a different period.")
                _status = "ok"
            else:
                _err = (
                    "We were unable to send your DP charges statement at this time.\n\n"
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

        try:
            from src.core.langchain_agent import get_profile
            profile = get_profile(state.sub_account_id or "")
            masked  = mask_email(profile.registered_email) or "your registered email"
        except Exception:
            masked = result.get("masked_email", "your registered email")

        reply = _msg_dp_sent(masked)
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

    logger.warning("[BROKERAGE] unknown flow_state %r — reset", fs)
    return handle_brokerage(state.model_copy(update={"flow_state": "start"}), customer_message)
