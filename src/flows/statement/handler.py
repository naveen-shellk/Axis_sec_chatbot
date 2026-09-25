"""
chatbot_web/src/flows/statement/handler.py
-------------------------------------------
Statement flow — post-login, requires sub_account_id.

Optimised pattern:
  - LLM handles customer INPUT (extracts selection from free text)
  - Response messages and quick_replies are HARDCODED (no LLM for output)
  - LLM only called when regex/exact match fails for free text input

State machine:
  start → account status check → show category buttons
  statement_category → report list buttons
  report_type        → date picker buttons
  date_range_fy / date_range_aod / date_range_month → API call → hardcoded confirm
  session_end_response → shared
"""

from __future__ import annotations

import calendar
import logging
import re
from datetime import date, timedelta

from src.core.llm import call_intent_llm
from src.core.session_store import save_session
from src.shared.session_end import handle_session_end
from models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

# ── Report catalogue ──────────────────────────────────────────────────────────

TOP_CATEGORIES = ["Tax Reports", "Demat Reports", "Trading Reports"]

REPORTS_BY_CATEGORY = {
    "Tax Reports": [
        {"name": "Tax Statement",    "jobname": "Tax Statement",    "endpoint": "exports",  "date_input": "fy_picker"},
        {"name": "P&L Statement",    "jobname": "P&L Statement",    "endpoint": "exports",  "date_input": "fy_picker"},
        {"name": "Capital Gains",    "jobname": "Capital Gains",    "endpoint": "exports",  "date_input": "fy_picker"},
    ],
    "Demat Reports": [
        {"name": "DP Holdings",              "jobname": "DP Holdings",  "endpoint": "exports",  "date_input": "as_on_date"},
        {"name": "DP Transaction Statement", "jobname": "DP",           "endpoint": "sendmail", "date_input": "month_picker"},
        {"name": "CML Report (NSDL)",        "jobname": "NSDLCML",      "endpoint": "sendmail", "date_input": "month_picker"},
    ],
    "Trading Reports": [
        {"name": "Ledger Report",            "jobname": "Ledger Report",        "endpoint": "exports",  "date_input": "fy_picker"},
        {"name": "Contract Notes",           "jobname": "CommonContractNotes",  "endpoint": "sendmail", "date_input": "month_picker"},
        {"name": "Account Statement",        "jobname": "Account Statement",    "endpoint": "exports",  "date_input": "fy_picker"},
        {"name": "Global Statement (AGTS)",  "jobname": "AGTS",                 "endpoint": "sendmail", "date_input": "month_picker"},
        {"name": "Equity Margin",            "jobname": "EquityMargin",         "endpoint": "sendmail", "date_input": "month_picker"},
    ],
}

MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
YEAR_OPTIONS = [str(date.today().year - i) for i in range(3)]

ALL_REPORT_NAMES = [r["name"] for cat in REPORTS_BY_CATEGORY.values() for r in cat]


def _get_report(name: str) -> dict | None:
    for reports in REPORTS_BY_CATEGORY.values():
        for r in reports:
            if r["name"].lower() == name.lower():
                return r
    return None


def _financial_years() -> list[str]:
    t = date.today()
    base = t.year if t.month >= 4 else t.year - 1
    return [f"FY {base-i}-{str(base-i+1)[-2:]}" for i in range(3)]


def _resolve_dates(label: str) -> tuple[str, str]:
    today = date.today()
    fy = re.match(r"FY\s*(\d{4})-(\d{2})", label, re.IGNORECASE)
    if fy:
        y = int(fy.group(1))
        return f"01-04-{y}", f"31-03-{y+1}"
    iso = re.match(r"(\d{4})-(\d{2})-(\d{2})", label)
    if iso:
        d = f"{iso.group(3)}-{iso.group(2)}-{iso.group(1)}"
        return d, d
    if re.match(r"^\d{2}-\d{2}-\d{4}$", label):
        return label, label
    s = today - timedelta(days=90)
    return s.strftime("%d-%m-%Y"), today.strftime("%d-%m-%Y")


# ── Hardcoded response messages ───────────────────────────────────────────────

def _msg_category() -> str:
    return "Which type of statement do you need?"

def _msg_reports(category: str) -> str:
    return f"Please select a report from {category}:"

def _msg_fy() -> str:
    return "Please select a financial year:"

def _msg_month_year() -> str:
    return "Please select the year:"

def _msg_month_name() -> str:
    return "Please select the month:"

def _msg_aod(today_str: str) -> str:
    return f"Please select the date (default: {today_str}):"

def _msg_confirm(report_name: str, masked_email: str) -> str:
    return (
        f"We have shared the requested {report_name} on your registered "
        f"email {masked_email}. You will receive it shortly."
    )

def _msg_deactivated() -> str:
    return (
        "Your account is currently deactivated. Please contact our support team "
        "to reactivate your account before requesting statements.\n\n"
        "📞 Customer Care: 022-40508080 / 022-61480808"
    )

def _msg_reprompt_category() -> str:
    return "I didn't catch that. Please select a statement category:"

def _msg_reprompt_report(reports: list[str]) -> str:
    return f"Please choose one of the available reports:"

def _msg_reprompt_fy() -> str:
    return "Please select a valid financial year from the options:"

def _msg_reprompt_month() -> str:
    return "Please select from the options shown:"


# ── LLM extraction (input only — extracts selection from free text) ───────────

_EXTRACT_CATEGORY_SYS = """\
The customer is choosing a statement category on Axis Direct.
Options: "Tax Reports", "Demat Reports", "Trading Reports"
Extract which category the customer meant from their message.
Return JSON only:
{"selected": "<exact option name or null>", "reasoning": "<one line>"}
"""

_EXTRACT_REPORT_SYS = """\
The customer is selecting a statement report type on Axis Direct.
Extract which report name they meant. Available reports are in context.
Return JSON only:
{"selected": "<exact report name or null>", "reasoning": "<one line>"}
"""

_EXTRACT_FY_SYS = """\
The customer is selecting a financial year on Axis Direct.
Available options are in context under "options".
Extract which financial year they meant (e.g. "FY 2025-26", "last year", "this year").
Return JSON only:
{"selected": "<exact FY label or null>", "reasoning": "<one line>"}
"""

_EXTRACT_DATE_SYS = """\
The customer is selecting a month/date for their statement on Axis Direct.
Extract which month name or date they mentioned.
Return JSON only:
{"selected_month": "<month name or null>", "selected_year": "<4-digit year or null>", "selected_date": "<DD-MM-YYYY or null>", "reasoning": "<one line>"}
"""


def _extract_via_llm(system: str, message: str, context: dict | None = None) -> dict:
    """Call Haiku to extract selection from free-text input. Returns parsed dict."""
    ctx = ""
    if context:
        ctx = "\n".join(f"{k}: {v}" for k, v in context.items())
    full_msg = f"{ctx}\n\nCustomer message: {message}" if ctx else f"Customer message: {message}"
    result = call_intent_llm(system, [{"role": "user", "content": [{"text": full_msg}]}])
    return result.get("parsed") or {}


def _match_exact(text: str, options: list[str]) -> str | None:
    """Exact or case-insensitive match against a list of options."""
    tl = text.strip().lower()
    for opt in options:
        if opt.lower() == tl:
            return opt
    # partial match
    for opt in options:
        if opt.lower() in tl or tl in opt.lower():
            return opt
    return None


def _save_hist(state: SessionState, msg: str, reply: str) -> list:
    return state.history + [
        {"role": "user",      "content": msg},
        {"role": "assistant", "content": reply},
    ]


def _send_and_confirm(
    state: SessionState,
    report: dict,
    start_date: str,
    end_date: str,
    customer_message: str,
) -> tuple[InternalMessageResponse, SessionState]:
    """Call statement API — show confirmation only on success, error message on failure."""
    from src.gateways.statement_api import request_statement_fireandforget

    try:
        result = request_statement_fireandforget(
            sub_account_id=state.sub_account_id or "",
            api_jobname=report["jobname"],
            endpoint=report["endpoint"],
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        logger.error("[STATEMENT] API call failed: %s", exc)
        reply = (
            "We were unable to process your request at this time. "
            "Please try again later or contact support at 022-40508080."
        )
        hist  = _save_hist(state, customer_message, reply)
        ns = state.model_copy(update={"flow_state": "session_end_response", "history": hist})
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=reply,
                quick_reply_options=["Go back to main menu", "End Chat"],
                flow_state="session_end_response",
                status="error",
            ),
            ns,
        )

    if not result.success:
        reply = (
            "We were unable to process your request at this time. "
            "Please try again later or contact support at 022-40508080."
        )
    else:
        masked = result.masked_email or "your registered email"
        reply  = _msg_confirm(report["name"], masked)

    hist  = _save_hist(state, customer_message, reply)
    ns = state.model_copy(update={"flow_state": "session_end_response", "history": hist})
    save_session(state.conversation_id, ns)
    return (
        InternalMessageResponse(
            reply_message=reply,
            quick_reply_options=["Go back to main menu", "End Chat"],
            flow_state="session_end_response",
            status="ok" if result.success else "error",
        ),
        ns,
    )


# ── Main handler ──────────────────────────────────────────────────────────────

def handle_statement(
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
            hist  = _save_hist(state, customer_message, reply)
            ns = state.model_copy(update={"flow_state": "session_end_response", "history": hist})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply,
                    quick_reply_options=["Go back to main menu", "End Chat"],
                    flow_state="session_end_response",
                    status="end",
                ),
                ns,
            )

        reply = _msg_category()
        hist  = _save_hist(state, customer_message, reply)
        ns = state.model_copy(update={
            "flow": "statement", "flow_state": "statement_category", "history": hist,
        })
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=reply,
                quick_reply_options=TOP_CATEGORIES,
                flow_state="statement_category",
                status="ok",
            ),
            ns,
        )

    # ── CATEGORY SELECTION ────────────────────────────────────────────────────
    if fs == "statement_category":
        # 1. Try exact/partial match first
        cat = _match_exact(customer_message, TOP_CATEGORIES)

        # 2. LLM extraction for free text
        if not cat:
            parsed = _extract_via_llm(_EXTRACT_CATEGORY_SYS, customer_message)
            cat = parsed.get("selected")

        if not cat or cat not in TOP_CATEGORIES:
            reply = _msg_reprompt_category()
            hist  = _save_hist(state, customer_message, reply)
            ns = state.model_copy(update={"history": hist})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply,
                    quick_reply_options=TOP_CATEGORIES,
                    flow_state="statement_category",
                    status="reprompt",
                ),
                ns,
            )

        reports = [r["name"] for r in REPORTS_BY_CATEGORY[cat]]
        reply = _msg_reports(cat)
        hist  = _save_hist(state, customer_message, reply)
        ns = state.model_copy(update={
            "flow_state": "report_type",
            "collected_data": {**state.collected_data, "category": cat},
            "history": hist,
        })
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=reply,
                quick_reply_options=reports,
                flow_state="report_type",
                status="ok",
            ),
            ns,
        )

    # ── REPORT TYPE SELECTION ─────────────────────────────────────────────────
    if fs == "report_type":
        cat     = state.collected_data.get("category", "")
        reports = [r["name"] for r in REPORTS_BY_CATEGORY.get(cat, [])]

        # 1. Exact match
        report_name = _match_exact(customer_message, reports)

        # 2. LLM extraction
        if not report_name:
            parsed = _extract_via_llm(
                _EXTRACT_REPORT_SYS, customer_message, {"available_reports": ", ".join(reports)}
            )
            report_name = parsed.get("selected")

        report = _get_report(report_name) if report_name else None

        if not report:
            reply = _msg_reprompt_report(reports)
            hist  = _save_hist(state, customer_message, reply)
            ns = state.model_copy(update={"history": hist})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply,
                    quick_reply_options=reports,
                    flow_state="report_type",
                    status="reprompt",
                ),
                ns,
            )

        ns = state.model_copy(update={
            "collected_data": {**state.collected_data, "report_name": report["name"]},
        })
        return _advance_to_date(ns, report, customer_message)

    # ── FY PICKER ─────────────────────────────────────────────────────────────
    if fs == "date_range_fy":
        fy_opts = _financial_years()

        # 1. Exact match
        label = _match_exact(customer_message, fy_opts)

        # 2. LLM extraction
        if not label:
            parsed = _extract_via_llm(
                _EXTRACT_FY_SYS, customer_message, {"options": ", ".join(fy_opts)}
            )
            label = parsed.get("selected")
            if label and label not in fy_opts:
                label = _match_exact(label, fy_opts)

        if not label:
            reply = _msg_reprompt_fy()
            hist  = _save_hist(state, customer_message, reply)
            ns = state.model_copy(update={"history": hist})
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply,
                    quick_reply_options=fy_opts,
                    flow_state="date_range_fy",
                    status="reprompt",
                ),
                ns,
            )

        start, end = _resolve_dates(label)
        report = _get_report(state.collected_data.get("report_name", ""))
        return _send_and_confirm(state, report or {}, start, end, customer_message)

    # ── AS-ON-DATE PICKER ─────────────────────────────────────────────────────
    if fs == "date_range_aod":
        today_str = date.today().strftime("%d-%m-%Y")

        # Any response = accept today's date (there's only one option)
        label = customer_message.strip() or today_str
        if not re.match(r"^\d{2}-\d{2}-\d{4}$", label):
            label = today_str

        start, end = _resolve_dates(label)
        report = _get_report(state.collected_data.get("report_name", ""))
        return _send_and_confirm(state, report or {}, start, end, customer_message)

    # ── MONTH PICKER ──────────────────────────────────────────────────────────
    if fs == "date_range_month":
        if "selected_year" not in state.collected_data:
            # Expecting year selection
            year = _match_exact(customer_message, YEAR_OPTIONS)
            if not year:
                parsed = _extract_via_llm(_EXTRACT_DATE_SYS, customer_message)
                year = parsed.get("selected_year")
                if year and year not in YEAR_OPTIONS:
                    year = None

            if not year:
                reply = _msg_reprompt_month()
                hist  = _save_hist(state, customer_message, reply)
                ns = state.model_copy(update={"history": hist})
                save_session(state.conversation_id, ns)
                return (
                    InternalMessageResponse(
                        reply_message=reply,
                        quick_reply_options=YEAR_OPTIONS,
                        flow_state="date_range_month",
                        status="reprompt",
                    ),
                    ns,
                )

            reply = _msg_month_name()
            hist  = _save_hist(state, customer_message, reply)
            ns = state.model_copy(update={
                "collected_data": {**state.collected_data, "selected_year": year},
                "history": hist,
            })
            save_session(state.conversation_id, ns)
            return (
                InternalMessageResponse(
                    reply_message=reply,
                    quick_reply_options=MONTH_NAMES,
                    flow_state="date_range_month",
                    status="ok",
                ),
                ns,
            )

        else:
            # Expecting month selection
            year = state.collected_data["selected_year"]
            month_name = _match_exact(customer_message, MONTH_NAMES)
            if not month_name:
                parsed = _extract_via_llm(_EXTRACT_DATE_SYS, customer_message)
                month_name = parsed.get("selected_month")
                if month_name:
                    month_name = _match_exact(month_name, MONTH_NAMES)

            if not month_name:
                reply = _msg_reprompt_month()
                hist  = _save_hist(state, customer_message, reply)
                ns = state.model_copy(update={"history": hist})
                save_session(state.conversation_id, ns)
                return (
                    InternalMessageResponse(
                        reply_message=reply,
                        quick_reply_options=MONTH_NAMES,
                        flow_state="date_range_month",
                        status="reprompt",
                    ),
                    ns,
                )

            m_num    = MONTH_NAMES.index(month_name) + 1
            last_day = calendar.monthrange(int(year), m_num)[1]
            start    = f"01-{m_num:02d}-{year}"
            end      = f"{last_day}-{m_num:02d}-{year}"
            report   = _get_report(state.collected_data.get("report_name", ""))
            return _send_and_confirm(state, report or {}, start, end, customer_message)

    if fs == "session_end_response":
        return handle_session_end(state, customer_message)

    logger.warning("[STATEMENT] unknown flow_state %r — reset", fs)
    return handle_statement(state.model_copy(update={"flow_state": "start"}), customer_message)


def _advance_to_date(
    state: SessionState,
    report: dict,
    customer_message: str,
) -> tuple[InternalMessageResponse, SessionState]:
    di = report.get("date_input", "fy_picker")

    if di == "fy_picker":
        fy_opts = _financial_years()
        reply = _msg_fy()
        hist  = _save_hist(state, customer_message, reply)
        ns = state.model_copy(update={"flow_state": "date_range_fy", "history": hist})
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=reply,
                quick_reply_options=fy_opts,
                flow_state="date_range_fy",
                status="ok",
            ),
            ns,
        )

    if di == "as_on_date":
        today_str = date.today().strftime("%d-%m-%Y")
        reply = _msg_aod(today_str)
        hist  = _save_hist(state, customer_message, reply)
        ns = state.model_copy(update={"flow_state": "date_range_aod", "history": hist})
        save_session(state.conversation_id, ns)
        return (
            InternalMessageResponse(
                reply_message=reply,
                quick_reply_options=[today_str],
                flow_state="date_range_aod",
                status="ok",
            ),
            ns,
        )

    # month_picker
    reply = _msg_month_year()
    hist  = _save_hist(state, customer_message, reply)
    ns = state.model_copy(update={"flow_state": "date_range_month", "history": hist})
    save_session(state.conversation_id, ns)
    return (
        InternalMessageResponse(
            reply_message=reply,
            quick_reply_options=YEAR_OPTIONS,
            flow_state="date_range_month",
            status="ok",
        ),
        ns,
    )
