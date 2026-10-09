"""
chatbot_langchain/src/flows/statement/handler.py
-------------------------------------------
Statement flow — post-login, requires sub_account_id.

Implements the full statement spec:
  Tax Reports    → Tax Statement (FY), Pan Level Transaction Report (segment→FY),
                   Portfolio Holding Statement (FY)
  Demat Reports  → Demat Holding cum Transaction (month), CML Reports (direct),
                   DP Transaction Statement (month), Statement of DP Holding (date)
  Trading Reports (sub-tabs):
                   Contract Note → Common / Commodity / Commodity Physical (30-day range)
                   M2M → M2M (n/a) / Derivative Physical Bills (30-day range)
                   Daily Margin Report → Equity (n/a) / Commodity (30-day range)
                   Global Statement (Trade Summary) (FY)
                   Ledger Report (FY)
                   Trade book report (segment→30-day range)
                   Other Reports → SLBM / SOAROS / Retention

Date inputs:
  fy_picker        → financial-year buttons
  month_picker     → year → month
  single_date      → single date (default today)
  date_range_30    → 30-day date range (start,end DD-MM-YYYY)
  segment_then_fy  → market segment → FY
  segment_then_range → market segment → 30-day range
  cml_direct       → no picker, send immediately
  unavailable      → report not serviceable by the API (informational message)

Response messages + quick replies are HARDCODED; the LLM is used only to
extract a selection from free-text input when exact/regex match fails.
"""

from __future__ import annotations

import calendar
import logging
import re
from datetime import date, datetime, timedelta

from src.core.llm import call_intent_llm
from src.core.session_store import save_session
from src.shared.session_end import handle_session_end
from models import InternalMessageResponse, SessionState

logger = logging.getLogger(__name__)

# ── Report catalogue ──────────────────────────────────────────────────────────
TOP_CATEGORIES = ["Tax Reports", "Demat Reports", "Trading Reports"]

# Market segments (for segment-first reports).
SEGMENTS = ["Equity", "Commodity", "Derivatives", "Currency"]

# A report:
#   {"name", "jobname", "endpoint": exports|sendmail, "date_input", ["unavailable": True]}
# A sub-tab (Trading Reports only):
#   {"name", "subtab": True, "reports": [ ...report dicts... ]}
REPORTS_BY_CATEGORY: dict[str, list[dict]] = {
    "Tax Reports": [
        {"name": "Tax Statement",                "jobname": "Tax Statement",                   "endpoint": "exports",  "date_input": "fy_picker"},
        {"name": "Pan Level Transaction Report", "jobname": "Pan-Level Transaction Statement", "endpoint": "exports",  "date_input": "segment_then_fy"},
        {"name": "Portfolio Holding Statement",  "jobname": "Portfolio Holding Statement",     "endpoint": "exports",  "date_input": "fy_picker"},
    ],
    "Demat Reports": [
        # Email-automation alignment: "DP holding cum transaction statement" == the
        # email automation's "dp transaction statement" → jobname DP (send-mail).
        {"name": "Demat Holding cum Transaction Statement", "jobname": "DP", "endpoint": "sendmail", "date_input": "month_picker"},
        # Email-automation alignment: CML sends BOTH depositories (CDSL + NSDL).
        {"name": "CML Reports",                    "jobname": "CDSLCML,NSDLCML", "endpoint": "sendmail", "date_input": "cml_direct"},
        {"name": "DP Transaction Statement",       "jobname": "DP",      "endpoint": "sendmail", "date_input": "month_picker"},
        {"name": "Statement of DP Holding",        "jobname": "DP",      "endpoint": "sendmail", "date_input": "single_date"},
    ],
    "Trading Reports": [
        {"name": "Contract Note", "subtab": True, "reports": [
            {"name": "Common Contract Notes",              "jobname": "CommonContractNotes",           "endpoint": "sendmail", "date_input": "date_range_30"},
            {"name": "Commodity Contract Notes",           "jobname": "CommodityContractNotes",        "endpoint": "sendmail", "date_input": "date_range_30"},
            {"name": "Commodity Physical Delivery Contract Notes", "jobname": "CommodityContractNotesPhysical", "endpoint": "sendmail", "date_input": "date_range_30"},
        ]},
        {"name": "M2M", "subtab": True, "reports": [
            {"name": "M2M",                        "jobname": "",                        "endpoint": "sendmail", "date_input": "unavailable"},
            {"name": "Derivative Physical Bills",  "jobname": "DerivativePhysicalBills", "endpoint": "sendmail", "date_input": "date_range_30"},
        ]},
        {"name": "Daily Margin Report", "subtab": True, "reports": [
            # Equity Daily Margin IS serviceable (thor: "equity margin" ->
            # EquityMargin send-mail; prompt maps "equity daily margin / margin
            # statement"). Emailed for a date range, like Commodity Daily Margin.
            {"name": "Equity Daily Margin",    "jobname": "EquityMargin",         "endpoint": "sendmail", "date_input": "date_range_30"},
            {"name": "Commodity Daily Margin", "jobname": "CommodityDailyMargin", "endpoint": "sendmail", "date_input": "date_range_30"},
        ]},
        {"name": "Global Statement (Trade Summary)", "jobname": "AGTS",                 "endpoint": "sendmail", "date_input": "fy_picker"},
        {"name": "Ledger Report",                     "jobname": "Ledger Statement",     "endpoint": "exports",  "date_input": "fy_picker"},
        {"name": "Trade book report",                 "jobname": "Trade Book Statement", "endpoint": "exports",  "date_input": "segment_then_range"},
        {"name": "Other Reports", "subtab": True, "reports": [
            {"name": "SLBM",       "jobname": "SLBMContractNotes",  "endpoint": "sendmail", "date_input": "date_range_30"},
            {"name": "SOAROS",     "jobname": "SOAcumROS",          "endpoint": "sendmail", "date_input": "date_range_30"},
            {"name": "Retention",  "jobname": "RetentionStatement", "endpoint": "sendmail", "date_input": "month_picker"},
        ]},
    ],
}

MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
YEAR_OPTIONS = [str(date.today().year - i) for i in range(3)]


def _top_items(category: str) -> list[dict]:
    return REPORTS_BY_CATEGORY.get(category, [])


def _find_report(name: str) -> dict | None:
    """Find a leaf report by name anywhere in the catalogue (incl. sub-tabs)."""
    n = (name or "").strip().lower()
    for items in REPORTS_BY_CATEGORY.values():
        for it in items:
            if it.get("subtab"):
                for r in it["reports"]:
                    if r["name"].lower() == n:
                        return r
            elif it["name"].lower() == n:
                return it
    return None


def _find_subtab(category: str, name: str) -> dict | None:
    n = (name or "").strip().lower()
    for it in _top_items(category):
        if it.get("subtab") and it["name"].lower() == n:
            return it
    return None


def _financial_years() -> list[str]:
    t = date.today()
    base = t.year if t.month >= 4 else t.year - 1
    return [f"FY {base-i}-{str(base-i+1)[-2:]}" for i in range(3)]


def _clamp_future(d: date) -> date:
    today = date.today()
    return today if d > today else d


def _resolve_fy_dates(label: str) -> tuple[str, str]:
    fy = re.match(r"FY\s*(\d{4})-(\d{2})", label, re.IGNORECASE)
    if fy:
        y = int(fy.group(1))
        start = date(y, 4, 1)
        end   = _clamp_future(date(y + 1, 3, 31))
        return start.strftime("%d-%m-%Y"), end.strftime("%d-%m-%Y")
    today = date.today()
    return (today - timedelta(days=90)).strftime("%d-%m-%Y"), today.strftime("%d-%m-%Y")


def _parse_date(text: str) -> date | None:
    """Parse DD-MM-YYYY or DD/MM/YYYY."""
    m = re.search(r"\b(\d{2})[-/](\d{2})[-/](\d{4})\b", text)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    return None


def _parse_range(text: str) -> tuple[date, date] | None:
    """Extract two DD-MM-YYYY/DD-MM-YYYY dates from free text (start, end)."""
    dates = re.findall(r"\b(\d{2})[-/](\d{2})[-/](\d{4})\b", text)
    if len(dates) >= 2:
        try:
            d1 = date(int(dates[0][2]), int(dates[0][1]), int(dates[0][0]))
            d2 = date(int(dates[1][2]), int(dates[1][1]), int(dates[1][0]))
            return (d1, d2) if d1 <= d2 else (d2, d1)
        except ValueError:
            return None
    return None


# The 30-day range step renders as a CALENDAR range picker in the UI (the UI
# keys off the flow_state "date_range_30" via quickReplies.reference). No quick
# replies are sent; the picker posts "DD-MM-YYYY to DD-MM-YYYY", parsed below.
def _preset_range(label: str) -> tuple[date, date] | None:
    """Optional preset labels still accepted for convenience (Last 7/15/30 days)."""
    days = {"last 7 days": 7, "last 15 days": 15, "last 30 days": 30}.get(
        (label or "").strip().lower())
    if not days:
        return None
    today = date.today()
    return (today - timedelta(days=days - 1), today)


# ── Hardcoded messages (per spec) ─────────────────────────────────────────────
def _msg_category() -> str:
    return "Which type of statement do you need?"

def _msg_reports(category: str) -> str:
    return f"Please select a report from {category}:"

def _msg_subtab(name: str) -> str:
    return f"Please select a report under {name}:"

def _for(report_name: str) -> str:
    """' for your <report>' suffix so date prompts say what they're generating."""
    return f" for your {report_name}" if report_name else " for your statement"

def _msg_fy(report_name: str = "") -> str:
    return (f"Please select the financial year{_for(report_name)}.")

def _msg_segment(report_name: str = "") -> str:
    return (f"Please select the market segment{_for(report_name)}.")

def _msg_month_year(report_name: str = "") -> str:
    return (f"Please select the year{_for(report_name)}.")

def _msg_month_name(report_name: str = "") -> str:
    return (f"Please select the month{_for(report_name)}.")

def _msg_single_date(today_str: str, report_name: str = "") -> str:
    return (f"Please select a date{_for(report_name)} (default: {today_str}).")

def _msg_range(report_name: str = "") -> str:
    return (f"Please select a date range within 30 days{_for(report_name)} "
            "(DD-MM-YYYY to DD-MM-YYYY) from the calendar.")

def _msg_range_wrong() -> str:
    return "Please choose a date range within the 30 days from your request."

def _msg_cml_ack() -> str:
    return ("We have received your request for NSDL CML and it will be sent to "
            "your registered email ID shortly.")

def _msg_confirm(report_name: str, masked_email: str) -> str:
    return (f"We have shared the requested {report_name} on your registered "
            f"email {masked_email}. You will receive it shortly.")

def _msg_no_data_fy() -> str:
    return "Sorry, no statement was found for the requested financial year."

def _msg_no_data_month() -> str:
    return "Sorry, no statement was found for the selected month."

def _msg_no_data_date() -> str:
    return "Sorry, we have not found any statement for the selected date."

def _msg_unavailable(report_name: str) -> str:
    return (f"{report_name} is currently not available through chat. "
            "Please use the Axis Direct portal to download this report.")

def _msg_deactivated() -> str:
    return ("Your account is currently deactivated. Please contact our support team "
            "to reactivate your account before requesting statements.\n\n"
            "📞 Customer Care: 022-40508080 / 022-61480808")

_END_QR = ["Go back to main menu", "End Chat"]


# ── LLM extraction (free-text input only) ─────────────────────────────────────
_EXTRACT_CATEGORY_SYS = """\
The customer is choosing a statement category on Axis Direct.
Options: "Tax Reports", "Demat Reports", "Trading Reports"
Return JSON only: {"selected": "<exact option or null>"}
"""
_EXTRACT_PICK_SYS = """\
The customer is selecting an item from a menu on Axis Direct.
The available items are in context under "options".
Return JSON only: {"selected": "<exact item name or null>"}
"""
_EXTRACT_FY_SYS = """\
The customer is selecting a financial year on Axis Direct (options in context).
Return JSON only: {"selected": "<exact FY label or null>"}
"""
_EXTRACT_DATE_SYS = """\
The customer is selecting a month/year for their statement on Axis Direct.
Return JSON only: {"selected_month": "<month name or null>", "selected_year": "<4-digit year or null>"}
"""


def _extract(system: str, message: str, context: dict | None = None) -> dict:
    ctx = "\n".join(f"{k}: {v}" for k, v in (context or {}).items())
    full = f"{ctx}\n\nCustomer message: {message}" if ctx else f"Customer message: {message}"
    result = call_intent_llm(system, [{"role": "user", "content": [{"text": full}]}])
    return result.get("parsed") or {}


# ── Slot-filling: extract EVERY slot from a free-text message in one shot ──────
# Lets the customer say e.g. "Tax Statement for 01-04-2018 to 31-03-2025 and
# 01-04-2025 to 15-01-2026" and skip the step-by-step questions. Runs ONLY on
# free text (gated in handle_statement) so button/picker taps stay instant.
_ALL_REPORT_NAMES = [
    r["name"]
    for items in REPORTS_BY_CATEGORY.values()
    for it in items
    for r in (it["reports"] if it.get("subtab") else [it])
]

_EXTRACT_SLOTS_SYS = """\
You extract statement-request details from an Axis Direct customer's message.

Known report names (match loosely to the customer's wording, return the EXACT name):
%s

Market segments: Equity, Commodity, Derivatives, Currency

Return JSON ONLY:
{
  "report_name": "<exact report name from the list, or null>",
  "category": "Tax Reports | Demat Reports | Trading Reports | null",
  "segment": "<segment or null>",
  "ranges": [ {"start": "DD-MM-YYYY", "end": "DD-MM-YYYY"} ]
}

Rules:
- Convert ANY date the customer gives into DD-MM-YYYY (e.g. "1st April 2018" -> "01-04-2018").
- SCOPE: extract ONLY the report + the date period(s) the customer tied to THAT
  STATEMENT/REPORT. If the message ALSO asks about other topics (order history,
  charges, orders, holdings in another product), IGNORE the dates that belong to
  those other topics — do NOT put them in "ranges". Example: "ledger statement
  for FY 2024-25 and my order history for last month" -> report_name="Ledger
  Report", ranges=[FY 2024-25 only]; the "last month" date is for order history,
  NOT this statement, so EXCLUDE it.
- "ranges" may contain MULTIPLE periods only if the customer asked for the SAME
  report across several periods; [] if none.
- Relative dates ("last month", "this financial year", "past 30 days") are
  resolved RELATIVE TO TODAY (today's date is given in the message context).
- NEVER HALLUCINATE DATES. Put a range in "ranges" ONLY when the customer gave
  an EXPLICIT date/period (e.g. "01-04-2024 to 31-03-2025", "FY 2024-25",
  "January 2026", "last month"). If the customer named NO period for this
  report, return ranges = []. If a period is vague/unresolvable, return
  ranges = []. When unsure, prefer [] over a guessed date. An empty ranges is
  CORRECT and expected — the flow will then ask the customer for the dates.
- Only fill a field if the customer actually implied it; otherwise null / [].
- Do NOT invent dates, periods, reports, categories, or segments.
""" % ("\n".join(f"  - {n}" for n in _ALL_REPORT_NAMES))


def _norm_ddmmyyyy(s: str) -> str | None:
    """Normalise a date string to DD-MM-YYYY, or None if unparseable."""
    s = (s or "").strip()
    m = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", s)
    if m:
        try:
            d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            return d.strftime("%d-%m-%Y")
        except ValueError:
            return None
    return None


def _extract_slots(message: str) -> dict:
    """One Haiku call -> all statement slots found in free text."""
    _today = date.today().strftime("%d-%m-%Y")
    parsed = _extract(_EXTRACT_SLOTS_SYS, message, {"today": _today}) or {}
    out: dict = {}

    rn = parsed.get("report_name")
    if rn and _find_report(rn):
        out["report_name"] = _find_report(rn)["name"]  # canonical casing

    cat = parsed.get("category")
    if cat in TOP_CATEGORIES:
        out["category"] = cat

    seg = parsed.get("segment")
    if seg in SEGMENTS:
        out["segment"] = seg

    ranges = []
    for r in (parsed.get("ranges") or []):
        s = _norm_ddmmyyyy(str(r.get("start", "")))
        e = _norm_ddmmyyyy(str(r.get("end", "")))
        if s and e:
            ds = datetime.strptime(s, "%d-%m-%Y").date()
            de = datetime.strptime(e, "%d-%m-%Y").date()
            if ds > de:
                s, e = e, s
            ranges.append({"start": s, "end": e})
    if ranges:
        out["ranges"] = ranges

    return out


def _looks_like_free_text(message: str, step_options: list[str]) -> bool:
    """True when the message is NOT an exact button/option tap and NOT a bare
    calendar range post - i.e. genuine free text worth running extraction on.

    IMPORTANT: a full sentence that merely *contains* dates (e.g. "Tax Statement
    for 01-04-2018 to 31-03-2025 and ...") is still free text. Only a BARE
    date / "DD-MM-YYYY to DD-MM-YYYY" picker post (just dates + connectors) is
    treated as a picker tap and skipped."""
    msg = (message or "").strip()
    if not msg:
        return False
    low = msg.lower()
    if any(opt.lower() == low for opt in (step_options or [])):
        return False
    if low in {"generate", "change", "yes", "no", "go back to main menu",
               "main menu", "end chat"}:
        return False
    # Bare calendar post? Strip out dates + connectors; if nothing meaningful
    # remains, it's a picker tap, not a sentence.
    residue = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", "", msg)
    residue = re.sub(r"[\s,./\-]|(?:\bto\b)|(?:\band\b)", "", residue, flags=re.IGNORECASE)
    if (_parse_range(msg) or _parse_date(msg)) and not residue.strip():
        return False
    return True


def _match(text: str, options: list[str]) -> str | None:
    tl = text.strip().lower()
    for opt in options:
        if opt.lower() == tl:
            return opt
    # EXACT match only. Non-exact (typed/free-text) input is left for the LLM
    # (Haiku) extractor — we don't guess via substring, per the routing rule
    # "exact predefined match → programmatic; everything else → Haiku".
    return None


def _hist(state: SessionState, msg: str, reply: str) -> list:
    return state.history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": reply},
    ]


def _reply(state, reply, qr, flow_state, status="ok", update=None):
    ns = state.model_copy(update={**(update or {}), "flow_state": flow_state,
                                  "history": _hist(state, "", reply) if False else state.history})
    return ns  # placeholder — not used; helpers below build responses inline


# ── Send + confirm ─────────────────────────────────────────────────────────────
def _send_and_confirm(state, report, start_date, end_date, customer_message, no_data_msg):
    """Call statement API; confirm on success, show no-data/unavailable otherwise."""
    from src.gateways.statement_api import request_statement_fireandforget
    try:
        result = None
        try:
            from src.core.langchain_agent import run_tool
            from src.gateways.statement_api import StatementResult
            tool_out = run_tool(
                "request_statement",
                sub_account_id=state.sub_account_id or "",
                report_name=report["jobname"],
                start_date=start_date, end_date=end_date, endpoint=report["endpoint"],
            )
            if tool_out is not None:
                result = StatementResult(
                    success=bool(tool_out.get("success")),
                    masked_email=tool_out.get("masked_email", "") or "",
                    error_message=tool_out.get("error") or "",
                )
        except Exception as exc:
            logger.warning("[STATEMENT] agent tool path failed: %s — direct fallback", exc)
        if result is None:
            result = request_statement_fireandforget(
                sub_account_id=state.sub_account_id or "",
                api_jobname=report["jobname"], endpoint=report["endpoint"],
                start_date=start_date, end_date=end_date,
            )
    except Exception as exc:
        logger.error("[STATEMENT] API call failed: %s", exc)
        result = None

    if result is None:
        reply = ("We were unable to process your request at this time. "
                 "Please try again later or contact support at 022-40508080.")
        status = "error"
    elif result.success:
        masked = result.masked_email or "your registered email"
        reply = _msg_confirm(report["name"], masked)
        status = "ok"
    else:
        # "No documents found" → per-spec no-data message; other errors → generic.
        err = (result.error_message or "").lower()
        if "no documents" in err or "not found" in err or "502" in err:
            reply = no_data_msg
        else:
            reply = ("We were unable to process your request at this time. "
                     "Please try again later or contact support at 022-40508080.")
        status = "ok"

    ns = state.model_copy(update={"flow_state": "session_end_response",
                                  "history": _hist(state, customer_message, reply)})
    save_session(state.conversation_id, ns)
    return (InternalMessageResponse(reply_message=reply, quick_reply_options=_END_QR,
                                    flow_state="session_end_response", status=status), ns)


def _resp(state, reply, qr, flow_state, customer_message, status="ok", update=None):
    ns = state.model_copy(update={**(update or {}), "flow_state": flow_state,
                                  "history": _hist(state, customer_message, reply)})
    save_session(state.conversation_id, ns)
    return (InternalMessageResponse(reply_message=reply, quick_reply_options=qr,
                                    flow_state=flow_state, status=status), ns)


def _generate_ranges(state, report, ranges, customer_message):
    """Generate ONE report across MULTIPLE date ranges (literal dates, decision b).
    Fires the statement API once per range, then returns a single combined reply."""
    from src.gateways.statement_api import request_statement_fireandforget, StatementResult

    masked = "your registered email"
    ok, no_data, errored = [], [], []
    for r in ranges:
        start, end = r["start"], r["end"]
        result = None
        try:
            from src.core.langchain_agent import run_tool
            tool_out = run_tool(
                "request_statement",
                sub_account_id=state.sub_account_id or "",
                report_name=report["jobname"],
                start_date=start, end_date=end, endpoint=report["endpoint"],
            )
            if tool_out is not None:
                result = StatementResult(
                    success=bool(tool_out.get("success")),
                    masked_email=tool_out.get("masked_email", "") or "",
                    error_message=tool_out.get("error") or "",
                )
        except Exception as exc:
            logger.warning("[STATEMENT] multi-range tool path failed: %s — direct", exc)
        if result is None:
            result = request_statement_fireandforget(
                sub_account_id=state.sub_account_id or "",
                api_jobname=report["jobname"], endpoint=report["endpoint"],
                start_date=start, end_date=end,
            )
        if result and result.masked_email:
            masked = result.masked_email
        label = f"{start} to {end}"
        if result and result.success:
            ok.append(label)
        elif result and (("no documents" in (result.error_message or "").lower())
                          or ("not found" in (result.error_message or "").lower())
                          or ("404" in (result.error_message or ""))
                          or ("502" in (result.error_message or ""))):
            no_data.append(label)
        else:
            errored.append(label)

    lines = []
    if ok:
        lines.append(
            f"We've emailed your {report['name']} for the following period(s) "
            f"to {masked}:\n" + "\n".join(f"• {l}" for l in ok))
    if no_data:
        lines.append("No statement was found for:\n" + "\n".join(f"• {l}" for l in no_data))
    if errored:
        lines.append("We couldn't process these period(s) right now (please retry):\n"
                     + "\n".join(f"• {l}" for l in errored))
    reply = "\n\n".join(lines) if lines else (
        "We were unable to process your request at this time. "
        "Please try again later or contact support at 022-40508080.")
    status = "ok" if ok or no_data else "error"

    ns = state.model_copy(update={"flow_state": "session_end_response",
                                  "history": _hist(state, customer_message, reply)})
    save_session(state.conversation_id, ns)
    return (InternalMessageResponse(reply_message=reply, quick_reply_options=_END_QR,
                                    flow_state="session_end_response", status=status), ns)


def _options_for_state(fs: str, cd: dict) -> list[str]:
    """The quick-reply options a given step shows — used to detect an exact
    button tap (so the free-text slot-fill pre-pass can skip it)."""
    if fs == "statement_category":
        return TOP_CATEGORIES
    if fs == "report_type":
        return [it["name"] for it in _top_items(cd.get("category", ""))]
    if fs == "subtab_select":
        sub = _find_subtab(cd.get("category", ""), cd.get("subtab", ""))
        return [r["name"] for r in sub["reports"]] if sub else []
    if fs == "segment_select":
        return SEGMENTS
    if fs == "date_range_fy":
        return _financial_years()
    if fs == "date_range_month":
        return MONTH_NAMES if cd.get("selected_year") else YEAR_OPTIONS
    return []


def _ranges_summary(ranges: list[dict]) -> str:
    return "\n".join(f"• {r['start']} to {r['end']}" for r in ranges)


def _msg_confirm_generate(report_name: str, ranges: list[dict]) -> str:
    if len(ranges) == 1:
        return (f"I'll email your {report_name} for {ranges[0]['start']} to "
                f"{ranges[0]['end']}.\n\nShall I generate it?")
    return (f"I'll email your {report_name} for these periods:\n\n"
            f"{_ranges_summary(ranges)}\n\nShall I generate them?")


def _try_slot_fill(state: SessionState, customer_message: str, step_options: list[str]):
    """
    Free-text pre-pass. If the message carries slots (report/dates), merge them
    and, when enough is known, jump straight to confirm_generate (or the right
    date step). Returns a response tuple to short-circuit, or None to let the
    normal step logic run.
    Gated: only runs on genuine free text (not button/picker taps).
    """
    if not _looks_like_free_text(customer_message, step_options):
        return None

    slots = _extract_slots(customer_message)
    if not slots:
        return None

    cd = dict(state.collected_data)
    # Merge WITHOUT clobbering values already chosen in-flow.
    if "report_name" not in cd and slots.get("report_name"):
        cd["report_name"] = slots["report_name"]
        # derive category from the resolved report for consistency
        rep = _find_report(cd["report_name"])
        for c, items in REPORTS_BY_CATEGORY.items():
            for it in items:
                leaves = it["reports"] if it.get("subtab") else [it]
                if any(l["name"] == cd["report_name"] for l in leaves):
                    cd["category"] = c
    if "category" not in cd and slots.get("category"):
        cd["category"] = slots["category"]
    if "segment" not in cd and slots.get("segment"):
        cd["segment"] = slots["segment"]
    if "ranges" not in cd and slots.get("ranges"):
        cd["ranges"] = slots["ranges"]

    report = _find_report(cd.get("report_name", "")) if cd.get("report_name") else None

    # Enough to confirm? need a resolved report + at least one date range.
    if report and cd.get("ranges"):
        di = report.get("date_input", "")
        # segment-first reports still need a segment before generating
        if di in ("segment_then_fy", "segment_then_range") and not cd.get("segment"):
            return _resp(state, _msg_segment(cd.get("report_name", "")), SEGMENTS, "segment_select",
                         customer_message, update={"flow": "statement", "collected_data": cd})
        return _resp(state, _msg_confirm_generate(cd["report_name"], cd["ranges"]),
                     ["Generate", "Change"], "confirm_generate", customer_message,
                     update={"flow": "statement", "collected_data": cd})

    # Report known but NO dates → jump to that report's date step.
    if report and not cd.get("ranges"):
        return _advance_to_date(
            state.model_copy(update={"flow": "statement", "collected_data": cd}),
            report, customer_message)

    # Only category known → jump to its report list.
    if cd.get("category") and not report:
        names = [it["name"] for it in _top_items(cd["category"])]
        return _resp(state, _msg_reports(cd["category"]), names, "report_type",
                     customer_message, update={"flow": "statement", "collected_data": cd})

    return None


# ── Main handler ──────────────────────────────────────────────────────────────
def handle_statement(state: SessionState, customer_message: str) -> tuple[InternalMessageResponse, SessionState]:
    fs = state.flow_state
    cd = state.collected_data

    # ── CONFIRM & GENERATE (multi-range, single report) ───────────────────────
    if fs == "confirm_generate":
        low = customer_message.strip().lower()
        if low in ("change", "no", "edit"):
            # fall back to the normal picker for this report's date step
            report = _find_report(cd.get("report_name", ""))
            if report:
                return _advance_to_date(state, report, customer_message)
            return _resp(state, _msg_category(), TOP_CATEGORIES, "statement_category",
                         customer_message)
        # anything else (Generate / yes) → generate every range for the one report
        report = _find_report(cd.get("report_name", "")) or {}
        ranges = cd.get("ranges", [])
        if not report or not ranges:
            return _resp(state, _msg_category(), TOP_CATEGORIES, "statement_category",
                         customer_message)
        return _generate_ranges(state, report, ranges, customer_message)

    # ── Free-text slot-fill pre-pass (skips when it's a button/picker tap) ─────
    # Not run at 'start' (no step options yet) nor on terminal states.
    if fs not in ("start", "session_end_response", "confirm_generate"):
        _step_opts = _options_for_state(fs, cd)
        short = _try_slot_fill(state, customer_message, _step_opts)
        if short is not None:
            return short

    # ── START: account check ──────────────────────────────────────────────────
    if fs == "start":
        from src.core.langchain_agent import get_profile
        try:
            status = get_profile(state.sub_account_id or "").account_status
        except Exception:
            status = "active"
        if status in ("deactivated", "purged"):
            return _resp(state, _msg_deactivated(), _END_QR, "session_end_response",
                         customer_message, status="end", update={"flow": "statement"})
        # Free-text slot-fill at entry: the customer's first message may already
        # name the report + date range(s) → skip the questions and confirm.
        short = _try_slot_fill(
            state.model_copy(update={"flow": "statement"}),
            customer_message, step_options=TOP_CATEGORIES)
        if short is not None:
            return short
        return _resp(state, _msg_category(), TOP_CATEGORIES, "statement_category",
                     customer_message, update={"flow": "statement"})

    # ── CATEGORY ───────────────────────────────────────────────────────────────
    if fs == "statement_category":
        cat = _match(customer_message, TOP_CATEGORIES) or _extract(_EXTRACT_CATEGORY_SYS, customer_message).get("selected")
        if cat not in TOP_CATEGORIES:
            return _resp(state, "Please select a statement category:", TOP_CATEGORIES,
                         "statement_category", customer_message, status="reprompt")
        item_names = [it["name"] for it in _top_items(cat)]
        return _resp(state, _msg_reports(cat), item_names, "report_type", customer_message,
                     update={"collected_data": {**cd, "category": cat}})

    # ── TOP-LEVEL ITEM (report or sub-tab) ─────────────────────────────────────
    if fs == "report_type":
        cat = cd.get("category", "")
        item_names = [it["name"] for it in _top_items(cat)]
        picked = _match(customer_message, item_names) or _extract(_EXTRACT_PICK_SYS, customer_message, {"options": ", ".join(item_names)}).get("selected")
        item = next((it for it in _top_items(cat) if picked and it["name"].lower() == picked.lower()), None)
        if not item:
            return _resp(state, "Please choose one of the available reports:", item_names,
                         "report_type", customer_message, status="reprompt")
        if item.get("subtab"):
            sub_names = [r["name"] for r in item["reports"]]
            return _resp(state, _msg_subtab(item["name"]), sub_names, "subtab_select",
                         customer_message, update={"collected_data": {**cd, "subtab": item["name"]}})
        return _advance_to_date(state.model_copy(update={"collected_data": {**cd, "report_name": item["name"]}}),
                                item, customer_message)

    # ── SUB-TAB → leaf report ──────────────────────────────────────────────────
    if fs == "subtab_select":
        cat = cd.get("category", "")
        subtab = _find_subtab(cat, cd.get("subtab", ""))
        reports = subtab["reports"] if subtab else []
        names = [r["name"] for r in reports]
        picked = _match(customer_message, names) or _extract(_EXTRACT_PICK_SYS, customer_message, {"options": ", ".join(names)}).get("selected")
        report = next((r for r in reports if picked and r["name"].lower() == picked.lower()), None)
        if not report:
            return _resp(state, "Please choose one of the available reports:", names,
                         "subtab_select", customer_message, status="reprompt")
        return _advance_to_date(state.model_copy(update={"collected_data": {**cd, "report_name": report["name"]}}),
                                report, customer_message)

    # ── SEGMENT select ──────────────────────────────────────────────────────────
    if fs == "segment_select":
        seg = _match(customer_message, SEGMENTS)
        _rn = cd.get("report_name", "")
        if not seg:
            return _resp(state, _msg_segment(_rn), SEGMENTS, "segment_select", customer_message, status="reprompt")
        report = _find_report(_rn)
        next_input = (report or {}).get("date_input", "")
        cd2 = {**cd, "segment": seg}
        if next_input == "segment_then_fy":
            return _resp(state, _msg_fy(_rn), _financial_years(), "date_range_fy", customer_message,
                         update={"collected_data": cd2})
        # segment_then_range → UI shows a 30-day calendar range picker.
        return _resp(state, _msg_range(_rn), [], "date_range_30", customer_message,
                     update={"collected_data": cd2})

    # ── FY PICKER ────────────────────────────────────────────────────────────────
    if fs == "date_range_fy":
        fy_opts = _financial_years()
        label = _match(customer_message, fy_opts)
        if not label:
            sel = _extract(_EXTRACT_FY_SYS, customer_message, {"options": ", ".join(fy_opts)}).get("selected")
            label = _match(sel or "", fy_opts)
        if not label:
            return _resp(state, f"Please select a valid financial year{_for(cd.get('report_name',''))} "
                         "from the options:", fy_opts,
                         "date_range_fy", customer_message, status="reprompt")
        start, end = _resolve_fy_dates(label)
        return _send_and_confirm(state, _find_report(cd.get("report_name", "")) or {},
                                 start, end, customer_message, _msg_no_data_fy())

    # ── SINGLE DATE ──────────────────────────────────────────────────────────────
    if fs == "single_date":
        d = _parse_date(customer_message) or date.today()
        d = _clamp_future(d)
        ds = d.strftime("%d-%m-%Y")
        return _send_and_confirm(state, _find_report(cd.get("report_name", "")) or {},
                                 ds, ds, customer_message, _msg_no_data_date())

    # ── 30-DAY DATE RANGE ─────────────────────────────────────────────────────────
    if fs == "date_range_30":
        # Preset button ("Last 7/15/30 days") first, then typed range.
        rng = _preset_range(customer_message) or _parse_range(customer_message)
        if not rng:
            return _resp(state, _msg_range(cd.get("report_name", "")), [], "date_range_30",
                         customer_message, status="reprompt")
        d1, d2 = rng
        if (d2 - d1).days > 30 or d2 > date.today():
            return _resp(state, _msg_range_wrong(), [], "date_range_30", customer_message, status="reprompt")
        return _send_and_confirm(state, _find_report(cd.get("report_name", "")) or {},
                                 d1.strftime("%d-%m-%Y"), d2.strftime("%d-%m-%Y"),
                                 customer_message, _msg_no_data_date())

    # ── MONTH PICKER (year → month) ───────────────────────────────────────────────
    if fs == "date_range_month":
        if "selected_year" not in cd:
            year = _match(customer_message, YEAR_OPTIONS) or _extract(_EXTRACT_DATE_SYS, customer_message).get("selected_year")
            if year not in YEAR_OPTIONS:
                return _resp(state, _msg_month_year(cd.get("report_name", "")), YEAR_OPTIONS, "date_range_month",
                             customer_message, status="reprompt")
            return _resp(state, _msg_month_name(cd.get("report_name", "")), MONTH_NAMES, "date_range_month", customer_message,
                         update={"collected_data": {**cd, "selected_year": year}})
        year = cd["selected_year"]
        month = _match(customer_message, MONTH_NAMES)
        if not month:
            sel = _extract(_EXTRACT_DATE_SYS, customer_message).get("selected_month")
            month = _match(sel or "", MONTH_NAMES)
        if not month:
            return _resp(state, _msg_month_name(cd.get("report_name", "")), MONTH_NAMES, "date_range_month",
                         customer_message, status="reprompt")
        m = MONTH_NAMES.index(month) + 1
        last = calendar.monthrange(int(year), m)[1]
        start = f"01-{m:02d}-{year}"
        end = _clamp_future(date(int(year), m, last)).strftime("%d-%m-%Y")
        return _send_and_confirm(state, _find_report(cd.get("report_name", "")) or {},
                                 start, end, customer_message, _msg_no_data_month())

    if fs == "session_end_response":
        return handle_session_end(state, customer_message)

    logger.warning("[STATEMENT] unknown flow_state %r — reset", fs)
    return handle_statement(state.model_copy(update={"flow_state": "start"}), customer_message)


def _advance_to_date(state, report, customer_message):
    """Route a chosen leaf report to its date-input step."""
    di = report.get("date_input", "fy_picker")
    cd = state.collected_data

    if di == "unavailable":
        return _resp(state, _msg_unavailable(report["name"]), _END_QR, "session_end_response",
                     customer_message, status="ok")

    if di == "cml_direct":
        # CML: no picker — acknowledge and submit (last 1 year → today).
        today = date.today()
        start = (today - timedelta(days=365)).strftime("%d-%m-%Y")
        end = today.strftime("%d-%m-%Y")
        # Show the CML-specific ack regardless of downstream (fire-and-forget).
        from src.gateways.statement_api import request_statement_fireandforget
        try:
            request_statement_fireandforget(
                sub_account_id=state.sub_account_id or "", api_jobname=report["jobname"],
                endpoint=report["endpoint"], start_date=start, end_date=end)
        except Exception as exc:
            logger.warning("[STATEMENT] CML submit failed: %s", exc)
        return _resp(state, _msg_cml_ack(), _END_QR, "session_end_response", customer_message)

    _rn = report.get("name", "")

    if di == "fy_picker":
        return _resp(state, _msg_fy(_rn), _financial_years(), "date_range_fy", customer_message)

    if di in ("segment_then_fy", "segment_then_range"):
        return _resp(state, _msg_segment(_rn), SEGMENTS, "segment_select", customer_message)

    if di == "single_date":
        today_str = date.today().strftime("%d-%m-%Y")
        return _resp(state, _msg_single_date(today_str, _rn), [today_str], "single_date", customer_message)

    if di == "date_range_30":
        return _resp(state, _msg_range(_rn), [], "date_range_30", customer_message)

    # month_picker (default)
    return _resp(state, _msg_month_year(_rn), YEAR_OPTIONS, "date_range_month", customer_message)
