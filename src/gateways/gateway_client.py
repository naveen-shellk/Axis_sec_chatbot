"""
chatbot_web/src/gateways/gateway_client.py
--------------------------------------------
Gateway interface for chatbot_web.

Pattern: AgentCore Gateway MCP first → direct HTTP fallback
  POLICY (aligned with v5 agent/gateway/base.py):
    - local_test environment: direct HTTP fallback ALLOWED
    - uat / prod: direct HTTP fallback BLOCKED — Gateway only
    - Controlled via ENVIRONMENT env var

Required Gateway tools (7 of 14):
  customer-info-api___get_customer_profile
  statement-api-v2___request_statement
  statement-api-v2___request_statement_reports
  statement-api-v2___request_statement_comtrack
  statement-api-v2___get_ledger
  order-status-api___get_trade_book
  closure-api___create_account_closure
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import date, datetime
from typing import Any

import requests as _requests

from src.gateways.mcp_gateway import MCPGatewayError, call_tool

# ── Environment policy (aligned with v5 agent/gateway/base.py) ───────────────
# Direct HTTP fallback is only allowed in local_test.
# In uat / prod all traffic must go through the Gateway.
_ENV = os.getenv("ENVIRONMENT", "local_test")

def _direct_allowed() -> bool:
    return _ENV == "local_test"

def _blocked(api_name: str) -> dict:
    logger.warning(
        "[DIRECT API BLOCKED] %s skipped — direct fallback only in local_test (current=%s)",
        api_name, _ENV,
    )
    return {
        "success": False,
        "error":   "direct_api_disabled",
        "details": f"Direct API '{api_name}' disabled in '{_ENV}'. All traffic goes via Gateway.",
    }

logger = logging.getLogger(__name__)

# ── Tools proxy (ngrok) ───────────────────────────────────────────────────────
# When TOOLS_PROXY_URL is set, tool calls go DIRECTLY to the ngrok-tunnelled
# tools proxy (runtime → ngrok → proxy → internal API), bypassing both the
# AgentCore Gateway and the raw internal-IP direct calls. This lets the deployed
# container reach internal APIs it otherwise can't route to.
_TOOLS_PROXY_URL = os.getenv("TOOLS_PROXY_URL", "").rstrip("/")
_TOOLS_PROXY_KEY = os.getenv("TOOLS_PROXY_KEY", "")


def _proxy_enabled() -> bool:
    return bool(_TOOLS_PROXY_URL)


def _proxy_call(endpoint: str, body: dict) -> dict:
    """POST to the tools proxy endpoint (e.g. '/get_customer_profile')."""
    import json as _json
    url = f"{_TOOLS_PROXY_URL}/{endpoint.lstrip('/')}"
    headers = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "1"}
    if _TOOLS_PROXY_KEY:
        headers["X-Proxy-Key"] = _TOOLS_PROXY_KEY
    logger.info("[PROXY] POST %s", url)
    resp = _requests.post(url, json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Direct HTTP base URLs (read from env, same as email automation) ───────────
_CUSTOMER_PROFILE_URL  = os.getenv("CUSTOMER_PROFILE_URL",
                                   "https://cust-pvt-api.uat.asldt.in/customer-profile/details")
_REPORTS_BASE_URL      = os.getenv("STATEMENT_REQUEST_URL",
                                   "https://reports-api.uat.asldt.in").rstrip("/exports").rstrip("/")
if _REPORTS_BASE_URL.endswith("/exports"):
    _REPORTS_BASE_URL = _REPORTS_BASE_URL[:-8]
_ORDERS_URL            = os.getenv("ORDER_BOOK_URL",
                                   "https://ord-api.uat.asldt.in/books/get-trade-book")
_CLOSURE_URL           = os.getenv("ILEVERAGE_CLOSURE_URL",
                                   "https://connapi-preprod.axissl.in/api/create_account_closure/v2")
_LEDGER_URL            = os.getenv("LEDGER_URL",
                                   "https://reports-api.uat.asldt.in/inquiry/get-ledger")

# ── Reports API Basic Auth (same creds as the email automation) ───────────────
_ILEVERAGE_USERNAME = os.getenv("ILEVERAGE_USERNAME", "")
_ILEVERAGE_PASSWORD = os.getenv("ILEVERAGE_PASSWORD", "")
_REPORTS_API_BASE   = "https://reports-api.uat.asldt.in"

# ── Report routing (ported verbatim from the working email automation) ────────
# Maps a report jobname/type → (endpoint_suffix, canonical_reportname).
# Two endpoints only:
#   "exports"              → oneclick reports (subAccountId/reportName, ISO dates)
#   "statements/send-mail" → comtrack fire-and-forget (customerId/jobname, DD-MM dates)
# The API emails the report to the customer's registered address.
_REPORT_ROUTING: dict[str, tuple[str, str]] = {
    # comtrack send-mail (fire-and-forget)
    "cml (nsdl)":                       ("statements/send-mail", "CDSLCML,NSDLCML"),
    "cdslcml,nsdlcml":                  ("statements/send-mail", "CDSLCML,NSDLCML"),
    "nsdlcml":                          ("statements/send-mail", "NSDLCML"),
    "cdslcml":                          ("statements/send-mail", "CDSLCML"),
    "contract note":                    ("statements/send-mail", "CommonContractNotes"),
    "contract notes":                   ("statements/send-mail", "CommonContractNotes"),
    "commoncontractnotes":              ("statements/send-mail", "CommonContractNotes"),
    "dp transaction statement":         ("statements/send-mail", "DP"),
    "dp":                               ("statements/send-mail", "DP"),
    "agts":                             ("statements/send-mail", "AGTS"),
    "global statement (agts)":          ("statements/send-mail", "AGTS"),
    "commodity contract note":          ("statements/send-mail", "CommodityContractNotes"),
    "commodity daily margin":           ("statements/send-mail", "CommodityDailyMargin"),
    "bill":                             ("statements/send-mail", "Bill"),
    "equity margin":                    ("statements/send-mail", "EquityMargin"),
    "equitymargin":                     ("statements/send-mail", "EquityMargin"),
    "commodity contract note physical": ("statements/send-mail", "CommodityContractNotesPhysical"),
    "derivative physical bill":         ("statements/send-mail", "DerivativePhysicalBills"),
    "retention statement":              ("statements/send-mail", "RetentionStatement"),
    "retentionstatement":               ("statements/send-mail", "RetentionStatement"),
    "soacumros":                        ("statements/send-mail", "SOAcumROS"),
    "slbm contract notes":              ("statements/send-mail", "SLBMContractNotes"),
    "slbmcontractnotes":                ("statements/send-mail", "SLBMContractNotes"),
    # Exact jobname keys the statement flow passes (so routing is 1:1, no guess)
    "commoncontractnotes":              ("statements/send-mail", "CommonContractNotes"),
    "commoditycontractnotes":           ("statements/send-mail", "CommodityContractNotes"),
    "commoditycontractnotesphysical":   ("statements/send-mail", "CommodityContractNotesPhysical"),
    "derivativephysicalbills":          ("statements/send-mail", "DerivativePhysicalBills"),
    "commoditydailymargin":             ("statements/send-mail", "CommodityDailyMargin"),
    # exports (API emails the report). Names below are the ONLY ones the live
    # UAT Reports API /exports accepts (verified 2026-09): Ledger Statement,
    # Tax Statement, Portfolio Holding Statement, Pan-Level Transaction
    # Statement, Trade Book Statement. All other names return 400 "invalid
    # report name".
    "tax statement":                    ("exports", "Tax Statement"),
    "p&l statement":                    ("exports", "Portfolio Holding Statement"),
    "portfolio holding statement":      ("exports", "Portfolio Holding Statement"),
    "ledger report":                    ("exports", "Ledger Statement"),
    "ledger statement":                 ("exports", "Ledger Statement"),
    "account statement":                ("exports", "Ledger Statement"),
    "pan level transaction report":     ("exports", "Pan-Level Transaction Statement"),
    "pan-level transaction statement":  ("exports", "Pan-Level Transaction Statement"),
    "trade book statement":             ("exports", "Trade Book Statement"),
    # DP Holdings exports name — email automation uses "DP Holding Statement".
    "dp holdings":                      ("exports", "DP Holding Statement"),
    "dp holding statement":             ("exports", "DP Holding Statement"),
}


def _resolve_report_routing(report_name: str, endpoint_hint: str = "") -> tuple[str, str]:
    """
    Return (endpoint_suffix, canonical_reportname) for a report.
    Uses the routing table (exact then partial match); if the report isn't found,
    falls back to the caller's endpoint hint ("exports"/"sendmail") and the name
    as-is. Mirrors the email automation's resolve_report_routing.
    """
    key = (report_name or "").lower().strip()
    if key in _REPORT_ROUTING:
        return _REPORT_ROUTING[key]
    for k, v in _REPORT_ROUTING.items():
        if key and (key in k or k in key):
            return v
    suffix = "statements/send-mail" if endpoint_hint == "sendmail" else "exports"
    return (suffix, report_name)


def _is_vpc_error(raw: Any) -> bool:
    """Return True if the Gateway returned an OpenAPI connectivity error."""
    if not isinstance(raw, dict):
        return False
    text = raw.get("text", "")
    return (
        "OpenAPIClientException" in text or
        "Error executing HTTP request" in text or
        "ConnectionError" in text or
        "ConnectionTimeout" in text
    )


# =============================================================================
# Helpers
# =============================================================================

def mask_email(email: str) -> str:
    if not email or "@" not in email:
        return email or "your registered email"
    local, domain = email.rsplit("@", 1)
    if len(local) <= 2:
        return email
    return f"{local[0]}{'*' * (len(local) - 2)}{local[-1]}@{domain}"


def mask_mobile(mobile: str) -> str:
    if not mobile or len(mobile) < 4:
        return mobile or ""
    return f"XXXXXXXX{mobile[-4:]}"


def get_masked_email(sub_account_id: str) -> str:
    try:
        raw = get_customer_profile(sub_account_id)
        email = raw.get("email", "")
        return mask_email(email) if email else "your registered email"
    except Exception:
        return "your registered email"


def _customer_headers(sub_account_id: str = "000000") -> dict:
    return {
        "Content-Type":    "application/json",
        "X-MACAddress":    "",
        "X-IPAddress":     "",
        "X-MsgSequence":   "",
        "X-MsgGroup":      "",
        "X-Source":        "ChatBot",
        "X-SourceChannel": "Web",
        "X-SubAccountID":  sub_account_id,
    }


def _to_iso(date_str: str) -> str:
    """Convert DD-MM-YYYY → YYYY-MM-DD."""
    if date_str and len(date_str) == 10 and date_str[2] == "-" and date_str[5] == "-":
        d, m, y = date_str.split("-")
        return f"{y}-{m}-{d}"
    return date_str


# =============================================================================
# Tool 1 — customer-info-api___get_customer_profile
# Flows: account_details, brokerage, closure, ipo, login_query, order_status, statement
# =============================================================================

def get_customer_profile(sub_account_id: str) -> dict[str, Any]:
    """
    Tools proxy (if TOOLS_PROXY_URL set) → Gateway → direct HTTP fallback.
    Returns raw profile dict (data envelope unwrapped).
    """
    logger.info("[GW] get_customer_profile sub=%s", sub_account_id)

    if _proxy_enabled():
        try:
            return _proxy_call("/get_customer_profile", {"sub_account_id": sub_account_id})
        except Exception as exc:
            logger.warning("[PROXY] get_customer_profile failed: %s — trying gateway/direct", exc)

    # ── Try Gateway ───────────────────────────────────────────────────────────
    try:
        raw = call_tool(
            "customer-info-api___get_customer_profile",
            {"X-SubAccountID": sub_account_id, "X-Source": "ChatBot", "X-SourceChannel": "Web"},
        )
        if not _is_vpc_error(raw):
            data = raw.get("data", raw) if isinstance(raw, dict) else raw
            logger.info("[GW] get_customer_profile via Gateway OK")
            return data if isinstance(data, dict) else raw
        logger.warning("[GW] get_customer_profile Gateway VPC error — falling back to direct HTTP")
    except (MCPGatewayError, Exception) as e:
        logger.warning("[GW] get_customer_profile Gateway error — falling back: %s", e)

    # ── Direct HTTP fallback (local_test only) ────────────────────────────────
    if not _direct_allowed():
        return _blocked("get_customer_profile")
    return _get_customer_profile_direct(sub_account_id)


def _get_customer_profile_direct(sub_account_id: str) -> dict[str, Any]:
    """Direct HTTP to cust-pvt-api.uat.asldt.in/customer-profile/details"""
    try:
        resp = _requests.post(
            _CUSTOMER_PROFILE_URL,
            json={"subAccountId": sub_account_id},
            headers=_customer_headers(sub_account_id),
            timeout=10,
        )
        logger.info("[GW] get_customer_profile direct HTTP status=%d", resp.status_code)
        if resp.status_code == 200:
            data = resp.json()
            inner = data.get("data", data)
            return inner if isinstance(inner, dict) else data
        return {}
    except Exception as e:
        logger.error("[GW] get_customer_profile direct HTTP failed: %s", e)
        return {}


# =============================================================================
# Tool 2/3/4 — statement-api-v2 (exports / oneclick / comtrack)
# Flows: statement
# =============================================================================

def request_statement(
    sub_account_id: str,
    report_name:    str,
    start_date:     str,
    end_date:       str,
    endpoint:       str = "exports",   # "exports" | "sendmail" | "oneclick"
    dp_account_no:  str = "",
) -> dict[str, Any]:
    """Tools proxy → Gateway → direct HTTP fallback."""
    logger.info("[GW] request_statement sub=%s report=%s endpoint=%s",
                sub_account_id, report_name, endpoint)

    # ── Tools proxy (ngrok) — same path as every other tool ───────────────────
    # When TOOLS_PROXY_URL is set (deployed runtime / local client), route the
    # statement request through the proxy so it reaches the internal Reports API.
    # On the proxy host itself TOOLS_PROXY_URL is empty, so it falls through to
    # the gateway/direct call below (no recursion).
    if _proxy_enabled():
        try:
            return _proxy_call("/request_statement", {
                "sub_account_id": sub_account_id,
                "report_name":    report_name,
                "start_date":     start_date,
                "end_date":       end_date,
                "endpoint":       endpoint,
                "dp_account_no":  dp_account_no,
            })
        except Exception as exc:
            logger.warning("[PROXY] request_statement failed: %s — trying gateway/direct", exc)

    # ── Try Gateway ───────────────────────────────────────────────────────────
    try:
        raw = _call_statement_gateway(sub_account_id, report_name, start_date, end_date,
                                       endpoint, dp_account_no)
        if not _is_vpc_error(raw):
            logger.info("[GW] request_statement via Gateway OK")
            return raw
        logger.warning("[GW] request_statement Gateway VPC error — falling back")
    except (MCPGatewayError, Exception) as e:
        logger.warning("[GW] request_statement Gateway error — falling back: %s", e)

    # ── Direct HTTP fallback (local_test only) ────────────────────────────────
    if not _direct_allowed():
        return _blocked("request_statement")
    return _request_statement_direct(sub_account_id, report_name, start_date, end_date,
                                      endpoint, dp_account_no)


def _call_statement_gateway(sub_account_id, report_name, start_date, end_date,
                              endpoint, dp_account_no) -> dict:
    if endpoint == "exports":
        return call_tool("statement-api-v2___request_statement", {
            "X-SubAccountId":   sub_account_id,
            "X-Source":         "ChatBot",
            "X-SourceChannel":  "Web",
            "subAccountId":     sub_account_id,
            "reportName":       report_name,
            "startDate":        _to_iso(start_date),
            "endDate":          _to_iso(end_date),
            "fileType":         "xlsx",
            "downloadTypeFlag": "E",
            "source":           "chat-bot",
            "dpaccountno":      dp_account_no,
            "holdingType":      "All",
        })
    if endpoint == "sendmail":
        return call_tool("statement-api-v2___request_statement_comtrack", {
            "X-SubAccountId": sub_account_id,
            "customer_id":    sub_account_id,
            "dp_id":          dp_account_no,
            "start_date":     start_date,
            "end_date":       end_date,
            "jobname":        report_name,
            "request_type":   "email",
        })
    # oneclick
    return call_tool("statement-api-v2___request_statement_reports", {
        "X-SubAccountId":  sub_account_id,
        "X-Source":        "ChatBot",
        "X-SourceChannel": "Web",
        "customer_id":     sub_account_id,
        "dp_id":           dp_account_no,
        "start_date":      start_date,
        "end_date":        end_date,
        "jobname":         report_name,
        "request_type":    "email",
    })


def _request_statement_direct(sub_account_id, report_name, start_date, end_date,
                                endpoint, dp_account_no) -> dict:
    """Direct HTTP to reports-api.uat.asldt.in"""
    try:
        endpoint_suffix, canonical_name = _resolve_report_routing(report_name, endpoint)

        headers = {
            "Content-Type":    "application/json",
            "X-SubAccountId":  sub_account_id,
            "X-Source":        "ChatBot",
            "X-SourceChannel": "Web",
        }

        # send-mail reports (and DP holdings on exports) need the DP account
        # number. Resolve it from the customer profile when not supplied — the
        # API requires dpId for CML, DP transaction, SOAcumROS, etc.
        if not dp_account_no and (endpoint_suffix == "statements/send-mail"
                                   or "dp" in canonical_name.lower()):
            try:
                from src.gateways.customer_api import get_customer_profile as _cp
                dp_account_no = _cp(sub_account_id).demat_account_no or ""
                logger.info("[GW] request_statement resolved dp account from profile: %s",
                            dp_account_no or "(none)")
            except Exception as exc:
                logger.warning("[GW] request_statement could not resolve dp account: %s", exc)

        if endpoint_suffix == "statements/send-mail":
            # send-mail expects all camelCase: customerId/dpId/startDate/endDate/
            # requestType + jobname; dates DD-MM-YYYY. (Confirmed against the live
            # UAT Reports API — the DP-account field must be "dpId", not "dp_id".)
            url = f"{_REPORTS_API_BASE}/statements/send-mail"
            payload = {
                "customerId":  sub_account_id,
                "dpId":        dp_account_no,
                "startDate":   start_date,          # DD-MM-YYYY
                "endDate":     end_date,
                "jobname":     canonical_name,
                "requestType": "email",
            }
        else:  # exports (oneclick)
            url = f"{_REPORTS_API_BASE}/exports"
            payload = {
                "subAccountId":     sub_account_id,
                "reportName":       canonical_name,
                "startDate":        _to_iso(start_date),   # YYYY-MM-DD
                "endDate":          _to_iso(end_date),
                "fileType":         "xlsx",
                "dpaccountno":      dp_account_no,
                "holdingType":      "All",
                "downloadTypeFlag": "D",
                "source":           "chat-bot",
            }

        auth = None
        if _ILEVERAGE_USERNAME and _ILEVERAGE_PASSWORD:
            auth = (_ILEVERAGE_USERNAME, _ILEVERAGE_PASSWORD)
        else:
            logger.warning("[GW] request_statement: ILEVERAGE creds missing — Basic Auth skipped")

        logger.info("[GW] request_statement direct → %s report=%s", url, canonical_name)
        resp = _requests.post(url, json=payload, headers=headers, auth=auth, timeout=60)
        logger.info("[GW] request_statement direct HTTP status=%d", resp.status_code)
        if resp.status_code in (200, 201):
            return resp.json()
        logger.error("[GW] request_statement direct failed: HTTP %d | %s",
                     resp.status_code, resp.text[:300])
        return {"error": f"HTTP {resp.status_code}", "detail": resp.text[:300]}
    except Exception as e:
        logger.error("[GW] request_statement direct HTTP failed: %s", e)
        return {"error": str(e)}


# =============================================================================
# Tool 4 — statement-api-v2___get_ledger
# Flows: brokerage (trading charges)
# =============================================================================

def get_ledger(sub_account_id: str, start_date: str, end_date: str) -> dict[str, Any]:
    """Tools proxy → Gateway → direct HTTP fallback."""
    logger.info("[GW] get_ledger sub=%s %s→%s", sub_account_id, start_date, end_date)

    if _proxy_enabled():
        try:
            return _proxy_call("/get_ledger_balance",
                               {"sub_account_id": sub_account_id,
                                "start_date": start_date, "end_date": end_date})
        except Exception as exc:
            logger.warning("[PROXY] get_ledger failed: %s — trying gateway/direct", exc)

    # ── Try Gateway ───────────────────────────────────────────────────────────
    try:
        raw = call_tool("statement-api-v2___get_ledger", {
            "x-SubAccountId": sub_account_id,
            "startDate":      start_date,
            "endDate":        end_date,
        })
        if not _is_vpc_error(raw):
            return _normalise_ledger(raw)
        logger.warning("[GW] get_ledger Gateway VPC error — falling back")
    except (MCPGatewayError, Exception) as e:
        logger.warning("[GW] get_ledger Gateway error — falling back: %s", e)

    # ── Direct HTTP fallback (local_test only) ────────────────────────────────
    if not _direct_allowed():
        return _blocked("get_ledger")
    return _get_ledger_direct(sub_account_id, start_date, end_date)


def _normalise_ledger(raw: Any) -> dict:
    if isinstance(raw, dict):
        data = raw.get("data", raw)
        if isinstance(data, dict):
            inner = data.get("data", data)
            if isinstance(inner, dict):
                return {
                    "success":         True,
                    "opening_balance": str(inner.get("openingBalance", "0")),
                    "closing_balance": str(inner.get("closingBalance",  "0")),
                    "emargin_balance": str(inner.get("emarginBalance",  "0")),
                }
    return {"success": False, "opening_balance": "0",
            "closing_balance": "0", "emargin_balance": "0"}


def _get_ledger_direct(sub_account_id: str, start_date: str, end_date: str) -> dict:
    """Direct HTTP to reports-api.uat.asldt.in/inquiry/get-ledger"""
    try:
        resp = _requests.post(
            _LEDGER_URL,
            json={"startDate": start_date, "endDate": end_date},
            headers={"x-SubAccountId": sub_account_id, "Content-Type": "application/json"},
            timeout=30,
        )
        logger.info("[GW] get_ledger direct HTTP status=%d", resp.status_code)
        if resp.status_code == 200:
            return _normalise_ledger({"data": resp.json()})
        return {"success": False, "opening_balance": "0",
                "closing_balance": "0", "emargin_balance": "0",
                "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        logger.error("[GW] get_ledger direct HTTP failed: %s", e)
        return {"success": False, "opening_balance": "0",
                "closing_balance": "0", "emargin_balance": "0", "error": str(e)}


def send_dp_bill(sub_account_id: str, start_date: str, end_date: str,
                 dp_id: str = "") -> dict[str, Any]:
    """Tools proxy → Gateway → direct HTTP fallback. Uses comtrack/sendmail route."""
    if _proxy_enabled():
        try:
            return _proxy_call("/send_dp_bill",
                               {"sub_account_id": sub_account_id,
                                "start_date": start_date, "end_date": end_date})
        except Exception as exc:
            logger.warning("[PROXY] send_dp_bill failed: %s — trying gateway/direct", exc)
    logger.info("[GW] send_dp_bill sub=%s", sub_account_id)
    masked = get_masked_email(sub_account_id)

    # NOTE: The legacy snake_case send-mail payload ({customer_id, dp_id,
    # start_date, ...} with no X-SubAccountId header / no Basic auth) is
    # REJECTED by the live UAT Reports API with 400 "sub account id is missing".
    # Route the DP bill through the same working send-mail contract every other
    # statement uses: _request_statement_direct resolves "Bill" → send-mail,
    # adds the X-SubAccountId header, camelCase payload (customerId/dpId/
    # startDate/endDate/jobname/requestType), resolves dpId, and Basic auth.
    if _direct_allowed():
        resp = _request_statement_direct(
            sub_account_id=sub_account_id,
            report_name="Bill",
            start_date=start_date,      # DD-MM-YYYY
            end_date=end_date,
            endpoint="sendmail",
            dp_account_no=dp_id,
        )
    else:
        resp = request_statement(
            sub_account_id=sub_account_id,
            report_name="Bill",
            start_date=start_date,
            end_date=end_date,
            endpoint="sendmail",
            dp_account_no=dp_id,
        )

    # A dict without "error" means the bill was dispatched. "No documents"/502
    # is a data condition (no bill for that period) → success=False + no_data.
    if isinstance(resp, dict) and "error" in resp:
        err = str(resp.get("detail") or resp.get("error") or "").lower()
        no_data = "no documents" in err or "not found" in err
        return {"success": False, "masked_email": masked,
                "no_data": no_data, "error": resp.get("error")}
    return {"success": True, "masked_email": masked}


# =============================================================================
# Tool 5 — order-status-api___get_trade_book
# Flows: order_status
# =============================================================================

_SEGMENT_MAP = {
    "Equity":       "EQ",
    "Commodity":    "COMM",
    "Derivatives":  "FO",
    "Mutual Funds": "MF",
}


def _filter_today(trades: list[dict]) -> list[dict]:
    today = date.today()
    result = []
    for t in trades:
        ts = t.get("tradeTS", "")
        if not ts:
            continue
        try:
            if datetime.fromisoformat(ts.replace("Z", "+00:00")).date() == today:
                result.append(t)
        except (ValueError, TypeError):
            pass
    return result


def _call_trade_book_gateway(sub_account_id: str, segment: str) -> dict:
    return call_tool("order-status-api___get_trade_book", {
        "X-SubAccountID": sub_account_id,
        "segment":        segment,
        "orderDetails":   [{"omsOrderId": ""}],
    })


def _call_trade_book_direct(sub_account_id: str, segment: str) -> dict:
    """Direct HTTP to ord-api.uat.asldt.in/books/get-trade-book"""
    try:
        resp = _requests.post(
            _ORDERS_URL,
            json={"segment": segment, "orderDetails": [{"omsOrderId": ""}]},
            headers={"Content-Type": "application/json", "X-SubAccountID": sub_account_id},
            timeout=30,
            verify=False,
        )
        logger.info("[GW] get_trade_book direct HTTP status=%d", resp.status_code)
        if resp.status_code == 200:
            data = resp.json()
            return {"success": True, "data": data.get("data", []), "api_response": data}
        return {"success": False, "data": [], "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        logger.error("[GW] get_trade_book direct HTTP failed: %s", e)
        return {"success": False, "data": [], "error": str(e)}


def get_todays_orders(sub_account_id: str, segment_label: str) -> dict[str, Any]:
    """Tools proxy → Gateway → direct HTTP fallback."""
    if _proxy_enabled():
        try:
            return _proxy_call("/get_todays_orders",
                               {"sub_account_id": sub_account_id, "segment": segment_label})
        except Exception as exc:
            logger.warning("[PROXY] get_todays_orders failed: %s — trying gateway/direct", exc)
    segment = _SEGMENT_MAP.get(segment_label, "EQ")
    logger.info("[GW] get_todays_orders sub=%s segment=%s", sub_account_id, segment)

    raw = None
    try:
        raw = _call_trade_book_gateway(sub_account_id, segment)
        if _is_vpc_error(raw):
            logger.warning("[GW] get_todays_orders Gateway VPC error — falling back")
            raw = None
    except (MCPGatewayError, Exception) as e:
        logger.warning("[GW] get_todays_orders Gateway error — falling back: %s", e)

    if raw is None:
        raw = _call_trade_book_direct(sub_account_id, segment)

    trades = raw.get("data", []) if isinstance(raw, dict) else []
    todays = _filter_today(trades) if isinstance(trades, list) else []
    return {"found": bool(todays), "orders": todays,
            "count": len(todays), "segment": segment_label}


def send_order_history_email(sub_account_id: str, segment_label: str,
                              date_str: str, end_date: str = "") -> dict[str, Any]:
    """
    Order history → Trade Book Statement (aligned with the email automation).

    The email automation's order_history_agent fetches order history via the
    Reports API "Trade Book Statement" on /exports for a date range (emailed to
    the customer), NOT the live intraday get-trade-book endpoint. We mirror that:
    request the "Trade Book Statement" report for the selected range. When
    end_date is omitted, date_str is used as both start and end (single day).
    """
    end = end_date or date_str
    if _proxy_enabled():
        try:
            return _proxy_call("/send_order_history_email",
                               {"sub_account_id": sub_account_id,
                                "segment": segment_label,
                                "date_str": date_str, "end_date": end})
        except Exception as exc:
            logger.warning("[PROXY] send_order_history_email failed: %s — trying gateway/direct", exc)

    masked = get_masked_email(sub_account_id)
    logger.info("[GW] send_order_history_email sub=%s segment=%s %s→%s → Trade Book Statement",
                sub_account_id, segment_label, date_str, end)

    # local_test runs with the gateway disabled → go straight to direct HTTP
    # (/exports). _request_statement_direct converts DD-MM-YYYY → YYYY-MM-DD and
    # resolves "Trade Book Statement" via the routing table.
    if _direct_allowed():
        resp = _request_statement_direct(
            sub_account_id=sub_account_id,
            report_name="Trade Book Statement",
            start_date=date_str,      # DD-MM-YYYY
            end_date=end,
            endpoint="exports",
            dp_account_no="",
        )
    else:
        # Deployed runtime: go through the standard proxy/gateway wrapper.
        resp = request_statement(
            sub_account_id=sub_account_id,
            report_name="Trade Book Statement",
            start_date=date_str,
            end_date=end,
            endpoint="exports",
        )

    # A dict without "error" means the report was dispatched (emailed).
    # An explicit error (incl. "no documents") → success=False; the flow shows
    # the appropriate message and offers the fallback contact.
    if isinstance(resp, dict) and "error" in resp:
        err = str(resp.get("detail") or resp.get("error") or "").lower()
        no_data = "no documents" in err or "not found" in err
        return {"success": False, "masked_email": masked,
                "no_data": no_data, "error": resp.get("error")}
    return {"success": True, "masked_email": masked}


# =============================================================================
# Tool 6 — closure-api___create_account_closure
# Flows: closure
# NOTE: Direct HTTP fallback for closure calls internal IP (10.9.161.132)
#       which only works when the Runtime is deployed inside the same VPC.
# =============================================================================

def create_closure_request(sub_account_id: str, email: str, name: str,
                            type_of_account_closure: str = "demat",
                            dp_account_no: str = "") -> dict[str, Any]:
    """Tools proxy → Gateway → direct HTTP fallback (internal IP — requires VPC)."""
    logger.info("[GW] create_closure_request sub=%s type=%s", sub_account_id, type_of_account_closure)

    if _proxy_enabled():
        try:
            return _proxy_call("/create_account_closure",
                               {"sub_account_id": sub_account_id, "email": email, "name": name,
                                "type_of_account_closure": type_of_account_closure,
                                "dp_account_no": dp_account_no})
        except Exception as exc:
            logger.warning("[PROXY] create_closure_request failed: %s — trying gateway/direct", exc)

    try:
        payload_gw = {
            "ent_id":                  sub_account_id,
            "leverage_email_number":   f"chatbot-{sub_account_id}",
            "from_email_id":           email or "unknown@chatbot",
            "type_of_account_closure": type_of_account_closure,
            "remarks": f"Account closure request via web chatbot for {name or sub_account_id}",
        }
        if dp_account_no:
            payload_gw["dpAccountNo"] = dp_account_no

        raw = call_tool("closure-api___create_account_closure", payload_gw)
        if not _is_vpc_error(raw):
            logger.info("[GW] create_closure_request via Gateway OK")
            return raw if isinstance(raw, dict) else {"api_response": raw}
        logger.warning("[GW] create_closure_request Gateway VPC error — falling back")
    except (MCPGatewayError, Exception) as e:
        logger.warning("[GW] create_closure_request Gateway error — falling back: %s", e)

    # Direct HTTP — only works inside VPC (internal IP)
    return _create_closure_direct(sub_account_id, email, name, type_of_account_closure, dp_account_no)


def _create_closure_direct(sub_account_id: str, email: str, name: str,
                             type_of_account_closure: str,
                             dp_account_no: str = "") -> dict:
    try:
        payload = {
            "ent_id":                  sub_account_id,
            "leverage_email_number":   f"chatbot-{sub_account_id}",
            "from_email_id":           email or "",
            "type_of_account_closure": type_of_account_closure,
            "remarks":                 f"Account closure request via web chatbot for {name or sub_account_id}",
        }
        if dp_account_no:
            payload["dpAccountNo"] = dp_account_no
        resp = _requests.post(
            _CLOSURE_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        logger.info("[GW] create_closure_request direct HTTP status=%d", resp.status_code)
        if resp.status_code in (200, 201):
            return {"success": True, "api_response": resp.json()}
        return {"success": False, "api_response": {"reason": "error"},
                "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        logger.error("[GW] create_closure_request direct HTTP failed: %s", e)
        return {"success": False, "api_response": {"reason": "error"}, "error": str(e)}


# =============================================================================
# Format helper (re-exported for order_api.py compat)
# =============================================================================

def format_orders_for_chat(orders: list[dict], segment_label: str) -> str:
    if not orders:
        return f"No orders found for today in {segment_label}."
    lines = [f"Today's {segment_label} Orders:\n"]
    for i, o in enumerate(orders[:10], 1):
        sym  = o.get("symbol", "N/A")
        txn  = o.get("transactionType", "N/A")
        qty  = o.get("tradeQty", 0)
        px   = o.get("tradePrice", 0.0)
        prod = o.get("product", "")
        lines.append(f"{i}. {sym} | {txn} | Qty: {qty} | ₹{px:.2f} | {prod}")
    if len(orders) > 10:
        lines.append(f"...and {len(orders) - 10} more orders")
    return "\n".join(lines)
