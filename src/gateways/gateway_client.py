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
    Gateway first → direct HTTP fallback.
    Returns raw profile dict (data envelope unwrapped).
    """
    logger.info("[GW] get_customer_profile sub=%s", sub_account_id)

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
    """Gateway first → direct HTTP fallback."""
    logger.info("[GW] request_statement sub=%s report=%s endpoint=%s",
                sub_account_id, report_name, endpoint)

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
        headers = {
            "Content-Type":    "application/json",
            "X-SubAccountID":  sub_account_id,
            "X-Source":        "ChatBot",
            "X-SourceChannel": "Web",
        }
        if endpoint == "exports":
            url     = f"https://reports-api.uat.asldt.in/exports"
            payload = {
                "subAccountId":     sub_account_id,
                "reportName":       report_name,
                "startDate":        _to_iso(start_date),
                "endDate":          _to_iso(end_date),
                "fileType":         "xlsx",
                "downloadTypeFlag": "E",
                "source":           "chat-bot",
                "dpaccountno":      dp_account_no,
                "holdingType":      "All",
            }
        elif endpoint == "sendmail":
            url     = f"https://reports-api.uat.asldt.in/statements/send-mail"
            payload = {
                "customer_id":  sub_account_id,
                "dp_id":        dp_account_no,
                "start_date":   start_date,
                "end_date":     end_date,
                "jobname":      report_name,
                "request_type": "email",
            }
        else:  # oneclick
            url     = f"https://reports-api.uat.asldt.in/oneclick/statements/request-statement"
            payload = {
                "customer_id":  sub_account_id,
                "dp_id":        dp_account_no,
                "start_date":   start_date,
                "end_date":     end_date,
                "jobname":      report_name,
                "request_type": "email",
            }

        resp = _requests.post(url, json=payload, headers=headers, timeout=60)
        logger.info("[GW] request_statement direct HTTP status=%d", resp.status_code)
        if resp.status_code in (200, 201):
            return resp.json()
        return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        logger.error("[GW] request_statement direct HTTP failed: %s", e)
        return {"error": str(e)}


# =============================================================================
# Tool 4 — statement-api-v2___get_ledger
# Flows: brokerage (trading charges)
# =============================================================================

def get_ledger(sub_account_id: str, start_date: str, end_date: str) -> dict[str, Any]:
    """Gateway first → direct HTTP fallback."""
    logger.info("[GW] get_ledger sub=%s %s→%s", sub_account_id, start_date, end_date)

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
    """Gateway first → direct HTTP fallback. Uses comtrack/sendmail route."""
    logger.info("[GW] send_dp_bill sub=%s", sub_account_id)
    masked = get_masked_email(sub_account_id)

    try:
        raw = call_tool("statement-api-v2___request_statement_comtrack", {
            "X-SubAccountId": sub_account_id,
            "customer_id":    sub_account_id,
            "dp_id":          dp_id,
            "start_date":     start_date,
            "end_date":       end_date,
            "jobname":        "Bill",
            "request_type":   "email",
        })
        if not _is_vpc_error(raw):
            return {"success": True, "masked_email": masked}
        logger.warning("[GW] send_dp_bill Gateway VPC error — falling back")
    except (MCPGatewayError, Exception) as e:
        logger.warning("[GW] send_dp_bill Gateway error — falling back: %s", e)

    # Direct HTTP (local_test only)
    if not _direct_allowed():
        return _blocked("send_dp_bill")
    try:
        resp = _requests.post(
            "https://reports-api.uat.asldt.in/statements/send-mail",
            json={"customer_id": sub_account_id, "dp_id": dp_id,
                  "start_date": start_date, "end_date": end_date,
                  "jobname": "Bill", "request_type": "email"},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        success = resp.status_code in (200, 201)
        return {"success": success, "masked_email": masked}
    except Exception as e:
        logger.error("[GW] send_dp_bill direct HTTP failed: %s", e)
        return {"success": False, "masked_email": masked, "error": str(e)}


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
    """Gateway first → direct HTTP fallback."""
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
                              date_str: str) -> dict[str, Any]:
    """Gateway first → direct HTTP fallback. Fetches trade book data."""
    segment = _SEGMENT_MAP.get(segment_label, "EQ")
    masked  = get_masked_email(sub_account_id)
    logger.info("[GW] send_order_history_email sub=%s segment=%s date=%s",
                sub_account_id, segment, date_str)

    success = False
    try:
        raw = _call_trade_book_gateway(sub_account_id, segment)
        if not _is_vpc_error(raw):
            success = True
        else:
            raw = _call_trade_book_direct(sub_account_id, segment)
            success = raw.get("success", False)
    except (MCPGatewayError, Exception):
        raw = _call_trade_book_direct(sub_account_id, segment)
        success = raw.get("success", False)

    return {"success": success, "masked_email": masked}


# =============================================================================
# Tool 6 — closure-api___create_account_closure
# Flows: closure
# NOTE: Direct HTTP fallback for closure calls internal IP (10.9.161.132)
#       which only works when the Runtime is deployed inside the same VPC.
# =============================================================================

def create_closure_request(sub_account_id: str, email: str, name: str,
                            type_of_account_closure: str = "demat",
                            dp_account_no: str = "") -> dict[str, Any]:
    """Gateway first → direct HTTP fallback (internal IP — requires VPC)."""
    logger.info("[GW] create_closure_request sub=%s type=%s", sub_account_id, type_of_account_closure)

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
