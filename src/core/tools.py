"""
chatbot_langchain/src/core/tools.py
------------------------------------
Tool definitions for the web channel chatbot — LangChain variant.

This is the LangChain port of chatbot_web/src/core/tools.py. The tool BODIES are
byte-for-byte identical (same gateway wiring); only the framework binding differs:

  chatbot_web (Strands):   functions decorated with `@tool` from `strands`.
  chatbot_langchain:       plain Python functions, wrapped as LangChain
                           StructuredTools for the agent, PLUS a name→callable
                           registry so run_tool() can invoke them directly.

Why plain functions + StructuredTool wrappers:
  - The deterministic flows call run_tool("name", **kwargs) and expect a plain
    dict back. Keeping the raw Python functions makes that a direct, reliable
    call (no framework result-unwrapping needed).
  - The fully-agentic path (run_agent_turn) needs LangChain-native tools to bind
    to the model; those are built from the same functions via
    StructuredTool.from_function, so the LLM sees identical names + docstrings.

Tool auth summary:
  NO AUTH (noauth / header-only):
    - escalate_to_agent          — no external call
    - get_account_status         — X-headers only
    - get_customer_profile       — X-headers only
    - get_todays_orders          — X-SubAccountID header only
    - send_order_history_email   — X-SubAccountID header only

  BASIC AUTH (REPORTS_USERNAME / REPORTS_PASSWORD):
    - request_statement          — Reports API
    - get_ledger_balance         — Reports API
    - send_dp_bill               — Reports API

AgentCore deployment:
  When running inside AgentCore Runtime, credentials come from the IAM execution
  role (no explicit AWS keys needed). ChatBedrockConverse uses the runtime's IAM
  role. The same tools work in local dev (with explicit keys in .env) and in
  production (with IAM role — keys not needed).
"""

from __future__ import annotations

import logging
import os

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)


# =============================================================================
# AUTHENTICATION — signals the handler to start the OTP phone flow
# =============================================================================

def request_authentication(reason: str = "account action requires verification") -> dict:
    """
    Request that the customer verify their identity before an account-specific
    action can proceed. Call this WHENEVER the customer asks for something that
    needs their account (statement, order status, account details, brokerage,
    login help, account closure, IPO) AND you have not been given a verified
    Sub-Account ID in the context.

    Do NOT ask the customer for their account number yourself — call this tool
    and the system will run secure OTP verification (mobile number → OTP) and
    resolve the account for you.

    Args:
        reason: Brief reason authentication is needed (e.g. "view statement").

    Returns:
        {"auth_required": True, "reason": str}
    """
    logger.info("[TOOL] request_authentication: %s", reason)
    return {"auth_required": True, "reason": reason}


# =============================================================================
# ESCALATION — no external call
# =============================================================================

def escalate_to_agent(reason: str) -> dict:
    """
    Escalate the conversation to a live human agent via Webex Contact Center (WxCC).
    Call this when the customer requests to speak to a person, has a complaint,
    or when the chatbot cannot resolve the issue.

    This signals eventid 1002 to the API Gateway. WxCC handles all queue routing —
    the chatbot does nothing else after this call.

    Auth: None — no external API call.

    Args:
        reason: Brief description of why escalation is needed.

    Returns:
        {"escalate": True, "reason": str, "eventid": "1002"}
    """
    logger.info("[TOOL] escalate_to_agent: %s", reason)
    return {"escalate": True, "reason": reason, "eventid": "1002"}


# =============================================================================
# CUSTOMER INFO — X-tracking headers only (no secret auth)
# =============================================================================

def get_account_status(sub_account_id: str) -> dict:
    """
    Check whether the customer's trading account is active, deactivated, or purged.
    Calls customer-info-api___get_customer_profile via AgentCore Gateway.
    Falls back to direct HTTP if Gateway is unavailable.
    Auth: X-headers only — no Basic Auth required.
    """
    try:
        from src.gateways.gateway_client import get_customer_profile
        result = get_customer_profile(sub_account_id)
        data   = result.get("data", result)
        raw    = str(data.get("accountStatus", "E")).strip().lower() if isinstance(data, dict) else "e"
        status = "active" if raw == "e" else ("deactivated" if raw == "d" else "purged")
        return {"status": status}
    except Exception as exc:
        logger.error("[TOOL] get_account_status failed: %s", exc)
        return {"status": "active", "error": str(exc)}


def get_customer_profile(sub_account_id: str) -> dict:
    """
    Fetch full customer profile via AgentCore Gateway (customer-info-api___get_customer_profile).
    Falls back to direct HTTP if Gateway is unavailable.
    Auth: X-headers only — no Basic Auth required.
    """
    try:
        from src.gateways.gateway_client import get_customer_profile as _gw
        from src.gateways.customer_api import mask_email, mask_mobile
        result = _gw(sub_account_id)
        data   = result.get("data", result) if isinstance(result, dict) else {}
        return {
            "name":                 data.get("name") or data.get("customerName", ""),
            "email":                data.get("email", ""),
            "masked_email":         mask_email(data.get("email", "")),
            "phone":                data.get("mobileNo") or data.get("phone", ""),
            "masked_phone":         mask_mobile(data.get("mobileNo") or data.get("phone", "")),
            "account_status":       "active" if str(data.get("accountStatus","e")).lower()=="e" else "deactivated",
            "portal_status":        1 if data.get("Device") else 0,
            "deactivation_code":    str(data.get("entStatusLov", "")).strip().upper(),
            "account_opening_date": data.get("accountOpeningDate", ""),
        }
    except Exception as exc:
        logger.error("[TOOL] get_customer_profile failed: %s", exc)
        return {"error": str(exc), "account_status": "active"}


def get_customer_profile_full(sub_account_id: str) -> dict:
    """
    Fetch the COMPLETE customer profile as a serialisable dict, including every
    field the flow handlers need (name, emails, account/demat/trading numbers,
    status, and the full raw API envelope under "raw").

    Use this when a flow needs detailed profile data (Account Details, Closure,
    Login Query, etc.). Returns the same data as the CustomerProfile dataclass.
    Auth: X-headers only — no Basic Auth required.
    """
    try:
        from src.gateways.customer_api import get_customer_profile as _typed
        p = _typed(sub_account_id)
        return {
            "sub_account_id":       p.sub_account_id,
            "account_status":       p.account_status,
            "name":                 p.name,
            "registered_email":     p.registered_email,
            "phone":                p.phone,
            "account_opening_date": p.account_opening_date,
            "portal_status":        p.portal_status,
            "deactivation_code":    p.deactivation_code,
            "deactivation_reason":  p.deactivation_reason,
            "demat_account_no":     p.demat_account_no,
            "trading_account_no":   p.trading_account_no,
            "raw":                  p.raw,
        }
    except Exception as exc:
        logger.error("[TOOL] get_customer_profile_full failed: %s", exc)
        return {"error": str(exc), "account_status": "active", "raw": {}}


def create_account_closure(
    sub_account_id: str,
    email: str,
    name: str,
    type_of_account_closure: str = "demat",
    dp_account_no: str = "",
) -> dict:
    """
    Submit an account-closure request to the closure API for the given account.
    Returns the API's raw response (used to detect already-closed / in-progress /
    eligible scenarios).

    Args:
        sub_account_id:          Customer sub-account ID (ent_id).
        email:                   Registered email (required by the API).
        name:                    Customer name (for remarks).
        type_of_account_closure: "demat" | "trading" | "demat_and_trading".
        dp_account_no:           Optional DP account number.

    Returns:
        The closure API response dict (e.g. {"api_response": {"reason": "..."}}).
    """
    try:
        from src.gateways.gateway_client import create_closure_request
        return create_closure_request(
            sub_account_id, email, name,
            type_of_account_closure=type_of_account_closure,
            dp_account_no=dp_account_no,
        )
    except Exception as exc:
        logger.error("[TOOL] create_account_closure failed: %s", exc)
        return {"api_response": {"reason": "error"}, "error": str(exc)}


# =============================================================================
# STATEMENT / REPORTS — Basic Auth (REPORTS_USERNAME / REPORTS_PASSWORD)
# =============================================================================

def request_statement(
    sub_account_id: str,
    report_name: str,
    start_date: str,
    end_date: str,
    endpoint: str = "exports",
) -> dict:
    """
    Submit a statement generation request. The Reports API processes the job and
    emails the report directly to the customer's registered email address.
    Use this for Tax Statement, Ledger Report, DP Holdings, P&L Statement,
    Capital Gains, Account Statement, Portfolio Holding Statement.

    Auth: Basic Auth — REPORTS_USERNAME and REPORTS_PASSWORD required.

    Args:
        sub_account_id: Customer sub-account ID.
        report_name:    Report name (e.g. "Tax Statement", "Ledger Report").
        start_date:     Start date in DD-MM-YYYY format.
        end_date:       End date in DD-MM-YYYY format.
        endpoint:       "exports" for general reports, "sendmail" for segment reports.

    Returns:
        {"success": bool, "masked_email": str, "error": str | None}
    """
    try:
        from src.gateways.statement_api import request_statement_fireandforget
        result = request_statement_fireandforget(
            sub_account_id=sub_account_id,
            api_jobname=report_name,
            endpoint=endpoint,
            start_date=start_date,
            end_date=end_date,
        )
        return {
            "success":      result.success,
            "masked_email": result.masked_email,
            "error":        result.error_message or None,
        }
    except Exception as exc:
        logger.error("[TOOL] request_statement failed: %s", exc)
        return {"success": False, "masked_email": "", "error": str(exc)}


def get_ledger_balance(
    sub_account_id: str,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Fetch the customer's ledger balance for a date range.
    Used in the Brokerage & Charges flow to determine if the customer
    has outstanding charges (closingBalance > 0) or no dues.

    Auth: Basic Auth — REPORTS_USERNAME and REPORTS_PASSWORD required.

    Args:
        sub_account_id: Customer sub-account ID.
        start_date:     Start date in DD-MM-YYYY format.
        end_date:       End date in DD-MM-YYYY format.

    Returns:
        {
          "success": bool,
          "opening_balance": str,
          "closing_balance": str,   # positive = customer owes money
          "emargin_balance": str
        }
    """
    try:
        from src.gateways.statement_api import get_ledger
        return get_ledger(sub_account_id, start_date, end_date)
    except Exception as exc:
        logger.error("[TOOL] get_ledger_balance failed: %s", exc)
        return {"success": False, "opening_balance": "0", "closing_balance": "0",
                "emargin_balance": "0", "error": str(exc)}


def send_dp_bill(
    sub_account_id: str,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Send the DP (Depository Participant) charges bill to the customer's
    registered email address. Used in the Brokerage & Charges flow
    after the customer selects the DP Charges option.

    Auth: Basic Auth — REPORTS_USERNAME and REPORTS_PASSWORD required.

    Args:
        sub_account_id: Customer sub-account ID.
        start_date:     Statement start date in DD-MM-YYYY format.
        end_date:       Statement end date in DD-MM-YYYY format.

    Returns:
        {"success": bool, "masked_email": str}
    """
    try:
        from src.gateways.statement_api import send_dp_bill as _api
        return _api(sub_account_id, start_date, end_date)
    except Exception as exc:
        logger.error("[TOOL] send_dp_bill failed: %s", exc)
        return {"success": False, "masked_email": "", "error": str(exc)}


# =============================================================================
# ORDER / TRADE BOOK — X-SubAccountID header only (no auth)
# =============================================================================

def get_todays_orders(sub_account_id: str, segment: str) -> dict:
    """
    Fetch today's executed trades for a customer in the given market segment.
    Used in the Order Status flow when the customer wants to check today's orders.
    Filters the trade book response to only return trades from today's date.

    Auth: No authorization header required — noauth per API spec.

    Args:
        sub_account_id: Customer sub-account ID.
        segment:        Market segment — one of: "Equity", "Commodity", "Derivatives", "Mutual Funds".

    Returns:
        {"found": bool, "orders": list, "count": int, "segment": str}
    """
    try:
        from src.gateways.order_api import get_todays_orders as _api
        return _api(sub_account_id, segment)
    except Exception as exc:
        logger.error("[TOOL] get_todays_orders failed: %s", exc)
        return {"found": False, "orders": [], "count": 0, "segment": segment, "error": str(exc)}


def send_order_history_email(
    sub_account_id: str,
    segment: str,
    date_str: str,
    end_date: str = "",
) -> dict:
    """
    Fetch order history for a date range and send it to the customer's
    registered email address. Used in the Order Status flow when the customer
    selects Order History and picks a 30-day range.

    Auth: No authorization header required — noauth per API spec.

    Args:
        sub_account_id: Customer sub-account ID.
        segment:        Market segment — one of: "Equity", "Commodity", "Derivatives", "Mutual Funds".
        date_str:       Range start in DD-MM-YYYY format.
        end_date:       Range end in DD-MM-YYYY (defaults to date_str for a single day).

    Returns:
        {"success": bool, "masked_email": str}
    """
    try:
        from src.gateways.order_api import send_order_history_email as _api
        return _api(sub_account_id, segment, date_str, end_date)
    except Exception as exc:
        logger.error("[TOOL] send_order_history_email failed: %s", exc)
        return {"success": False, "masked_email": "", "error": str(exc)}


# =============================================================================
# Registries
# =============================================================================

# Raw Python callables, by name — used by run_tool() for direct, reliable
# invocation from the deterministic flows (no framework unwrapping needed).
_RAW_FUNCS = [
    request_authentication,
    escalate_to_agent,
    get_account_status,
    get_customer_profile,
    get_customer_profile_full,
    create_account_closure,
    get_todays_orders,
    send_order_history_email,
    request_statement,
    get_ledger_balance,
    send_dp_bill,
]

TOOL_FUNCS: dict = {fn.__name__: fn for fn in _RAW_FUNCS}

# LangChain-native tools (StructuredTool) built from the same functions — used
# by the fully-agentic path to bind to the model. Names + docstrings are taken
# from the functions, so the LLM sees the identical tool surface as Strands.
ALL_TOOLS = [
    StructuredTool.from_function(fn, name=fn.__name__, description=(fn.__doc__ or "").strip())
    for fn in _RAW_FUNCS
]
