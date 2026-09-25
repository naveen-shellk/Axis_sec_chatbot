"""
chatbot_web/src/gateways/statement_api.py
-------------------------------------------
Statement/reports API calls via AgentCore Gateway MCP.

All calls use gateway_client.py which routes to the correct
Gateway tool based on endpoint type:
  exports  → statement-api-v2___request_statement
  sendmail → statement-api-v2___request_statement_comtrack
  oneclick → statement-api-v2___request_statement_reports

Report catalogue (from statement/handler.py):
  Tax Reports:     Tax Statement, P&L Statement, Capital Gains          → exports
  Demat Reports:   DP Holdings                                          → exports
                   DP Transaction Statement                             → oneclick
                   CML Report (NSDL)                                    → sendmail
  Trading Reports: Ledger Report, Account Statement                     → exports
                   Contract Notes, AGTS, Equity Margin                  → oneclick
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.gateways.gateway_client import (
    request_statement as _gw_request,
    get_ledger        as _gw_ledger,
    send_dp_bill      as _gw_dp_bill,
    get_masked_email,
)


@dataclass
class StatementResult:
    success:       bool
    masked_email:  str = ""
    report_id:     int = 0
    error_message: str = ""


def request_statement_fireandforget(
    sub_account_id: str,
    api_jobname:    str,
    endpoint:       str,        # "exports" | "sendmail" | "oneclick"
    start_date:     str,        # DD-MM-YYYY
    end_date:       str,
    dp_account_no:  str = "",
) -> StatementResult:
    """
    Submit statement request via AgentCore Gateway MCP.
    Selects the correct tool based on endpoint:
      exports  → statement-api-v2___request_statement
      sendmail → statement-api-v2___request_statement_comtrack
      oneclick → statement-api-v2___request_statement_reports
    Returns StatementResult with masked_email for confirmation message.
    """
    masked = get_masked_email(sub_account_id)

    try:
        raw = _gw_request(
            sub_account_id=sub_account_id,
            report_name=api_jobname,
            start_date=start_date,
            end_date=end_date,
            endpoint=endpoint,
            dp_account_no=dp_account_no,
        )
    except Exception as exc:
        return StatementResult(
            success=False,
            masked_email=masked,
            error_message=str(exc),
        )

    # Check for Gateway-level errors (VPC routing etc.)
    if isinstance(raw, dict):
        text = raw.get("text", "")
        if "OpenAPIClientException" in text or "Error executing HTTP request" in text:
            return StatementResult(
                success=False,
                masked_email=masked,
                error_message=f"Gateway VPC error: {text[:120]}",
            )
        if "error" in raw and raw.get("error"):
            return StatementResult(
                success=False,
                masked_email=masked,
                error_message=str(raw["error"]),
            )
        # Extract report_id if present (exports endpoint returns it)
        report_id = 0
        try:
            data = raw.get("data", {})
            report_id = data.get("report_id", 0) if isinstance(data, dict) else 0
        except Exception:
            pass
        return StatementResult(success=True, masked_email=masked, report_id=report_id)

    return StatementResult(
        success=False,
        masked_email=masked,
        error_message=f"Unexpected response: {str(raw)[:120]}",
    )


def get_ledger(
    sub_account_id: str,
    start_date:     str,
    end_date:       str,
) -> dict[str, Any]:
    """Fetch ledger balance via Gateway MCP — statement-api-v2___get_ledger."""
    return _gw_ledger(sub_account_id, start_date, end_date)


def send_dp_bill(
    sub_account_id: str,
    start_date:     str,
    end_date:       str,
    dp_id:          str = "",
) -> dict[str, Any]:
    """Send DP bill email via Gateway MCP — statement-api-v2___request_statement_comtrack."""
    return _gw_dp_bill(sub_account_id, start_date, end_date, dp_id)


__all__ = [
    "StatementResult",
    "request_statement_fireandforget",
    "get_ledger",
    "send_dp_bill",
]
