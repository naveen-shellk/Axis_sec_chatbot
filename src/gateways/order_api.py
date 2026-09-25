"""
chatbot_web/src/gateways/order_api.py
---------------------------------------
All calls route through AgentCoreGatewayClient — see gateway_client.py.
Re-exports the same interface so flow handler imports keep working.
"""

from __future__ import annotations

from typing import Any

from src.gateways.gateway_client import (
    get_todays_orders         as _gw_orders,
    send_order_history_email  as _gw_order_email,
    format_orders_for_chat,
)


def get_todays_orders(sub_account_id: str, segment_label: str) -> dict[str, Any]:
    """Fetch today's trades via AgentCore Gateway."""
    return _gw_orders(sub_account_id, segment_label)


def send_order_history_email(
    sub_account_id: str,
    segment_label:  str,
    date_str:       str,
) -> dict[str, Any]:
    """Send order history email via AgentCore Gateway."""
    return _gw_order_email(sub_account_id, segment_label, date_str)


__all__ = [
    "get_todays_orders",
    "send_order_history_email",
    "format_orders_for_chat",
]
