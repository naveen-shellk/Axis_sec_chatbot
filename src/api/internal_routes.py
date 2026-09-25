"""
chatbot_web/src/api/internal_routes.py
-----------------------------------------
Internal test API — plain fields, no web envelope.
Used by Postman / curl / CI test suites before channel integration.

Routes:
  POST /internal/message      — Single turn, no auth required
  DELETE /internal/session    — Clear a session (test cleanup)
  GET  /internal/health       — Internal health check
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from entry.handler import handle_message
from models import InternalMessageRequest, InternalMessageResponse
from src.core.session_store import clear_session, get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/internal", tags=["Internal — Test Endpoints (no auth)"])


# ── POST /internal/message ────────────────────────────────────────────────────

class _Examples:
    greeting = {
        "Greeting (new session)": {
            "summary": "First message — get greeting + full menu",
            "value": {"conversation_id": "test-001", "raw_input": "hi", "input_type": "free_text"},
        },
        "Bank Query (button tap)": {
            "summary": "Pre-login — Bank Query",
            "value": {"conversation_id": "test-001", "raw_input": "Bank Query", "input_type": "quick_reply"},
        },
        "How To Trade (free text)": {
            "summary": "Pre-login — How To Trade via free text",
            "value": {"conversation_id": "test-002", "raw_input": "How do I place an intraday buy order?", "input_type": "free_text"},
        },
        "Need More Help — Escalation": {
            "summary": "Pre-login — Escalate to live agent (returns eventid 1002)",
            "value": {"conversation_id": "test-003", "raw_input": "Need More Help", "input_type": "quick_reply"},
        },
        "Edit Profile": {
            "summary": "Pre-login — Edit Profile deeplink",
            "value": {"conversation_id": "test-004", "raw_input": "Edit Profile", "input_type": "quick_reply"},
        },
        "Statement (post-login)": {
            "summary": "Post-login — Statement flow (sub_account_id required)",
            "value": {"conversation_id": "test-005", "raw_input": "Statement", "input_type": "quick_reply", "sub_account_id": "7032318"},
        },
        "Order Status (post-login)": {
            "summary": "Post-login — Order Status flow",
            "value": {"conversation_id": "test-006", "raw_input": "Order Status", "input_type": "quick_reply", "sub_account_id": "7032318"},
        },
        "Account Details (post-login)": {
            "summary": "Post-login — Account Details",
            "value": {"conversation_id": "test-007", "raw_input": "Account Details", "input_type": "quick_reply", "sub_account_id": "7032318"},
        },
        "IPO (post-login)": {
            "summary": "Post-login — IPO deeplink",
            "value": {"conversation_id": "test-008", "raw_input": "IPO", "input_type": "quick_reply", "sub_account_id": "7032318"},
        },
        "Brokerage (post-login)": {
            "summary": "Post-login — Brokerage & Charges",
            "value": {"conversation_id": "test-009", "raw_input": "Brokerage and Charges", "input_type": "quick_reply", "sub_account_id": "7032318"},
        },
        "Login Query (post-login)": {
            "summary": "Post-login — Login Query",
            "value": {"conversation_id": "test-010", "raw_input": "Login Query", "input_type": "quick_reply", "sub_account_id": "7032318"},
        },
        "Auth Required (no sub_account_id)": {
            "summary": "Post-login flow without auth — returns auth_required",
            "value": {"conversation_id": "test-011", "raw_input": "Statement", "input_type": "quick_reply"},
        },
    }


@router.post(
    "/message",
    summary="💬 Test a conversation turn (all flows)",
    description="""
Send a message and get a response. Use the same `conversation_id` across turns to maintain session.

**Pre-login flows** — no `sub_account_id` needed:
Bank Query, How To Trade, Need More Help, Edit Profile

**Post-login flows** — pass `sub_account_id` to simulate completed authentication:
Statement, IPO, Account Details, Brokerage, Login Query, Order Status

**Quick-reply simulation**: set `input_type: "quick_reply"` to simulate a button tap.

**Tip**: Use `conversation_id` consistently to test multi-turn flows like How To Trade (5+ turns).
""",
    response_model=InternalMessageResponse,
    responses={
        200: {
            "description": "Bot response",
            "content": {
                "application/json": {
                    "examples": {
                        "normal_reply": {
                            "summary": "Normal reply (eventid 1001)",
                            "value": {
                                "reply_message": "👋 Welcome to Axis Direct! How can I help you today?",
                                "quick_reply_options": ["Bank Query","How To Trade","Need More Help","Edit Profile","Statement","IPO","Account Details","Brokerage and Charges","Login Query","Order Status"],
                                "flow_state": "main_menu",
                                "status": "ok",
                                "eventid": "1001",
                                "debug_info": {},
                            },
                        },
                        "escalation": {
                            "summary": "Escalation (eventid 1002)",
                            "value": {
                                "reply_message": "I understand you need more help! Let me connect you to a live agent.",
                                "quick_reply_options": [],
                                "flow_state": "escalated",
                                "status": "escalate",
                                "eventid": "1002",
                                "debug_info": {},
                            },
                        },
                        "auth_required": {
                            "summary": "Auth required (post-login flow without sub_account_id)",
                            "value": {
                                "reply_message": "To access this feature, you need to verify your identity first.",
                                "quick_reply_options": [],
                                "flow_state": "auth_required",
                                "status": "auth_required",
                                "eventid": "1001",
                                "debug_info": {},
                            },
                        },
                    }
                }
            },
        }
    },
)
def message_endpoint(
    request: InternalMessageRequest = None,
) -> InternalMessageResponse:
    """Process one turn and return a plain response dict."""
    logger.info(
        "POST /internal/message conv=%s input_type=%s",
        request.conversation_id, request.input_type,
    )
    try:
        response = handle_message(
            conversation_id=request.conversation_id,
            raw_input=request.raw_input,
            input_type=request.input_type,
            sub_account_id=request.sub_account_id,
            event=request.event,
        )
        return response
    except Exception as exc:
        logger.exception("internal/message: unhandled error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── GET /internal/session ─────────────────────────────────────────────────────

@router.get(
    "/session/{conversation_id}",
    summary="Inspect session state",
    description="Returns the current session state for a conversation. Useful for debugging.",
    response_model=dict[str, Any],
)
def get_session_state(conversation_id: str) -> dict[str, Any]:
    state = get_session(conversation_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return state.model_dump()


# ── DELETE /internal/session ──────────────────────────────────────────────────

@router.delete(
    "/session/{conversation_id}",
    summary="Clear a session",
    description="Delete a session (test cleanup).",
    response_model=dict[str, str],
)
def delete_session(conversation_id: str) -> dict[str, str]:
    clear_session(conversation_id)
    return {"status": "cleared", "conversation_id": conversation_id}


# ── GET /internal/health ──────────────────────────────────────────────────────

@router.get("/health", summary="Internal health check", tags=["system"])
def internal_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "asl-web-chatbot-internal",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/stats", summary="Token usage and flow statistics")
def get_stats(last_n: int = 1000) -> dict[str, Any]:
    """
    Returns aggregated stats from logs/log.txt:
    total tokens, per-model breakdown, flow usage, latency, escalation count.
    """
    from src.utils.token_logger import get_stats as _get_stats
    return _get_stats(last_n_lines=last_n)


@router.get("/logs", summary="Recent log entries from logs/log.txt")
def get_logs(limit: int = 50) -> dict[str, Any]:
    """Returns the last N log entries from logs/log.txt as parsed JSON."""
    import json as _json
    from pathlib import Path
    log_file = Path(__file__).parent.parent.parent.parent / "logs" / "log.txt"
    if not log_file.exists():
        return {"entries": [], "total": 0, "log_file": str(log_file)}
    entries = []
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[-limit:]:
        stripped = line.strip()
        if stripped.startswith("JSON:"):
            try:
                entries.append(_json.loads(stripped[5:].strip()))
            except _json.JSONDecodeError:
                pass
    return {"entries": entries, "total": len(entries), "log_file": str(log_file)}
