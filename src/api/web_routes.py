"""
chatbot_web/src/api/web_routes.py
-----------------------------------
Single public API endpoint for Simcomm/WebX integration.

ENDPOINT:   POST /api/chat
AUTH:       Authorization: Bearer <customer-issued token>
CONTRACT:   Exact Simcomm payload and response format from integration doc.

Request (Simcomm → our backend):
  {
    "Conversationid": "CSR0122XNRRMENM4",
    "Message": "What are your support hours?",
    "Event": "Incoming message",
    "Channel": "WEB",
    "timestamp": "2026-08-13T09:15:00.000Z"
  }

Response — Normal (eventid 1001):
  {
    "eventid": "1001",
    "conversation_id": "CSR0122XNRRMENM4",
    "message": "We are available 24/7...",
    "messagetype": "text",
    "timestamp": "2026-08-13T09:15:01.200Z",
    "customer": {"email": "", "phone": ""},
    "quickReplies": {
      "reference": "main_menu",
      "options": [
        {"type": "quickReplyPostback", "identifier": "bank_query",
         "title": "Bank Query", "imageUrl": "", "payload": {"payload": {"action": "bank_query"}}}
      ]
    }
  }

Response — Escalation (eventid 1002):
  {
    "eventid": "1002",
    "conversation_id": "CSR0122XNRRMENM4",
    "timestamp": "...",
    "customparam1": "escalation_reason:need_more_help",
    "customer": {"email": "", "phone": ""},
    "context": {"summary": "...", "customerIntent": "need_more_help"}
  }
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import StreamingResponse

from entry.handler import handle_message
from src.core.conversation import reset_turn_tokens, get_turn_tokens
from src.core.session_store import get_session
from src.utils.token_logger import write_turn_log
from models import (
    CustomerInfo,
    QuickReplies,
    QuickReplyOption,
    WebChatRequest,
    WebChatResponse,
    WebEscalationResponse,
    WebEndChatAck,
)

_INTENT_MODEL   = os.getenv("INTENT_MODEL_ID",   "anthropic.claude-3-haiku-20240307-v1:0")
_RESPONSE_MODEL = os.getenv("RESPONSE_MODEL_ID",  "qwen.qwen3-235b-a22b-2507-v1:0")


def _generate_escalation_summary(conversation_id: str, last_message: str) -> str:
    """
    Generate a concise LLM summary of the full conversation for the live agent.
    Called only when eventid == 1002 (escalation).
    Falls back to a structured text summary if LLM fails.
    """
    state = get_session(conversation_id)
    if not state or not state.history:
        return f"Customer requested live agent. Last message: {last_message}"

    # Build readable conversation transcript
    turns = []
    for h in state.history:
        role    = h.get("role", "")
        content = h.get("content", "").strip()
        if not content:
            continue
        label = "Customer" if role == "user" else "Bot"
        turns.append(f"{label}: {content}")

    if not turns:
        return f"Customer requested live agent. Last message: {last_message}"

    transcript = "\n".join(turns[-20:])  # last 10 turns max

    _SUMMARY_SYS = """\
You are summarizing a customer service chat conversation for a live agent who is about to take over.
Write a concise 2-3 sentence summary covering:
1. What the customer needed help with
2. What was discussed or attempted
3. Why they are being escalated to a live agent

Be factual, specific, and professional. No filler phrases.
Return plain text only — no JSON, no markdown."""

    try:
        from src.core.llm import call_intent_llm
        from src.core.conversation import _turn_tokens as _tt
        result = call_intent_llm(
            _SUMMARY_SYS,
            [{"role": "user", "content": [{"text": f"Conversation transcript:\n{transcript}"}]}],
        )
        # Track summary LLM tokens in the accumulator
        _tt["input_tokens"]   += result.get("input_tokens",  0)
        _tt["output_tokens"]  += result.get("output_tokens", 0)
        _tt["llm_call_count"] += 1
        summary = result.get("text", "").strip()
        if summary:
            return summary
    except Exception as exc:
        logger.warning("[SUMMARY] LLM summary failed: %s", exc)

    # Fallback: structured text summary from session state
    flow   = state.flow or "unknown"
    lines  = [f"Customer was in '{flow}' flow."]
    if turns:
        lines.append(f"Last customer message: {last_message}")
    lines.append("Customer requested live agent support.")
    return " ".join(lines)


logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chatbot API — Simcomm Integration"])

_WEB_API_TOKEN: str = os.getenv("WEB_API_TOKEN", "dev-token-change-in-prod")
_bearer = HTTPBearer(auto_error=False)


def _verify_token(credentials: HTTPAuthorizationCredentials | None = Security(_bearer)):
    """Validate Bearer token. Always enforced — no dev bypass."""
    if credentials is None or credentials.credentials != _WEB_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")
    return credentials


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _chunk_text(text: str, size: int = 12):
    """Yield word-boundary chunks of `text` for streaming the message out."""
    words = text.split(" ")
    buf = ""
    for w in words:
        if len(buf) + len(w) + 1 > size and buf:
            yield buf + " "
            buf = w
        else:
            buf = f"{buf} {w}".strip()
    if buf:
        yield buf


def _sse(event: dict) -> str:
    """Format a dict as one SSE `data:` line."""
    return f"data: {json.dumps(event)}\n\n"


def _build_quick_replies(
    options: list[str],
    reference: str = "",
) -> QuickReplies | None:
    """Build quickReplies object in exact Simcomm format."""
    if not options:
        return None
    seen = set()
    deduped = [o for o in options if o and not (o in seen or seen.add(o))]
    if not deduped:
        return None
    return QuickReplies(
        reference=reference,
        options=[
            QuickReplyOption(
                type="quickReplyPostback",
                identifier=label.lower().replace(" ", "_").replace("&", "and")[:50],
                title=label,
                imageUrl="",
                payload={"payload": {"action": label.lower().replace(" ", "_").replace("&", "and")}},
            )
            for label in deduped
        ],
    )


def _get_customer_info(conversation_id: str) -> CustomerInfo:
    """
    Build customer object from session.
    Pre-login: empty strings.
    Post-login: masked email and phone from cached profile.
    """
    state = get_session(conversation_id)
    if state is None:
        return CustomerInfo()
    return CustomerInfo(
        email=state.customer_email or "",
        phone=state.customer_phone or "",
    )


# ── POST /api/chat ────────────────────────────────────────────────────────────

@router.post(
    "/api/chat",
    summary="Chat — Send a customer message",
    description="""
**Single endpoint for Simcomm/WebX integration.**

Handles every customer conversation turn. Returns a normal bot reply (`eventid: 1001`)
or an escalation signal (`eventid: 1002`) when the customer needs a live agent.

**Pre-login flows** (no authentication required):
Bank Query, How To Trade, Need More Help, Edit Profile

**Post-login flows** (requires authenticated session with sub_account_id):
Statement, IPO, Account Details, Brokerage & Charges, Login Query, Order Status

**First message behaviour:** A greeting with the full menu is returned automatically
on the first message of any new `Conversationid`.
""",
    responses={
        200: {
            "description": "Normal bot reply (eventid 1001) or escalation (eventid 1002)",
            "content": {
                "application/json": {
                    "examples": {
                        "normal_reply": {
                            "summary": "Normal reply — eventid 1001",
                            "value": {
                                "eventid": "1001",
                                "conversation_id": "CSR0122XNRRMENM4",
                                "message": "Welcome to Axis Direct! How can I help you today?",
                                "messagetype": "text",
                                "timestamp": "2026-08-13T09:15:01.200Z",
                                "customer": {"email": "", "phone": ""},
                                "quickReplies": {
                                    "reference": "main_menu",
                                    "options": [
                                        {"type": "quickReplyPostback", "identifier": "bank_query",
                                         "title": "Bank Query", "imageUrl": "",
                                         "payload": {"payload": {"action": "bank_query"}}},
                                        {"type": "quickReplyPostback", "identifier": "how_to_trade",
                                         "title": "How To Trade", "imageUrl": "",
                                         "payload": {"payload": {"action": "how_to_trade"}}},
                                    ],
                                },
                            },
                        },
                        "escalation": {
                            "summary": "Escalation — eventid 1002",
                            "value": {
                                "eventid": "1002",
                                "conversation_id": "CSR0122XNRRMENM4",
                                "timestamp": "2026-08-13T09:15:02.000Z",
                                "customparam1": "escalation_reason:need_more_help",
                                "customer": {"email": "", "phone": ""},
                                "context": {
                                    "summary": "Customer requested live agent support",
                                    "customerIntent": "need_more_help",
                                },
                            },
                        },
                    }
                }
            },
        },
        401: {"description": "Unauthorized — invalid or missing Bearer token"},
    },
)
def chat(
    body: WebChatRequest,
    _: HTTPAuthorizationCredentials = Depends(_verify_token),
) -> JSONResponse:
    """
    Process one customer conversation turn.

    **Request:** Exact Simcomm WebX payload format.
    **Response:** eventid 1001 (reply) or eventid 1002 (escalate to WxCC).
    """
    t_start  = time.time()
    conv_id  = body.Conversationid
    message  = body.Message
    event    = body.Event

    _quick_events = {"quick_reply", "quickreply", "postback", "button_click"}
    input_type = (
        "quick_reply"
        if event.lower().replace(" ", "_") in _quick_events
        else "free_text"
    )

    logger.info(
        "POST /api/chat conv=%s event=%r input_type=%s msg_len=%d",
        conv_id, event, input_type, len(message),
    )

    # Reset token accumulator for this turn
    reset_turn_tokens()

    try:
        response = handle_message(
            conversation_id=conv_id,
            raw_input=message,
            input_type=input_type,
            sub_account_id=None,
            event=event,           # forwarded — needed for need_more_help no-response handling
        )
    except Exception as exc:
        logger.exception("chat error conv=%s: %s", conv_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    t_response   = time.time()
    latency_ms   = round((t_response - t_start) * 1000, 2)  # float ms, e.g. 0.45ms or 1432.1ms
    t_tokens     = get_turn_tokens()
    total_in     = t_tokens["input_tokens"]
    total_out    = t_tokens["output_tokens"]
    call_count   = t_tokens["llm_call_count"]

    # Use the REAL per-model token counts tracked in the turn accumulator
    # (handler.py records intent_* for Haiku, response_* for Qwen). Only fall
    # back to an even split if those weren't populated for some reason.
    from src.core.conversation import _turn_tokens as _tt_raw
    intent_in    = _tt_raw.get("intent_input_tokens",    0)
    intent_out   = _tt_raw.get("intent_output_tokens",   0)
    response_in  = _tt_raw.get("response_input_tokens",  0)
    response_out = _tt_raw.get("response_output_tokens", 0)
    if (intent_in + intent_out + response_in + response_out) == 0 and (total_in + total_out) > 0:
        # Nothing was attributed — attribute everything to response as a fallback.
        response_in, response_out = total_in, total_out

    session_after = get_session(conv_id)

    # Write to chatbot_web/logs/log.txt
    try:
        write_turn_log(
            conversation_id=conv_id,
            flow=session_after.flow if session_after else None,
            flow_state=response.flow_state,
            intent=session_after.flow if session_after else None,
            customer_message=message,
            bot_reply=response.reply_message,
            intent_model=_INTENT_MODEL,
            response_model=_RESPONSE_MODEL,
            intent_input_tokens=intent_in,
            intent_output_tokens=intent_out,
            response_input_tokens=response_in,
            response_output_tokens=response_out,
            llm_call_count=call_count,
            latency_ms=latency_ms,
            escalated=(response.status == "escalate"),
            auth_required=(response.status == "auth_required"),
            sub_account_id=session_after.sub_account_id if session_after else None,
            request_time=t_start,
            response_time=t_response,
        )
    except Exception as exc:
        logger.warning("[LOG] write_turn_log failed: %s", exc)

    ts            = _now_iso()
    customer_info = _get_customer_info(conv_id)
    quick_replies = _build_quick_replies(response.quick_reply_options, reference=response.flow_state)

    # ── Escalation — eventid 1002 ─────────────────────────────────────────────
    if response.eventid == "1002" or response.status == "escalate":
        # Generate full conversation summary for the live agent
        summary = _generate_escalation_summary(conv_id, message)
        logger.info("POST /api/chat conv=%s → ESCALATION summary=%r", conv_id, summary[:100])

        esc = WebEscalationResponse(
            conversation_id=conv_id,
            timestamp=ts,
            customparam1=f"escalation_reason:{response.flow_state}",
            customer=customer_info,
            context={
                "summary":        summary,
                "customerIntent": response.flow_state,
                "last_message":   message,
                "flow":           session_after.flow if session_after else "unknown",
            },
        )
        return JSONResponse(content=esc.model_dump())

    # ── Normal reply — eventid 1001 ───────────────────────────────────────────
    reply = WebChatResponse(
        conversation_id=conv_id,
        message=response.reply_message,
        messagetype="text",
        timestamp=ts,
        customer=customer_info,
        quickReplies=quick_replies,
    )
    return JSONResponse(content=reply.model_dump())


# ── POST /api/chat/end ────────────────────────────────────────────────────────

@router.post(
    "/api/chat/end",
    summary="Chat — End-chat notification",
    description="Called by Simcomm when the customer closes the chat widget. Clears server-side session.",
    response_model=WebEndChatAck,
)
def chat_end(
    body: WebChatRequest,
    _: HTTPAuthorizationCredentials = Depends(_verify_token),
) -> WebEndChatAck:
    """Clear session when the chat widget closes."""
    from src.core.session_store import clear_session
    logger.info("POST /api/chat/end conv=%s", body.Conversationid)
    try:
        clear_session(body.Conversationid)
    except Exception as exc:
        logger.warning("chat/end session clear error: %s", exc)
    return WebEndChatAck()


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    summary="Health check",
    description="Liveness probe — no auth required. Used by API Gateway health checks.",
    tags=["System"],
)
def health():
    return {"status": "ok", "service": "asl-web-chatbot", "timestamp": _now_iso()}
