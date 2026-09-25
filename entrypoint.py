"""
chatbot_web/entrypoint.py
--------------------------
Production entrypoint for AWS BedrockAgentCore Runtime.

CMD target in the Dockerfile — NOT uvicorn.
For local dev use: python app.py

Payload from Cisco (Simcomm/WebX via API Gateway):
  {
    "Conversationid": "CSR0122XNRRMENM4",
    "Message": "What are your support hours?",
    "Event": "Incoming message",
    "Channel": "WEB",
    "timestamp": "2026-08-13T09:15:00.000Z"
  }

Token tracking:
  reset_turn_tokens() clears the accumulator before each turn.
  Every call_llm() / call_intent_llm() adds to _turn_tokens in conversation.py.
  get_turn_tokens() returns the totals after handle_message() completes.
  write_turn_log() appends a JSON line to chatbot_web/logs/log.txt.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from typing import Any

from dotenv import load_dotenv

# Load config/.env — contains client account credentials for Bedrock access
load_dotenv(
    os.path.join(os.path.dirname(__file__), "config", ".env"),
    override=False,
)

import structlog
from bedrock_agentcore import BedrockAgentCoreApp, RequestContext

from entry.handler import handle_message
from src.core.conversation import reset_turn_tokens, get_turn_tokens
from src.core.session_store import get_session
from src.utils.token_logger import write_turn_log
from src.core.llm import call_llm

# ── Structured logging ────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO"))
    ),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)

_log = structlog.get_logger(__name__)

_INTENT_MODEL   = os.getenv("INTENT_MODEL_ID",   "anthropic.claude-3-haiku-20240307-v1:0")
_RESPONSE_MODEL = os.getenv("RESPONSE_MODEL_ID",  "qwen.qwen3-235b-a22b-2507-v1:0")

# ── BedrockAgentCoreApp ───────────────────────────────────────────────────────
app = BedrockAgentCoreApp()

# Warm the AgentCore Memory clients at container startup so the first customer
# request doesn't pay the ~1-2s boto3 client-init cost.
try:
    from src.core.session_store import warmup as _warmup_memory
    _warmup_memory()
except Exception as _exc:
    _log.warning("memory warmup skipped: %s", _exc)
try:
    from src.core.strands_agent import warmup as _warmup_agent
    _warmup_agent()
except Exception as _exc:
    _log.warning("agent warmup skipped: %s", _exc)


def _generate_escalation_summary(session, conversation_id: str, last_message: str) -> str:
    """
    Generate a brief conversation summary using Qwen for the live agent.
    Uses the full conversation history so the agent has complete context.
    Falls back to a simple text summary if LLM fails.
    """
    if not session or not session.history:
        return f"Customer requested live agent. Last message: {last_message}"

    # Build full conversation transcript from all history
    transcript_lines = []
    for h in session.history:
        role = "Customer" if h.get("role") == "user" else "Bot"
        content = h.get("content", "")
        if content:
            transcript_lines.append(f"{role}: {content}")

    transcript = "\n".join(transcript_lines[-20:])  # all turns, cap at 20 lines

    _SUMMARY_SYS = """\
You are summarising a chatbot conversation for a live customer service agent at Axis Direct.
Write a brief 2-3 sentence summary covering:
1. What the customer wanted
2. What was done / attempted
3. Why escalation was requested

Be concise and factual. Do not use markdown. Plain text only.
"""
    try:
        messages = [{"role": "user", "content": [{"text": f"Conversation:\n{transcript}\n\nProvide a brief summary for the live agent."}]}]
        result = call_llm(_SUMMARY_SYS, messages, expect_json=False)
        summary = result.get("text", "").strip()
        return summary if summary else f"Customer escalated from {session.flow or 'main menu'} flow."
    except Exception:
        return f"Customer escalated from {session.flow or 'main menu'} flow. Last message: {last_message}"


def _chunk_text(text: str, size: int = 24):
    """Yield word-boundary-aware chunks of `text` for streaming."""
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


@app.entrypoint
def invoke(
    payload: dict[str, Any],
    context: "RequestContext | None" = None,
):
    """
    Called by AgentCore Runtime for every customer message turn.

    Streaming mode (payload.stream == true):
      Yields SSE-style dicts:
        {"type": "chunk", "text": "..."}   — partial message text
        {"type": "done",  ...full metadata...}  — final structured payload

    Non-streaming mode (default):
      Returns a single structured dict (backward compatible with Simcomm).
    """
    stream = bool(payload.get("stream") or payload.get("Stream"))
    if stream:
        return _invoke_stream(payload, context)
    return _invoke_sync(payload, context)


def _invoke_sync(
    payload: dict[str, Any],
    context: "RequestContext | None" = None,
) -> dict[str, Any]:
    """Non-streaming path — single structured response."""
    t_start = time.time()

    conversation_id = (
        payload.get("Conversationid") or
        payload.get("conversation_id") or
        payload.get("session_id") or "unknown"
    )
    message = (
        payload.get("Message") or
        payload.get("message") or
        payload.get("raw_input") or ""
    )
    event = payload.get("Event", "Incoming message")

    _quick_events = {"quick_reply", "quickreply", "postback", "button_click"}
    input_type = "quick_reply" if event.lower().replace(" ", "_") in _quick_events else "free_text"

    sub_account_id = payload.get("sub_account_id") or payload.get("SubAccountId")

    _log.info(
        "invoke", conv=conversation_id, evt=event,
        input_type=input_type, msg_len=len(message),
    )

    # ── Reset token accumulator for this turn ─────────────────────────────────
    reset_turn_tokens()

    try:
        response = handle_message(
            conversation_id=conversation_id,
            raw_input=message,
            input_type=input_type,
            sub_account_id=sub_account_id,
            event=event,
        )
    except Exception as exc:
        _log.error("invoke.error", conv=conversation_id, error=traceback.format_exc())
        t_tokens = get_turn_tokens()
        latency_ms = int((time.time() - t_start) * 1000)
        write_turn_log(
            conversation_id=conversation_id,
            flow=None, flow_state="error", intent=None,
            customer_message=message, bot_reply="",
            intent_model=_INTENT_MODEL, response_model=_RESPONSE_MODEL,
            intent_input_tokens=t_tokens["input_tokens"],
            intent_output_tokens=t_tokens["output_tokens"],
            response_input_tokens=0, response_output_tokens=0,
            llm_call_count=t_tokens["llm_call_count"],
            latency_ms=latency_ms,
            escalated=True, sub_account_id=sub_account_id,
            request_time=t_start, response_time=time.time(),
        )
        return {
            "eventid": "1002",
            "conversation_id": conversation_id,
            "reason": f"Internal error: {exc}",
            "context": {"error": str(exc)},
        }

    # ── Read accumulated tokens for this turn ─────────────────────────────────
    t_tokens   = get_turn_tokens()
    total_in   = t_tokens["input_tokens"]
    total_out  = t_tokens["output_tokens"]
    call_count = t_tokens["llm_call_count"]

    # Token split: intent tokens are tracked separately in handler.py via
    # _tt accumulator. If only 1 call happened it was either intent-only or
    # response-only. We read the per-model split from the accumulator directly.
    from src.core.conversation import _turn_tokens as _tt_raw
    intent_in    = _tt_raw.get("intent_input_tokens",    0)
    intent_out   = _tt_raw.get("intent_output_tokens",   0)
    response_in  = _tt_raw.get("response_input_tokens",  total_in  - intent_in)
    response_out = _tt_raw.get("response_output_tokens", total_out - intent_out)

    latency_ms = int((time.time() - t_start) * 1000)
    t_response = time.time()

    session_after = get_session(conversation_id)

    _log.info(
        "invoke.done", conv=conversation_id,
        eventid=response.eventid, flow_state=response.flow_state,
        input_tokens=total_in, output_tokens=total_out,
        total_tokens=total_in + total_out, latency_ms=latency_ms,
    )

    # ── Write to chatbot_web/logs/log.txt ─────────────────────────────────────
    write_turn_log(
        conversation_id=conversation_id,
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
        sub_account_id=sub_account_id or (session_after.sub_account_id if session_after else None),
        request_time=t_start,
        response_time=t_response,
    )

    # ── Build response ────────────────────────────────────────────────────────
    if response.eventid == "1002" or response.status == "escalate":
        # Generate summary from full conversation history for live agent context
        summary = _generate_escalation_summary(
            session_after, conversation_id, message
        )
        return {
            "eventid":         "1002",
            "conversation_id": conversation_id,
            "reason":          "Customer requested live agent support",
            "context": {
                "last_flow":    response.flow_state,
                "last_message": message,
                "summary":      summary,
                "sub_account_id": sub_account_id or (session_after.sub_account_id if session_after else ""),
            },
        }

    return {
        "eventid":         "1001",
        "conversation_id": conversation_id,
        "message":         response.reply_message,
        "quick_replies": [
            {"identifier": opt, "title": opt}
            for opt in response.quick_reply_options
        ],
        "flow_state": response.flow_state,
        "status":     response.status,
    }


def _invoke_stream(
    payload: dict[str, Any],
    context: "RequestContext | None" = None,
):
    """
    Streaming path — yields message text in chunks, then a final metadata event.

    Event shapes:
      {"type": "start", "conversation_id": "..."}
      {"type": "chunk", "text": "partial text "}
      {"type": "done",  "eventid": "1001", "message": "<full>", "quick_replies": [...],
                        "flow_state": "...", "status": "...", "latency_ms": N}

    NOTE: handle_message() runs fully first (flows build templated text +
    quick replies + state), then the assembled message is streamed in chunks.
    This gives a streaming UX while preserving the structured flow contract.
    """
    t_start = time.time()

    conversation_id = (
        payload.get("Conversationid") or
        payload.get("conversation_id") or
        payload.get("session_id") or "unknown"
    )
    message = (
        payload.get("Message") or
        payload.get("message") or
        payload.get("raw_input") or ""
    )
    event = payload.get("Event", "Incoming message")

    _quick_events = {"quick_reply", "quickreply", "postback", "button_click"}
    input_type = "quick_reply" if event.lower().replace(" ", "_") in _quick_events else "free_text"
    sub_account_id = payload.get("sub_account_id") or payload.get("SubAccountId")

    _log.info("invoke.stream", conv=conversation_id, evt=event, msg_len=len(message))

    reset_turn_tokens()

    yield {"type": "start", "conversation_id": conversation_id}

    try:
        response = handle_message(
            conversation_id=conversation_id,
            raw_input=message,
            input_type=input_type,
            sub_account_id=sub_account_id,
            event=event,
        )
    except Exception as exc:
        _log.error("invoke.stream.error", conv=conversation_id, error=traceback.format_exc())
        yield {
            "type": "done",
            "eventid": "1002",
            "conversation_id": conversation_id,
            "reason": f"Internal error: {exc}",
            "context": {"error": str(exc)},
        }
        return

    session_after = get_session(conversation_id)

    # ── Escalation → single done event ────────────────────────────────────────
    if response.eventid == "1002" or response.status == "escalate":
        summary = _generate_escalation_summary(session_after, conversation_id, message)
        latency_ms = int((time.time() - t_start) * 1000)
        _write_stream_log(conversation_id, response, message, sub_account_id,
                          session_after, t_start, latency_ms)
        yield {
            "type": "done",
            "eventid": "1002",
            "conversation_id": conversation_id,
            "reason": "Customer requested live agent support",
            "context": {
                "last_flow":    response.flow_state,
                "last_message": message,
                "summary":      summary,
                "sub_account_id": sub_account_id or (session_after.sub_account_id if session_after else ""),
            },
        }
        return

    # ── Stream the message text ────────────────────────────────────────────
    # handle_message() already generated the full message (needed for buttons +
    # flow state + session save). We replay it as chunks for the streaming UX.
    # For genuinely long free-text (how-to-trade instructions, closure), this
    # is streamed word-by-word so the client renders progressively.
    full_message = response.reply_message or ""
    for chunk in _chunk_text(full_message, size=12):
        yield {"type": "chunk", "text": chunk}

    latency_ms = int((time.time() - t_start) * 1000)
    _write_stream_log(conversation_id, response, message, sub_account_id,
                      session_after, t_start, latency_ms)

    # ── Final metadata event ──────────────────────────────────────────────────
    yield {
        "type":            "done",
        "eventid":         "1001",
        "conversation_id": conversation_id,
        "message":         full_message,
        "quick_replies": [
            {"identifier": opt, "title": opt}
            for opt in response.quick_reply_options
        ],
        "flow_state": response.flow_state,
        "status":     response.status,
        "latency_ms": latency_ms,
    }


def _write_stream_log(conversation_id, response, message, sub_account_id,
                      session_after, t_start, latency_ms):
    """Shared token/turn log writer for the streaming path."""
    t_tokens = get_turn_tokens()
    from src.core.conversation import _turn_tokens as _tt_raw
    intent_in    = _tt_raw.get("intent_input_tokens",    0)
    intent_out   = _tt_raw.get("intent_output_tokens",   0)
    response_in  = _tt_raw.get("response_input_tokens",  t_tokens["input_tokens"]  - intent_in)
    response_out = _tt_raw.get("response_output_tokens", t_tokens["output_tokens"] - intent_out)
    write_turn_log(
        conversation_id=conversation_id,
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
        llm_call_count=t_tokens["llm_call_count"],
        latency_ms=latency_ms,
        escalated=(response.status == "escalate"),
        auth_required=(response.status == "auth_required"),
        sub_account_id=sub_account_id or (session_after.sub_account_id if session_after else None),
        request_time=t_start,
        response_time=time.time(),
    )


if __name__ == "__main__":
    _log.info("chatbot_web AgentCore Runtime entrypoint starting")
    app.run()
