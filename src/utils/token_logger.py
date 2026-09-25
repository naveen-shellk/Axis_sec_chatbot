"""
chatbot_web/src/utils/token_logger.py
---------------------------------------
Token usage tracker and conversation logger.

Writes every conversation turn to chatbot_web/logs/log.txt as structured JSON.
Each entry captures:
  - conversation_id, timestamp, flow, flow_state
  - customer message preview (truncated for PII safety)
  - intent classified
  - model used (intent = Haiku, response = Qwen)
  - input_tokens, output_tokens per LLM call
  - total_tokens for the turn
  - latency_ms, request_time, response_time
  - escalated / auth_required flags

One JSON object per line — easy to grep, tail, and parse.

Log location: chatbot_web/logs/log.txt
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Log file location ─────────────────────────────────────────────────────────
# Resolves to chatbot_web/logs/log.txt regardless of working directory
_LOG_DIR  = Path(__file__).parent.parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "log.txt"


def _ensure_log_dir() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def write_turn_log(
    conversation_id: str,
    flow: str | None,
    flow_state: str,
    intent: str | None,
    customer_message: str,
    bot_reply: str,
    intent_model: str,
    response_model: str,
    intent_input_tokens: int,
    intent_output_tokens: int,
    response_input_tokens: int,
    response_output_tokens: int,
    llm_call_count: int,
    latency_ms: int,
    escalated: bool = False,
    auth_required: bool = False,
    sub_account_id: str | None = None,
    request_time: float | None = None,   # epoch seconds — when request arrived
    response_time: float | None = None,  # epoch seconds — when response was sent
) -> None:
    """
    Write one conversation turn log entry to chatbot_web/logs/log.txt.
    Called by entrypoint.py after each handle_message() call.
    """
    _ensure_log_dir()

    total_input  = intent_input_tokens  + response_input_tokens
    total_output = intent_output_tokens + response_output_tokens
    total_tokens = total_input + total_output

    _now     = time.time()
    _req_ts  = request_time  or (_now - latency_ms / 1000)
    _resp_ts = response_time or _now

    entry = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "conversation_id":    conversation_id,
        "sub_account_id":     sub_account_id or "anonymous",
        "flow":               flow or "main_menu",
        "flow_state":         flow_state,
        "intent":             intent or "unknown",
        "escalated":          escalated,
        "auth_required":      auth_required,

        # Message (truncate to 200 chars — avoid PII in logs)
        "customer_message_len":     len(customer_message),
        "customer_message_preview": customer_message[:200],
        "bot_reply_len":            len(bot_reply),

        # Token breakdown — both models tracked separately
        "tokens": {
            "intent_model":           intent_model,
            "intent_input_tokens":    intent_input_tokens,
            "intent_output_tokens":   intent_output_tokens,
            "intent_total":           intent_input_tokens + intent_output_tokens,

            "response_model":         response_model,
            "response_input_tokens":  response_input_tokens,
            "response_output_tokens": response_output_tokens,
            "response_total":         response_input_tokens + response_output_tokens,

            "total_input_tokens":  total_input,
            "total_output_tokens": total_output,
            "total_tokens":        total_tokens,
            "llm_call_count":      llm_call_count,
        },

        # Performance
        "latency_ms":    latency_ms,
        "latency_sec":   round(latency_ms / 1000, 3),
        "request_time":  datetime.fromtimestamp(_req_ts,  tz=timezone.utc).isoformat(),
        "response_time": datetime.fromtimestamp(_resp_ts, tz=timezone.utc).isoformat(),
    }

    latency_sec = latency_ms / 1000

    # ── Human-readable block + compact JSON on same file ─────────────────────
    lines = [
        "=" * 70,
        f"  Timestamp       : {entry['timestamp']}",
        f"  Conversation ID : {conversation_id}",
        f"  Sub Account ID  : {sub_account_id or 'anonymous'}",
        f"  Flow            : {flow or 'main_menu'}  |  State: {flow_state}",
        f"  Intent          : {intent or 'unknown'}",
        f"  Escalated       : {escalated}  |  Auth Required: {auth_required}",
        f"  Customer Msg    : {customer_message[:120]}{'...' if len(customer_message) > 120 else ''}",
        f"  Bot Reply Len   : {len(bot_reply)} chars",
        "  " + "-" * 40,
        f"  Request Time    : {entry['request_time']}",
        f"  Response Time   : {entry['response_time']}",
        f"  Latency         : {latency_sec:.3f}s  ({latency_ms}ms)",
        "  " + "-" * 40,
        f"  Intent Model    : {intent_model}",
        f"    Input Tokens  : {intent_input_tokens}",
        f"    Output Tokens : {intent_output_tokens}",
        f"    Subtotal      : {intent_input_tokens + intent_output_tokens}",
        f"  Response Model  : {response_model}",
        f"    Input Tokens  : {response_input_tokens}",
        f"    Output Tokens : {response_output_tokens}",
        f"    Subtotal      : {response_input_tokens + response_output_tokens}",
        "  " + "-" * 40,
        f"  Total Input     : {total_input}  |  Total Output : {total_output}",
        f"  Total Tokens    : {total_tokens}  |  LLM Calls    : {llm_call_count}",
        f"  JSON: {json.dumps(entry, ensure_ascii=False)}",
        "",
    ]

    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.debug("[TOKEN_LOGGER] Wrote log entry conv=%s tokens=%d latency=%.2fs",
                     conversation_id, total_tokens, latency_sec)
    except Exception as exc:
        logger.warning("[TOKEN_LOGGER] Failed to write log entry: %s", exc)


def get_log_path() -> str:
    """Return the absolute path to log.txt."""
    return str(_LOG_FILE.resolve())


def get_stats(last_n_lines: int = 1000) -> dict[str, Any]:
    """
    Read the last N log entries and compute summary statistics.
    Used by GET /internal/stats endpoint.
    """
    _ensure_log_dir()
    if not _LOG_FILE.exists():
        return {
            "error": "No log file yet",
            "log_file": str(_LOG_FILE.resolve()),
            "total_entries": 0,
        }

    entries = []
    try:
        with open(_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-last_n_lines:]:
            stripped = line.strip()
            # Parse from the embedded JSON line: "  JSON: {...}"
            if stripped.startswith("JSON:"):
                try:
                    entries.append(json.loads(stripped[5:].strip()))
                except json.JSONDecodeError:
                    pass
    except Exception as exc:
        return {"error": str(exc)}

    if not entries:
        return {"total_entries": 0, "log_file": str(_LOG_FILE.resolve())}

    total_tokens  = sum(e.get("tokens", {}).get("total_tokens",       0) for e in entries)
    total_input   = sum(e.get("tokens", {}).get("total_input_tokens",  0) for e in entries)
    total_output  = sum(e.get("tokens", {}).get("total_output_tokens", 0) for e in entries)
    escalations   = sum(1 for e in entries if e.get("escalated"))
    auth_required = sum(1 for e in entries if e.get("auth_required"))
    latencies     = [e["latency_ms"] for e in entries if e.get("latency_ms")]
    avg_latency   = round(sum(latencies) / len(latencies), 1) if latencies else 0

    flow_counts:   dict[str, int] = {}
    intent_counts: dict[str, int] = {}
    for e in entries:
        f = e.get("flow", "unknown")
        i = e.get("intent", "unknown")
        flow_counts[f]   = flow_counts.get(f, 0) + 1
        intent_counts[i] = intent_counts.get(i, 0) + 1

    return {
        "total_entries":       len(entries),
        "total_tokens":        total_tokens,
        "total_input_tokens":  total_input,
        "total_output_tokens": total_output,
        "avg_latency_ms":      avg_latency,
        "min_latency_ms":      min(latencies) if latencies else 0,
        "max_latency_ms":      max(latencies) if latencies else 0,
        "escalations":         escalations,
        "auth_required_count": auth_required,
        "flow_breakdown":      dict(sorted(flow_counts.items(),   key=lambda x: -x[1])),
        "intent_breakdown":    dict(sorted(intent_counts.items(), key=lambda x: -x[1])),
        "log_file":            str(_LOG_FILE.resolve()),
        "log_size_kb":         round(_LOG_FILE.stat().st_size / 1024, 1),
    }
