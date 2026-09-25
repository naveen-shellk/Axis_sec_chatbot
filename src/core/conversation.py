"""
chatbot_web/src/core/conversation.py
--------------------------------------
LLM conversation utility.

Every flow handler calls run_conversation_turn() with a system_override
scoped to that flow's decision node. The function builds the context block,
appends the customer message, calls Qwen, and returns the parsed JSON dict.

Token tracking:
  A per-request accumulator (_turn_tokens) collects tokens from every LLM call
  within a single handle_message() turn. The entrypoint reads it via
  get_turn_tokens() after handle_message() returns, then resets it for the
  next request via reset_turn_tokens().
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.llm import call_llm, call_intent_llm

logger = logging.getLogger(__name__)

# ── Per-turn token accumulator ────────────────────────────────────────────────
_turn_tokens: dict[str, int] = {
    "input_tokens":           0,
    "output_tokens":          0,
    "llm_call_count":         0,
    # Per-model split (set by handler.py when intent LLM is called)
    "intent_input_tokens":    0,
    "intent_output_tokens":   0,
    "response_input_tokens":  0,
    "response_output_tokens": 0,
}


def reset_turn_tokens() -> None:
    """Call at the START of each handle_message() turn."""
    for k in _turn_tokens:
        _turn_tokens[k] = 0


def get_turn_tokens() -> dict[str, int]:
    """Call at the END of each handle_message() turn to read totals."""
    return dict(_turn_tokens)


_DEFAULT_SYSTEM = """\
You are a helpful virtual assistant for Axis Direct (Axis Securities Limited).
You are embedded in the customer's website via a live chat widget.
Tone: professional, warm, concise. Maximum one short paragraph per reply.
Never ask for sensitive information (Aadhaar, PAN, card number, password, CVV).
This is a PRE-LOGIN chatbot — you do NOT have access to account data.

Return ONLY valid JSON, no markdown fences:
{
  "message":       "<what to say to the customer>",
  "quick_replies": ["<label 1>", ...],
  "intent":        "<intent or null>",
  "flow_action":   "<action>",
  "reasoning":     "<one sentence, not shown to customer>"
}
"""


def run_conversation_turn(
    state_data: dict[str, Any],
    history: list[dict[str, str]],
    customer_message: str,
    backend_data: dict[str, Any] | None = None,
    system_override: str | None = None,
    track_tokens: bool = True,
    state_obj: Any = None,  # unused — kept for API compatibility
    use_haiku: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run one LLM turn (Qwen by default, or Haiku if use_haiku=True) and accumulate
    tokens into the turn accumulator.

    Args:
        state_data:       {"flow", "flow_state", ...}
        history:          list of {"role": "user"|"assistant", "content": "..."}
        customer_message: raw user input this turn
        backend_data:     extra facts injected into the context block
        system_override:  custom system prompt for this specific flow node
        track_tokens:     accumulate tokens when True (default)
        use_haiku:        use Haiku model instead of Qwen (for fast template output)

    Returns:
        {
          "message", "quick_replies", "intent",
          "flow_action", "reasoning",
          "input_tokens", "output_tokens"
        }
    """
    system = system_override or _DEFAULT_SYSTEM

    # Build context block
    lines: list[str] = []
    if state_data.get("flow"):
        lines.append(f"Active flow: {state_data['flow']}")
    if state_data.get("flow_state"):
        lines.append(f"Flow step: {state_data['flow_state']}")
    if state_data.get("is_first_message"):
        lines.append("This is the customer's FIRST message — greet warmly.")
    for k, v in (backend_data or {}).items():
        lines.append(f"{k}: {v}")

    context = "\n".join(lines) if lines else "Pre-login web channel."

    # Build message list — last 6 messages (~3 turns) for response LLM
    messages: list[dict[str, Any]] = []
    for h in history[-6:]:
        role    = h.get("role", "user")
        content = h.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": [{"text": content}]})

    # Bedrock Converse requires the messages list to start with a user message.
    # Drop any leading assistant messages that could appear when history ends
    # with a bot reply and the next turn's user message hasn't been appended yet.
    while messages and messages[0]["role"] != "user":
        messages.pop(0)

    messages.append({
        "role": "user",
        "content": [{"text": f"[CONTEXT]\n{context}\n\n[CUSTOMER MESSAGE]\n{customer_message}"}],
    })

    if use_haiku:
        result = call_intent_llm(system, messages)
    else:
        result = call_llm(system, messages, expect_json=True)
    parsed = result.get("parsed") or {}

    in_tok  = result.get("input_tokens",  0)
    out_tok = result.get("output_tokens", 0)

    # Accumulate into turn-level totals
    if track_tokens:
        _turn_tokens["input_tokens"]   += in_tok
        _turn_tokens["output_tokens"]  += out_tok
        _turn_tokens["llm_call_count"] += 1
        if use_haiku:
            _turn_tokens["intent_input_tokens"]  += in_tok
            _turn_tokens["intent_output_tokens"] += out_tok
        else:
            _turn_tokens["response_input_tokens"]  += in_tok
            _turn_tokens["response_output_tokens"] += out_tok
        logger.debug(
            "[CONV] flow=%s step=%s model=%s tokens: %d in / %d out (turn total: %d in / %d out)",
            state_data.get("flow"), state_data.get("flow_state"),
            result.get("model_id", "default"),
            in_tok, out_tok,
            _turn_tokens["input_tokens"], _turn_tokens["output_tokens"],
        )

    message_text = parsed.get("message")
    if not message_text:
        raw_fallback = result.get("text", "").strip()
        if raw_fallback and not raw_fallback.startswith("{"):
            message_text = raw_fallback
        else:
            message_text = "I'm sorry, something went wrong. Please try again."

    quick_replies = parsed.get("quick_replies")
    if not quick_replies:
        if state_data.get("flow") == "account_details" or state_data.get("flow_state") == "session_end_response":
            quick_replies = ["Go back to main menu", "End Chat"]
        else:
            quick_replies = []

    return {
        "message":       message_text,
        "quick_replies": quick_replies,
        "intent":        parsed.get("intent"),
        "flow_action":   parsed.get("flow_action", "reprompt"),
        "reasoning":     parsed.get("reasoning", ""),
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
    }
