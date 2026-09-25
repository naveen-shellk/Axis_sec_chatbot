"""
chatbot_web/src/core/strands_agent.py
---------------------------------------
Strands Agent instance for the web channel.

Pattern (same as chatbot/src/core/strands_agent.py):
  - Agent is instantiated once at module level
  - Flow handlers call helper functions (escalate / check_account)
    which invoke the agent with the tool name + args
  - The agent uses Haiku via BedrockModel — same model as conversation turns

Why Strands Agent here:
  - Tools decorated with @tool are registered with the agent
  - The agent handles tool invocation, retries, and result parsing
  - For post-login flows, the agent can call multiple tools in sequence
    without the flow handler managing that orchestration
  - Consistent with chatbot/ pattern — single framework across channels

AgentCore deployment:
  - When running inside AgentCore Runtime, the agent uses the same
    Bedrock credentials injected by the Runtime execution role
  - No explicit credentials needed in production (IAM role via container)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from strands import Agent
from strands.models import BedrockModel

from src.core.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

_MODEL_ID    = os.getenv("RESPONSE_MODEL_ID", "qwen.qwen3-235b-a22b-2507-v1:0")  # Qwen for responses
_REGION      = os.getenv("AWS_REGION", "ap-south-1")
_MAX_TOKENS  = int(os.getenv("CHATBOT_MAX_TOKENS", "512"))
_TEMPERATURE = float(os.getenv("CHATBOT_TEMPERATURE", "0.2"))

# ── Strands Agent (instantiated once) ────────────────────────────────────────

_model = BedrockModel(
    model_id=_MODEL_ID,
    region_name=_REGION,
    max_tokens=_MAX_TOKENS,
    temperature=_TEMPERATURE,
)

_AGENT_SYSTEM_PROMPT = (
    "You are the Axis Direct virtual assistant for the web (pre-login) channel. "
    "You help customers with: bank queries, how to trade, editing profile, account "
    "statements, IPO, account details, brokerage & charges, login queries, order "
    "status, and account closure.\n\n"
    "You have tools to fetch customer profiles, request statements, get orders, "
    "fetch ledger balances, send DP bills, create account-closure requests, and "
    "escalate to a live agent. DECIDE YOURSELF which tools to call and in what "
    "order to fulfil the customer's request. Call get_customer_profile_full first "
    "when you need account data.\n\n"
    "Rules:\n"
    "- For account-specific actions you need the customer's Sub-Account ID. If it "
    "is provided in the context, use it; never invent one.\n"
    "- If the customer explicitly asks for a human/live agent, call escalate_to_agent.\n"
    "- Be warm, concise, and professional. Reply in plain text (no markdown/JSON).\n"
    "- Do not fabricate data — only state what the tools return."
)

_agent = Agent(
    model=_model,
    tools=ALL_TOOLS,
    system_prompt=_AGENT_SYSTEM_PROMPT,
)


# ── Helper functions (called by flow handlers) ────────────────────────────────

def escalate(reason: str) -> dict[str, Any]:
    """
    Signal live agent escalation.
    Called by the need_more_help flow.

    Returns:
        {"escalate": True, "reason": str, "eventid": "1002"}
    """
    logger.info("[STRANDS] escalate: %s", reason)
    try:
        result = _agent.tool.escalate_to_agent(reason=reason)
        # Strands returns ToolResult — extract the content
        if hasattr(result, "content"):
            for block in result.content:
                if hasattr(block, "json"):
                    return block.json
        return {"escalate": True, "reason": reason, "eventid": "1002"}
    except Exception as exc:
        logger.error("[STRANDS] escalate failed: %s — returning direct result", exc)
        return {"escalate": True, "reason": reason, "eventid": "1002"}


def check_account(customer_id: str) -> dict[str, Any]:
    """
    Check account status via Strands agent.
    Used by post-login flows before any account action.

    Returns:
        {"action": "proceed"|"blocked", "tool_results": {"get_account_status": {...}}}
    """
    logger.info("[STRANDS] check_account: %s", customer_id)
    try:
        result = _agent.tool.get_account_status(customer_id=customer_id)
        status = "active"
        if hasattr(result, "content"):
            for block in result.content:
                if hasattr(block, "json"):
                    status = block.json.get("status", "active")

        action = "proceed" if status == "active" else "blocked"
        return {
            "action": action,
            "tool_results": {"get_account_status": {"status": status}},
        }
    except Exception as exc:
        logger.error("[STRANDS] check_account failed: %s — defaulting active", exc)
        return {
            "action": "proceed",
            "tool_results": {"get_account_status": {"status": "active"}},
        }


# ── Generic tool bridge ───────────────────────────────────────────────────────
# Flow handlers call these to invoke a tool THROUGH the Strands agent (agent-
# orchestrated), instead of calling the gateway functions directly. The flow
# sequence is still owned by the handler (deterministic, compliance-safe) — the
# agent just mediates the tool invocation. Falls back to the direct gateway call
# if the agent path errors, so behaviour never breaks.

def _extract_tool_json(result: Any) -> dict[str, Any] | None:
    """
    Pull the JSON payload out of a Strands ToolResult.

    Strands returns a dict like:
      {"status": "success", "content": [{"text": "<json string>"}]}
    or (older) an object with .content blocks exposing .json / .text.
    """
    import json as _json

    def _parse(s: str) -> dict[str, Any] | None:
        try:
            v = _json.loads(s)
            return v if isinstance(v, dict) else None
        except Exception:
            return None

    # dict-shaped ToolResult (current Strands)
    if isinstance(result, dict):
        if result.get("status") == "success" and isinstance(result.get("content"), list):
            for block in result["content"]:
                if isinstance(block, dict) and "text" in block:
                    parsed = _parse(block["text"])
                    if parsed is not None:
                        return parsed
        # already a plain payload dict
        if "eventid" in result or "status" in result and "content" not in result:
            return result
        return None

    # object-shaped ToolResult (older Strands)
    if hasattr(result, "content"):
        for block in result.content:
            if hasattr(block, "json") and block.json is not None:
                return block.json
            if hasattr(block, "text") and block.text:
                parsed = _parse(block.text)
                if parsed is not None:
                    return parsed
    return None


def run_tool(tool_name: str, /, **kwargs: Any) -> dict[str, Any] | None:
    """
    Invoke a registered tool by name through the Strands agent.
    Returns the tool's JSON dict, or None if the agent path failed
    (caller should then use its own direct fallback).
    """
    logger.info("[STRANDS] run_tool %s args=%s", tool_name, list(kwargs))
    try:
        tool_fn = getattr(_agent.tool, tool_name)
        result  = tool_fn(**kwargs)
        return _extract_tool_json(result)
    except Exception as exc:
        logger.error("[STRANDS] run_tool %s failed: %s — caller will fall back", tool_name, exc)
        return None


def run_agent_turn(prompt: str) -> dict[str, Any]:
    """
    Fully agentic turn: hand the prompt to the Strands agent and let IT decide
    which tools to call, in what order, and when it's done (ReAct-style loop).

    The agent reasons over the registered tool set (profile, statement, orders,
    ledger, DP bill, closure, escalation) and returns a final natural-language
    answer. We surface that text plus a best-effort escalation flag.

    Returns:
        {"message": str, "escalate": bool}
    """
    logger.info("[STRANDS] run_agent_turn (LLM decides tools) prompt_len=%d", len(prompt))
    try:
        result = _agent(prompt)
        # AgentResult stringifies to the final assistant text.
        text = str(result).strip()
        escalate = "1002" in text or "escalate to a live agent" in text.lower()

        # ── Token accounting ──────────────────────────────────────────────────
        # The Strands loop makes its own LLM calls (not via conversation.py), so
        # pull the real usage from AgentResult.metrics.accumulated_usage and push
        # it into the per-turn accumulator so log.txt records correct totals.
        in_tok = out_tok = 0
        try:
            usage = result.metrics.accumulated_usage
            in_tok  = int(usage.get("inputTokens", 0))
            out_tok = int(usage.get("outputTokens", 0))
        except Exception:
            pass

        try:
            from src.core.conversation import _turn_tokens
            _turn_tokens["input_tokens"]           += in_tok
            _turn_tokens["output_tokens"]          += out_tok
            _turn_tokens["response_input_tokens"]  += in_tok
            _turn_tokens["response_output_tokens"] += out_tok
            _turn_tokens["llm_call_count"]         += 1
        except Exception as exc:
            logger.warning("[STRANDS] token accounting failed: %s", exc)

        return {"message": text, "escalate": escalate,
                "input_tokens": in_tok, "output_tokens": out_tok}
    except Exception as exc:
        logger.error("[STRANDS] run_agent_turn failed: %s", exc)
        return {"message": "", "escalate": False, "error": str(exc)}


def warmup() -> None:
    """
    Pre-initialise the Strands agent + BedrockModel credentials at startup so the
    FIRST tool dispatch (run_tool) doesn't pay the ~0.4-0.6s one-time agent-init
    cost on a customer's request. Uses a trivial no-op tool (escalate reason).
    """
    try:
        import time as _t
        _t0 = _t.perf_counter()
        # One real (harmless) tool dispatch forces the agent + BedrockModel
        # credential/model init. escalate_to_agent makes NO external call.
        run_tool("escalate_to_agent", reason="__warmup__")
        logger.info("[STRANDS] agent warmup complete in %d ms",
                    int((_t.perf_counter() - _t0) * 1000))
    except Exception as exc:
        logger.warning("[STRANDS] agent warmup failed: %s", exc)


def get_profile(sub_account_id: str):
    """
    Fetch the full customer profile THROUGH the Strands agent and return it as a
    CustomerProfile object (same type the flows expect). Falls back to the direct
    typed gateway call if the agent path fails, so flows never break.
    """
    from src.gateways.customer_api import CustomerProfile, get_customer_profile as _direct
    import time as _t
    _t0 = _t.perf_counter()

    data = run_tool("get_customer_profile_full", sub_account_id=sub_account_id)
    logger.info("[TIMING] profile_fetch_ms=%d", int((_t.perf_counter() - _t0) * 1000))
    if data and not data.get("error"):
        try:
            return CustomerProfile(
                sub_account_id       = data.get("sub_account_id", sub_account_id),
                account_status       = data.get("account_status", "active"),
                name                 = data.get("name", ""),
                registered_email     = data.get("registered_email", ""),
                phone                = data.get("phone", ""),
                account_opening_date = data.get("account_opening_date", ""),
                portal_status        = data.get("portal_status", 0),
                deactivation_code    = data.get("deactivation_code", ""),
                deactivation_reason  = data.get("deactivation_reason", ""),
                demat_account_no     = data.get("demat_account_no", ""),
                trading_account_no   = data.get("trading_account_no", ""),
                raw                  = data.get("raw", {}) or {},
            )
        except Exception as exc:
            logger.warning("[STRANDS] get_profile rebuild failed: %s — direct fallback", exc)
    return _direct(sub_account_id)
