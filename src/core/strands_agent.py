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

_agent = Agent(
    model=_model,
    tools=ALL_TOOLS,
    system_prompt=(
        "You are a backend orchestrator for the Axis Direct web chatbot. "
        "Use the available tools to complete requested actions. "
        "Return results as JSON. Never ask the customer questions — "
        "that is handled by the conversation layer."
    ),
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
