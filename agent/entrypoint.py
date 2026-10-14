"""
agent/entrypoint.py
-------------------
Thin AgentCore Runtime entry point for the Axis Direct chatbot.

Responsibility: ONLY request → response. It receives one invocation payload,
calls the channel-agnostic brain (agent.handler.handle_message), and returns a
Simcomm-shaped JSON response. All routing / flow / LLM / KB logic lives in the
agent package — this file adds no business logic.

Local testing does NOT use this file; it runs channels/web/app.py (uvicorn).
This entry is used only when deployed to AWS Bedrock AgentCore Runtime, which
is why the bedrock-agentcore SDK is imported LAZILY (inside __main__), so the
rest of the agent package imports fine locally without that dependency.

Payload (same contract as POST /api/chat):
    {
      "Conversationid": "...",
      "Message": "...",
      "Event": "Incoming message",
      "Channel": "WEB",
      "Client_ID": "<sub-account id, optional>"
    }

Response (eventid 1001 normal reply):
    {
      "eventid": "1001",
      "conversation_id": "...",
      "message": "...",
      "messagetype": "text",
      "quickReplies": {"reference": "<flow_state>", "options": [ ... ]}
    }
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

# Project root = chatbot_langchain/ (one level up from agent/). Put it on
# sys.path so `import agent.*` resolves when AgentCore runs this module, and
# load config/.env from the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, "config", ".env"), override=True)

from agent.handler import handle_message

logger = logging.getLogger(__name__)

_QUICK_EVENTS = {"quick_reply", "quickreply", "postback", "button_click"}


def _input_type(event: str) -> str:
    return ("quick_reply"
            if (event or "").lower().replace(" ", "_") in _QUICK_EVENTS
            else "free_text")


def _build_quick_replies(options: list[str], reference: str) -> dict | None:
    """Simcomm quickReplies object (same shape the web channel emits)."""
    seen, deduped = set(), []
    for opt in options or []:
        if opt and opt not in seen:
            seen.add(opt)
            deduped.append(opt)
    if not deduped:
        return None
    return {
        "reference": reference or "",
        "options": [
            {
                "type": "quickReplyPostback",
                "identifier": opt,
                "title": opt,
                "imageUrl": "",
                "payload": {"payload": {"action": opt}},
            }
            for opt in deduped
        ],
    }


def process_turn(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Core request → response. Pure function of the payload: no AgentCore SDK
    needed, so this is unit-testable and reusable. Returns the Simcomm JSON.
    """
    conv_id   = payload.get("Conversationid") or payload.get("conversation_id") or ""
    message   = payload.get("Message") or payload.get("message") or ""
    event     = payload.get("Event", "Incoming message")
    client_id = (payload.get("Client_ID") or "").strip() or None

    response = handle_message(
        conversation_id=conv_id,
        raw_input=message,
        input_type=_input_type(event),
        sub_account_id=client_id,
        event=event,
    )

    # Escalation → eventid 1002
    if response.eventid == "1002" or response.status == "escalate":
        return {
            "eventid": "1002",
            "conversation_id": conv_id,
            "customparam1": f"escalation_reason:{response.flow_state}",
            "context": {
                "customerIntent": response.flow_state,
                "last_message": message,
            },
        }

    # Normal reply → eventid 1001
    out: dict[str, Any] = {
        "eventid": "1001",
        "conversation_id": conv_id,
        "message": response.reply_message,
        "messagetype": "text",
    }
    qr = _build_quick_replies(response.quick_reply_options, response.flow_state)
    if qr:
        out["quickReplies"] = qr
    return out


# ── AgentCore Runtime entry (deploy-time only) ────────────────────────────────
# The bedrock-agentcore SDK is imported LAZILY here so importing agent.entrypoint
# (or running the local web channel) never requires the SDK to be installed.
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from bedrock_agentcore import BedrockAgentCoreApp  # lazy: deploy-time dep

    app = BedrockAgentCoreApp()

    @app.entrypoint
    def invoke(payload: dict[str, Any], *_args, **_kwargs) -> dict[str, Any]:
        try:
            return process_turn(payload or {})
        except Exception as exc:  # never crash the runtime on a single turn
            logger.exception("entrypoint error")
            return {
                "eventid": "1001",
                "conversation_id": (payload or {}).get("Conversationid", ""),
                "message": "Sorry, something went wrong. Please try again.",
                "messagetype": "text",
            }

    logger.info("agent AgentCore Runtime entrypoint starting")
    app.run()
