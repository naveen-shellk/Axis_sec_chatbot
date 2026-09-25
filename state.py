"""
chatbot_web/state.py
---------------------
WebState TypedDict — the single shared state that flows through every
LangGraph node in the web channel graph.

Mirrors EmailState from agent/state.py but scoped to web channel:
  - Multi-turn conversation (session_id = Conversationid from Simcomm)
  - Pre-login only: no customer authentication at start
  - Intents: bank_query | how_to_trade | need_more_help | edit_profile
  - No iLeverage CRM (NSR / SendMail) — those are email-only
  - Escalation = eventid 1002 to WxCC (not UpdateEmailStatus)
"""

from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class WebIntent:
    BANK_QUERY    = "bank_query"
    HOW_TO_TRADE  = "how_to_trade"
    NEED_MORE_HELP = "need_more_help"
    EDIT_PROFILE  = "edit_profile"
    UNKNOWN       = "unknown"


class WebState(TypedDict):
    """
    Shared mutable context flowing through every LangGraph node.

    Populated in stages:
      session_node      → session_id, conversation_history, is_first_turn
      intent_node       → intent, intent_confidence
      flow nodes        → flow_state, collected_data, reply_message,
                          quick_reply_options, reply_sent, escalate
    """

    # ── Session identity ───────────────────────────────────────────────────
    session_id: str                          # = Conversationid from Simcomm
    customer_message: str                    # raw input this turn
    input_type: str                          # "quick_reply" | "free_text"

    # ── Conversation history (LangGraph-managed) ───────────────────────────
    messages: Annotated[list, add_messages]

    # ── Intent classification ──────────────────────────────────────────────
    intent: Optional[str]                    # WebIntent value
    intent_confidence: float                 # 0.0–1.0 from Haiku classifier

    # ── Flow routing ───────────────────────────────────────────────────────
    flow: Optional[str]                      # active flow name
    flow_state: str                          # step within the flow
    collected_data: dict[str, Any]           # data accumulated within a flow
    is_first_turn: bool                      # True → send greeting

    # ── Output (set by flow nodes, read by reply_node) ─────────────────────
    reply_message: str
    quick_reply_options: list[str]
    reply_sent: bool                         # True when reply is ready
    escalate: bool                           # True → return eventid 1002
    eventid: str                             # "1001" | "1002"

    # ── Token tracking ─────────────────────────────────────────────────────
    total_input_tokens: int
    total_output_tokens: int
    llm_call_count: int


def initial_web_state(
    session_id: str,
    customer_message: str,
    input_type: str = "free_text",
    # Restored from AgentCore Memory
    flow: str | None = None,
    flow_state: str = "start",
    collected_data: dict | None = None,
    is_first_turn: bool = True,
    history_messages: list | None = None,
) -> WebState:
    """
    Factory — build a clean or restored WebState.
    Called by the entrypoint on each AgentCore invocation.
    """
    return WebState(
        session_id=session_id,
        customer_message=customer_message,
        input_type=input_type,
        messages=history_messages or [],
        intent=None,
        intent_confidence=0.0,
        flow=flow,
        flow_state=flow_state,
        collected_data=collected_data or {},
        is_first_turn=is_first_turn,
        reply_message="",
        quick_reply_options=[],
        reply_sent=False,
        escalate=False,
        eventid="1001",
        total_input_tokens=0,
        total_output_tokens=0,
        llm_call_count=0,
    )
