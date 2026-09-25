"""
chatbot_web/models.py
---------------------
All shared Pydantic models for the web channel.

Single API contract: POST /api/chat
  Request:  Simcomm WebX payload (Conversationid, Message, Event, Channel, timestamp)
  Response: Simcomm response contract (eventid, message, quickReplies, customer, etc.)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Session State (internal — not part of API contract)
# ---------------------------------------------------------------------------

class SessionState(BaseModel):
    """
    Per-conversation state persisted between turns via AgentCore Memory / Postgres.
    Anonymous pre-login: sub_account_id is None.
    Authenticated post-login: sub_account_id is set after auth flow.
    """
    conversation_id: str

    # ── Customer identity (set after in-chat auth) ─────────────────────────
    sub_account_id: str | None = None     # None = pre-login / unauthenticated
    customer_name:  str | None = None
    customer_email: str | None = None     # used in response customer object
    customer_phone: str | None = None     # used in response customer object

    # ── Flow routing ──────────────────────────────────────────────────────
    flow: Literal[
        "bank_query", "how_to_trade", "need_more_help", "edit_profile",
        "statement", "ipo", "account_details", "brokerage", "login_query",
        "order_status", "closure",
        # Transient state used by the entry handler while collecting the
        # Sub-Account ID inline. Must be allowed here so the session can
        # round-trip through AgentCore Memory (which re-validates on read).
        "awaiting_sub_account_id",
    ] | None = None
    flow_state: str = "start"

    # ── In-flight data ────────────────────────────────────────────────────
    collected_data: dict[str, Any] = Field(default_factory=dict)

    # ── API cache ─────────────────────────────────────────────────────────
    customer_profile: dict[str, Any] | None = None

    # ── LLM history ───────────────────────────────────────────────────────
    history: list[dict[str, str]] = Field(default_factory=list)

    # ── Control flags ─────────────────────────────────────────────────────
    escalate: bool = False

    # ── Token tracking ────────────────────────────────────────────────────
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    llm_call_count: int = 0


# ---------------------------------------------------------------------------
# Account Status
# ---------------------------------------------------------------------------

class AccountStatus(str, Enum):
    ACTIVE      = "active"
    DEACTIVATED = "deactivated"
    PURGED      = "purged"


# ---------------------------------------------------------------------------
# INBOUND REQUEST — exact Simcomm WebX payload contract
# POST /api/chat
# Authorization: Bearer <customer-issued token>
# ---------------------------------------------------------------------------

class WebChatRequest(BaseModel):
    """
    Inbound message from Simcomm/WebX via API Gateway.

    Sample Request (from Simcomm integration doc):
        POST https://<endpoint>/api/chat
        Authorization: Bearer <token>
        {
          "Conversationid": "CSR0122XNRRMENM4",
          "Message": "What are your support hours?",
          "Event": "Incoming message",
          "Channel": "WEB",
          "timestamp": "2026-08-13T09:15:00.000Z"
        }
    """
    Conversationid: str = Field(..., description="Unique session ID from WebX widget")
    Message:        str = Field(..., description="Customer's raw input text")
    Event:          str = Field("Incoming message", description="Event type from Simcomm")
    Channel:        str = Field("WEB", description="Always WEB for this integration")
    timestamp:      str = Field("", description="ISO-8601 client timestamp")

    @property
    def conversation_id(self) -> str:
        return self.Conversationid

    @property
    def message(self) -> str:
        return self.Message


# ---------------------------------------------------------------------------
# OUTBOUND RESPONSE — exact Simcomm response contract
# ---------------------------------------------------------------------------

class QuickReplyOption(BaseModel):
    """Single quick-reply button — matches Simcomm quickReplies.options[] shape."""
    type:       str = Field("quickReplyPostback")
    identifier: str = Field(..., description="Action identifier sent back on tap")
    title:      str = Field(..., description="Label shown on the button")
    imageUrl:   str = Field("")
    payload:    dict[str, Any] = Field(default_factory=dict)


class QuickReplies(BaseModel):
    """quickReplies object — matches Simcomm contract exactly."""
    reference: str = Field("", description="Context reference for this set of options")
    options:   list[QuickReplyOption] = Field(default_factory=list)


class CustomerInfo(BaseModel):
    """
    customer object in every response.
    Pre-login: empty strings.
    Post-login: filled from customer profile (email masked, phone masked).
    """
    email: str = Field("", description="Masked registered email (pre-login: empty)")
    phone: str = Field("", description="Masked registered mobile (pre-login: empty)")


class WebChatResponse(BaseModel):
    """
    Normal bot reply — eventid 1001.

    Sample Response (from Simcomm integration doc):
        {
          "eventid": "1001",
          "conversation_id": "CSR0122XNRRMENM4",
          "message": "We are available 24/7. Want to see your statements?",
          "messagetype": "text",
          "timestamp": "2026-08-13T09:15:01.200Z",
          "customer": {
            "email": "jane.doe@example.com",
            "phone": "+91XXXXXXXXXX"
          },
          "quickReplies": {
            "reference": "support_hours_followup",
            "options": [
              {
                "type": "quickReplyPostback",
                "identifier": "statements",
                "title": "Statements",
                "imageUrl": "",
                "payload": { "payload": { "action": "statements" } }
              }
            ]
          }
        }
    """
    eventid:         str          = Field("1001")
    conversation_id: str
    message:         str
    messagetype:     str          = Field("text")
    timestamp:       str          = Field("")
    customer:        CustomerInfo = Field(default_factory=CustomerInfo)
    quickReplies:    QuickReplies | None = Field(None)


class WebEscalationResponse(BaseModel):
    """
    Escalation to live agent — eventid 1002.
    Returned when customer triggers Need More Help or chatbot cannot resolve.
    Simcomm routes this to Webex Contact Center (WxCC).
    """
    eventid:         str              = Field("1002")
    conversation_id: str
    timestamp:       str              = Field("")
    customparam1:    str              = Field("")
    customer:        CustomerInfo     = Field(default_factory=CustomerInfo)
    context:         dict[str, str]   = Field(default_factory=dict)


class WebEndChatAck(BaseModel):
    """Response to end-chat notification."""
    status: str = "acknowledged"


# ---------------------------------------------------------------------------
# Internal test model (dev only — not part of Simcomm contract)
# ---------------------------------------------------------------------------

class InternalMessageRequest(BaseModel):
    """Internal dev/test endpoint — simulates the Simcomm payload without the envelope."""
    conversation_id: str
    raw_input:       str
    input_type:      Literal["quick_reply", "free_text"] = "free_text"
    sub_account_id:  str | None = None  # simulates post-auth state
    event:           str = "Incoming message"  # pass "no_response" to test timeout logic


class InternalMessageResponse(BaseModel):
    """Internal test response."""
    reply_message:        str
    quick_reply_options:  list[str]     = Field(default_factory=list)
    flow_state:           str
    status:               str
    eventid:              str           = Field("1001")
    debug_info:           dict[str, Any] = Field(default_factory=dict)
