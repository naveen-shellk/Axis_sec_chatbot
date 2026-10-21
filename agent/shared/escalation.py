"""
agent/shared/escalation.py
--------------------------
Shared live-agent escalation summary.

When the bot hands a conversation to a human (WxCC), the live agent needs a
SHORT, USEFUL brief: what the customer actually wants and why the bot couldn't
resolve it — so the agent can pick up without making the customer repeat
themselves. This is NOT a flow trace ("was in X flow, last message was Y"); it
is a problem statement for the human.

Escalation happens in two ways in this bot:
  1. The customer explicitly asks for a live agent (need_more_help → Yes).
  2. The bot cannot resolve the request — unknown intent, or a question the
     FAQ/flows couldn't answer (tougher / out-of-scope queries).

Generated from the agent's own conversation history + LLM, so it lives in the
brain. Both the web channel (eventid 1002 context) and the WhatsApp handler
(WxCC Append message) call this one function so the summaries never drift.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SUMMARY_SYS = """\
You are briefing a live customer-service agent who is taking over a chat from a bot.
Write 2-3 short sentences, addressed to the agent, that tell them:
1. What the customer is actually asking for or trying to do (their real need/issue).
2. What specific information the customer already provided (IDs, dates, amounts, etc.).
3. Why it is coming to a human — e.g. the customer explicitly asked for an agent, OR the
   bot could not resolve it (couldn't answer the question / out of scope / unknown request).

Rules:
- Lead with the customer's need, not with chatbot mechanics.
- Do NOT say things like "was in the statement flow" or "last message was". Describe the
  problem, not the bot's internal state.
- Be concrete: pull the actual question/details from the transcript.
- If the customer just asked for a human without a clear issue, say so plainly.
- Plain text only. No markdown, no greetings, no filler."""


def generate_escalation_summary(conversation_id: str, last_message: str) -> str:
    """
    Build a short, agent-facing brief explaining the customer's need and why the
    conversation is being escalated. Summary only — not the full transcript.

    Uses the agent session transcript + the intent LLM. Falls back to a concise
    need-based line if there's no history or the LLM call fails.
    """
    from agent.core.session_store import get_session

    state = get_session(conversation_id)
    history = getattr(state, "history", None) if state else None

    # Build a readable transcript to feed the summariser (not sent to the agent).
    turns = []
    for h in (history or []):
        content = (h.get("content") or "").strip()
        if not content:
            continue
        label = "Customer" if h.get("role") == "user" else "Bot"
        turns.append(f"{label}: {content}")

    # No usable transcript — fall back to the latest customer message.
    if not turns:
        if last_message.strip():
            return (f"Customer needs help with: \"{last_message.strip()}\". "
                    "The bot could not resolve it, so it is being handed to a live agent.")
        return "Customer asked to speak with a live agent. No further detail was provided."

    transcript = "\n".join(turns[-20:])

    try:
        from agent.core.llm import call_intent_llm
        result = call_intent_llm(
            _SUMMARY_SYS,
            [{"role": "user", "content": [{"text": f"Chat transcript:\n{transcript}"}]}],
        )
        summary = (result.get("text") or "").strip()
        if summary:
            return summary
    except Exception as exc:
        logger.warning("[ESCALATION] LLM summary failed: %s", exc)

    # Deterministic fallback — need-based, customer's own words (not flow state).
    customer_msgs = [h.get("content", "").strip()
                     for h in (history or [])
                     if h.get("role") == "user" and h.get("content")]
    asked = customer_msgs[-1] if customer_msgs else (last_message or "")
    if asked:
        return (f"Customer needs help with: \"{asked}\". "
                "The bot could not fully resolve it and is escalating to a live agent.")
    return "Customer requested a live agent. No specific issue was captured from the chat."
