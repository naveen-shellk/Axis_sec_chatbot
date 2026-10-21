"""
agent/shared/feedback.py
------------------------
Persist end-of-chat feedback.

When the customer says "End Chat", the bot asks for feedback and the next
message is stored here as the feedback comment (chat_feedback.comment), after
which the chat ends. Shared by all flows and both channels (web + WhatsApp).

The DB (Aurora chat_feedback table) isn't wired in all environments yet, so
this runs in STUB mode by default: it logs the feedback. Set FEEDBACK_DB_URL
to enable the real INSERT. The call site never changes.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

FEEDBACK_DB_URL = os.getenv("FEEDBACK_DB_URL", "")


def save_feedback(conversation_id: str, comment: str) -> None:
    """
    Store a customer's end-of-chat feedback as chat_feedback.comment.

    STUB (no FEEDBACK_DB_URL): logs the feedback. When the DB URL is set, insert
    into chat_feedback (conversation_id, comment, created_at). The comment should
    already be PII-masked by the guardrail on the model path; this is free-text
    the customer typed, stored as-is.
    """
    comment = (comment or "").strip()
    if not comment:
        return

    if not FEEDBACK_DB_URL:
        logger.info("[FEEDBACK stub] conv=%s comment=%r", conversation_id, comment[:200])
        return

    try:
        import psycopg2
        conn = psycopg2.connect(FEEDBACK_DB_URL)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_feedback (conversation_id, comment) VALUES (%s, %s)",
                (conversation_id, comment),
            )
        conn.close()
        logger.info("[FEEDBACK] stored for conv=%s", conversation_id)
    except Exception as exc:
        logger.error("[FEEDBACK] save failed for conv=%s: %s", conversation_id, exc)
