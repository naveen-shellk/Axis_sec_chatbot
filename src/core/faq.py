"""
chatbot_langchain/src/core/faq.py
----------------------------------
KB-backed FAQ answering (retrieve + LLM compose).

Pipeline:
  1. retrieve_kb_chunks(query)  → top-K FAQ chunks from the Bedrock KB
     (uses the SEPARATE KB credentials in kb_retrieval.py, not the LLM creds).
  2. call_llm(...)             → compose a grounded answer using ONLY those
     chunks. If the chunks don't actually answer the question, the model is
     told to say so, which we map to found=False so the caller can fall back.

Returns a structured result the entry handler uses to decide whether to answer
from the KB or fall back to the main menu.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from src.core.kb_retrieval import retrieve_kb_chunks
# FAQ compose uses Qwen (call_llm) for higher-quality, natural answers.
from src.core.llm import call_llm

logger = logging.getLogger(__name__)

# Minimum top-chunk score to even attempt an answer. Below this the retrieval is
# too weak to be a real FAQ match → skip the LLM, let the caller show the menu.
_MIN_SCORE = float(os.getenv("FAQ_MIN_SCORE", "0.4"))
_NUM_CHUNKS = int(os.getenv("FAQ_NUM_CHUNKS", "3"))   # top-3 is enough for FAQ answering

# The KB data source holds MULTIPLE documents; restrict FAQ retrieval to ONLY
# the FAQ workbook so the other document(s) never leak into FAQ answers.
_FAQ_SOURCE_URI = os.getenv(
    "FAQ_SOURCE_URI",
    "s3://asl-aws-dev-orion-bot-kb-bucket/FAQs_260926.xlsx",
)

_FAQ_SYSTEM = """\
You are the Axis Direct assistant answering a customer's general/FAQ question.
You are given FAQ reference snippets retrieved from the official knowledge base.

STRICT rules:
- Answer ONLY using the facts in the provided snippets. Do NOT use outside
  knowledge and do NOT invent details, numbers, URLs, or steps.
- If the snippets do NOT contain enough information to answer the question,
  set "answered" to false and leave "answer" empty.
- Keep the answer concise, warm, and professional. Plain text, no markdown headers.
- Do not mention "snippets", "knowledge base", or "context" in the answer.
- Answer ONLY the question asked. Do NOT add a follow-up question, a call to
  action, or an upsell. NEVER append things like "Would you like to proceed
  with guidance on a trade?" or "Would you like help with a specific type of
  trade?". End right after the factual answer.

Return ONLY valid JSON:
{
  "answered": <true|false>,
  "answer":   "<the answer text, or empty string if answered is false>"
}
"""


def answer_faq(query: str) -> dict[str, Any]:
    """
    Try to answer a general/FAQ question from the knowledge base.

    Returns:
      {
        "found":     bool,   # True only when we have a confident, grounded answer
        "answer":    str,    # the composed answer (empty when found is False)
        "chunks":    int,    # how many chunks were retrieved
        "score":     float,  # top chunk score
        "retrieved": list,   # [{score, source_uri, text}] for the debug sidebar
      }
    Never raises — any failure (KB down, LLM error) returns found=False so the
    caller falls back to its normal behaviour (the menu).
    """
    q = (query or "").strip()
    if not q:
        return {"found": False, "answer": "", "chunks": 0, "score": 0.0,
                "retrieved": []}

    # 1. Retrieve — restricted to the FAQ document only.
    chunks = retrieve_kb_chunks(q, num_results=_NUM_CHUNKS,
                                source_uri=_FAQ_SOURCE_URI or None)
    top_score = chunks[0]["score"] if chunks else 0.0

    # Compact chunk view for the debug sidebar (what retrieval actually returned).
    retrieved = [
        {
            "score":      round(float(c.get("score", 0.0) or 0.0), 4),
            "source_uri": c.get("source_uri", ""),
            "text":       c.get("text", ""),
        }
        for c in chunks
    ]

    if not chunks or top_score < _MIN_SCORE:
        logger.info("[FAQ] no confident KB match (chunks=%d top_score=%.3f) — skip",
                    len(chunks), top_score)
        return {"found": False, "answer": "", "chunks": len(chunks),
                "score": top_score, "retrieved": retrieved}

    # 2. LLM compose (grounded on the retrieved snippets only)
    context = "\n\n".join(
        f"[snippet {i+1}]\n{c['text']}" for i, c in enumerate(chunks) if c.get("text")
    )
    user_text = f"FAQ reference snippets:\n{context}\n\nCustomer question: {q}"
    try:
        result = call_llm(   # Qwen — richer, more natural FAQ answers
            _FAQ_SYSTEM,
            [{"role": "user", "content": [{"text": user_text}]}],
            expect_json=True,
        )
        parsed = result.get("parsed") or {}
    except Exception as exc:
        logger.error("[FAQ] LLM compose failed: %s", exc)
        return {"found": False, "answer": "", "chunks": len(chunks),
                "score": top_score, "retrieved": retrieved}

    answered = bool(parsed.get("answered"))
    answer = (parsed.get("answer") or "").strip()
    if not answered or not answer:
        logger.info("[FAQ] LLM could not answer from KB — fall back")
        return {"found": False, "answer": "", "chunks": len(chunks),
                "score": top_score, "retrieved": retrieved}

    logger.info("[FAQ] answered from KB (chunks=%d top_score=%.3f)", len(chunks), top_score)
    return {"found": True, "answer": answer, "chunks": len(chunks),
            "score": top_score, "retrieved": retrieved}
