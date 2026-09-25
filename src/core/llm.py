"""
chatbot_web/src/core/llm.py
----------------------------
Two-model LLM interface for the web channel:

  HAIKU  → intent classification only (fast, cheap, bounded JSON output)
  QWEN   → all conversational response generation (richer, more natural)

Environment variables:
  INTENT_MODEL_ID   — model for intent classification  (default: claude-3-haiku)
  RESPONSE_MODEL_ID — model for flow responses         (default: qwen3-235b)
  AWS_REGION        — e.g. ap-south-1

Usage:
  from src.core.llm import call_llm, call_intent_llm

  # Intent classification → always Haiku
  result = call_intent_llm(system, messages)

  # Flow response generation → always Qwen
  result = call_llm(system, messages, expect_json=True)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import boto3

logger = logging.getLogger(__name__)

# ── Model IDs ─────────────────────────────────────────────────────────────────
_INTENT_MODEL_ID   = os.getenv("INTENT_MODEL_ID",   "anthropic.claude-3-haiku-20240307-v1:0")
_RESPONSE_MODEL_ID = os.getenv("RESPONSE_MODEL_ID",  "qwen.qwen3-235b-a22b-2507-v1:0")
_REGION            = os.getenv("AWS_REGION",          "ap-south-1")

# Intent classification — strict settings: no creativity, deterministic output
_INTENT_MAX_TOKENS  = int(os.getenv("INTENT_MAX_TOKENS",   "256"))   # only needs short JSON
_INTENT_TEMPERATURE = float(os.getenv("INTENT_TEMPERATURE", "0.0"))  # deterministic

# Response generation — allow some creativity
_MAX_TOKENS        = int(os.getenv("CHATBOT_MAX_TOKENS",  "1024"))
_TEMPERATURE       = float(os.getenv("CHATBOT_TEMPERATURE", "0.2"))

_client: Any = None


def _get_client() -> Any:
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=_REGION)
    return _client


def _invoke(model_id: str, system: str, messages: list[dict[str, Any]],
            expect_json: bool) -> dict[str, Any]:
    """Shared Bedrock Converse call used by both model wrappers."""
    client = _get_client()

    response = client.converse(
        modelId=model_id,
        system=[{"text": system}],
        messages=messages,
        inferenceConfig={
            "maxTokens":   _MAX_TOKENS,
            "temperature": _TEMPERATURE,
        },
    )

    raw_text: str = response["output"]["message"]["content"][0]["text"].strip()
    usage = response.get("usage", {})

    result: dict[str, Any] = {
        "text":          raw_text,
        "parsed":        None,
        "input_tokens":  usage.get("inputTokens",  0),
        "output_tokens": usage.get("outputTokens", 0),
        "model_id":      model_id,
    }

    if expect_json:
        clean = raw_text.strip()
        if "```" in clean:
            import re
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
            if m:
                clean = m.group(1).strip()
            elif clean.startswith("```"):
                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            result["parsed"] = json.loads(clean, strict=False)
        except (json.JSONDecodeError, ValueError):
            import re
            m = re.search(r"\{[\s\S]*\}", clean)
            if m:
                try:
                    result["parsed"] = json.loads(m.group(0).strip(), strict=False)
                except Exception:
                    pass

            if result["parsed"] is None:
                # Regex fallback to extract message if JSON string is unescaped or unclosed
                m_msg = re.search(r'"message"\s*:\s*"([\s\S]*?)(?:"\s*,\s*"|\s*"\s*\}|$)', clean)
                if m_msg:
                    raw_val = m_msg.group(1).replace('\\"', '"').replace('\\n', '\n')
                    result["parsed"] = {
                        "message": raw_val.strip(),
                        "quick_replies": ["Go back to main menu", "End Chat"],
                        "flow_action": "session_end",
                    }

        if result["parsed"] is None:
            logger.error("[LLM] JSON parse failed (%s) | raw: %r",
                         model_id, raw_text[:200])

    return result


# ── Public functions ──────────────────────────────────────────────────────────

def call_intent_llm(
    system: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Call Haiku for INTENT CLASSIFICATION only.
    Returns a parsed JSON dict with {"intent", "confidence", "reasoning"}.
    Always fast, always cheap — Haiku handles bounded classification well.
    """
    logger.debug("[LLM:intent] using %s", _INTENT_MODEL_ID)
    return _invoke(_INTENT_MODEL_ID, system, messages, expect_json=True)


def call_llm(
    system: str,
    messages: list[dict[str, Any]],
    expect_json: bool = False,
) -> dict[str, Any]:
    """
    Call Qwen for RESPONSE GENERATION — all flow conversation turns.
    Returns:
        {
          "text":          raw model output,
          "parsed":        parsed JSON dict (if expect_json=True),
          "input_tokens":  int,
          "output_tokens": int,
          "model_id":      str,
        }
    """
    logger.debug("[LLM:response] using %s", _RESPONSE_MODEL_ID)
    return _invoke(_RESPONSE_MODEL_ID, system, messages, expect_json)


def call_llm_stream(
    system: str,
    messages: list[dict[str, Any]],
):
    """
    Stream RESPONSE GENERATION token-by-token using Bedrock converse_stream.

    Yields plain text deltas as they arrive from the model:
        for delta in call_llm_stream(system, messages):
            print(delta, end="")

    After the generator is exhausted, the accumulated usage is available via
    the StopIteration value (return). To capture it, use:
        gen = call_llm_stream(...)
        text = ""
        try:
            while True:
                text += next(gen)
        except StopIteration as e:
            usage = e.value   # {"text", "input_tokens", "output_tokens", "model_id"}

    Use this ONLY for plain-text (non-JSON) responses — greeting, how-to-trade
    guidance, closure messages. Quick replies are set by the flow handler.
    """
    client = _get_client()

    logger.debug("[LLM:stream] using %s", _RESPONSE_MODEL_ID)
    response = client.converse_stream(
        modelId=_RESPONSE_MODEL_ID,
        system=[{"text": system}],
        messages=messages,
        inferenceConfig={
            "maxTokens":   _MAX_TOKENS,
            "temperature": _TEMPERATURE,
        },
    )

    full_text = ""
    in_tok = 0
    out_tok = 0

    for event in response["stream"]:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"].get("text", "")
            if delta:
                full_text += delta
                yield delta
        elif "metadata" in event:
            usage = event["metadata"].get("usage", {})
            in_tok = usage.get("inputTokens", 0)
            out_tok = usage.get("outputTokens", 0)

    return {
        "text":          full_text,
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
        "model_id":      _RESPONSE_MODEL_ID,
    }
