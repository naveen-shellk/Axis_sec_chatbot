"""
chatbot_langchain/src/core/kb_retrieval.py
-------------------------------------------
Bedrock Knowledge Base retrieval for the web chatbot.

The KB (default id JTGNRX1HDB) lives in a DIFFERENT AWS dev account than the
Bedrock model / runtime, so it needs its OWN credentials — separate from the
AWS_* keys in config/.env that the LLM uses.

Credential strategy (first match wins):
  1. KB_AWS_ACCESS_KEY_ID / KB_AWS_SECRET_ACCESS_KEY / KB_AWS_SESSION_TOKEN
     → explicit static creds for the KB account (set these in config/.env).
  2. KB_AWS_PROFILE
     → a named profile in ~/.aws/credentials for the KB account.
  3. Default credential chain (SSO / env / IMDS) as a last resort.

Mirrors thor's agent/services/kb_retrieval.py retrieve() contract.

Usage:
    from src.core.kb_retrieval import retrieve_kb_chunks
    chunks = retrieve_kb_chunks("customer wants to close demat account")
    for c in chunks:
        print(c["score"], c["text"][:80])
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Config (read by NAME; values live in env, never hardcoded secrets) ─────────
KB_ID        = os.getenv("INTENT_KB_ID", "JTGNRX1HDB")
KB_REGION    = os.getenv("KB_AWS_REGION", os.getenv("AWS_REGION", "ap-south-1"))
KB_NUM_RESULTS = int(os.getenv("INTENT_KB_NUM_RESULTS", "5"))
KB_SEARCH_TYPE = os.getenv("KB_SEARCH_TYPE", "HYBRID")  # HYBRID | SEMANTIC

# Built-in Bedrock KB metadata key that holds each chunk's S3 source URI.
# Used to restrict retrieval to a single document in a multi-doc data source.
_SOURCE_URI_KEY = "x-amz-bedrock-kb-source-uri"

_kb_client: Any = None


def _get_kb_client():
    """
    Build (once) a bedrock-agent-runtime client for the KB account, using the
    KB-specific credentials so it does NOT reuse the LLM/Bedrock AWS_* creds.
    """
    global _kb_client
    if _kb_client is not None:
        return _kb_client

    import boto3

    key     = os.getenv("KB_AWS_ACCESS_KEY_ID")
    secret  = os.getenv("KB_AWS_SECRET_ACCESS_KEY")
    token   = os.getenv("KB_AWS_SESSION_TOKEN")
    profile = os.getenv("KB_AWS_PROFILE")

    if key and secret:
        logger.info("[KB] using static KB_AWS_* credentials")
        session = boto3.Session(
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            aws_session_token=token,
            region_name=KB_REGION,
        )
    elif profile:
        logger.info("[KB] using KB_AWS_PROFILE=%s", profile)
        session = boto3.Session(profile_name=profile, region_name=KB_REGION)
    else:
        logger.info("[KB] no KB-specific creds — using default credential chain")
        session = boto3.Session(region_name=KB_REGION)

    _kb_client = session.client("bedrock-agent-runtime", region_name=KB_REGION)
    return _kb_client


def retrieve_kb_chunks(
    query: str,
    num_results: int | None = None,
    knowledge_base_id: str | None = None,
    search_type: str | None = None,
    source_uri: str | None = None,
) -> list[dict[str, Any]]:
    """
    Query the Bedrock Knowledge Base and return the top-K chunks.

    Args:
      source_uri: if given, restrict results to the chunks whose S3 source URI
        EQUALS this value — i.e. only this one document in a multi-doc data
        source (e.g. the FAQ xlsx). Uses the built-in source-URI metadata key.

    Returns a list of dicts: {"text", "score", "source_uri", "metadata"}
    sorted by score (descending). Returns [] on any error (never raises) so a
    KB outage degrades gracefully — the caller can fall back to its normal
    (non-KB) behaviour.
    """
    kb_id = knowledge_base_id or KB_ID
    n     = num_results if num_results is not None else KB_NUM_RESULTS
    stype = search_type or KB_SEARCH_TYPE

    if not kb_id:
        logger.warning("[KB] no knowledge base id configured — skipping retrieve")
        return []
    if not (query or "").strip():
        return []

    vector_cfg: dict[str, Any] = {
        "numberOfResults": n,
        "overrideSearchType": stype,
    }
    # Restrict to a single source document when requested (e.g. FAQ only).
    if source_uri:
        vector_cfg["filter"] = {
            "equals": {"key": _SOURCE_URI_KEY, "value": source_uri}
        }

    try:
        client = _get_kb_client()
        resp = client.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": vector_cfg},
        )
    except Exception as exc:
        logger.error("[KB] retrieve failed kb_id=%s: %s", kb_id, exc)
        return []

    chunks: list[dict[str, Any]] = []
    for r in resp.get("retrievalResults", []):
        content = r.get("content", {}) or {}
        loc = r.get("location", {}) or {}
        s3 = (loc.get("s3Location") or {}) if isinstance(loc, dict) else {}
        chunks.append({
            "text":       content.get("text", ""),
            "score":      float(r.get("score", 0.0) or 0.0),
            "source_uri": s3.get("uri", ""),
            "metadata":   r.get("metadata", {}) or {},
        })

    chunks.sort(key=lambda c: c["score"], reverse=True)
    logger.info("[KB] retrieved %d chunk(s) from KB=%s", len(chunks), kb_id)
    return chunks


def warmup() -> None:
    """Pre-warm the KB path at startup so the first customer FAQ turn doesn't
    pay the one-time cold-start cost (observed ~20s: boto3 client init + DNS +
    TLS handshake + first-call credential/signing to the KB endpoint).

    Building the client alone is NOT enough — the expensive part is the FIRST
    real network call. So we fire ONE throwaway retrieve to fully establish the
    connection. Safe no-op on failure (lazy-init still works on first request)."""
    try:
        import time as _t
        _t0 = _t.perf_counter()
        _get_kb_client()                       # build the boto3 client
        # Real (tiny) retrieve → forces DNS + TLS + signing now, not on turn 1.
        retrieve_kb_chunks("warmup", num_results=1)
        logger.info("[KB] warmup complete in %d ms", int((_t.perf_counter() - _t0) * 1000))
    except Exception as exc:
        logger.warning("[KB] warmup failed (lazy-init on first request): %s", exc)


def kb_healthcheck() -> dict[str, Any]:
    """
    Quick connectivity probe: runs a trivial retrieve and reports whether the
    KB is reachable with the configured credentials. For local testing.
    """
    try:
        chunks = retrieve_kb_chunks("test connectivity", num_results=1)
        return {"ok": True, "kb_id": KB_ID, "region": KB_REGION,
                "chunks_returned": len(chunks)}
    except Exception as exc:  # retrieve_kb_chunks already swallows, but be safe
        return {"ok": False, "kb_id": KB_ID, "region": KB_REGION, "error": str(exc)}
