"""
chatbot_web/src/core/session_store.py
--------------------------------------
Session store for the web channel — three backends, same public interface.

Backend selection via SESSION_BACKEND env var:
  memory    — in-process dict  (default; dev / single-instance)
  postgres  — PostgreSQL via psycopg2 (staging / multi-instance)
  agentcore — AWS BedrockAgentCore Memory API (production)

AgentCore short-term memory model:
  - One Memory resource per deployment (AGENTCORE_MEMORY_ID)
  - One session per conversation (session_id = conversation_id)
  - One actor per conversation (actor_id = conversation_id, anonymous pre-login)
  - create_event()  → writes a turn (save_session stores the full state JSON)
  - list_events()   → retrieves the latest event (get_session reads state back)
  - deleteMemoryRecord not needed — events expire via eventExpiryDuration on the Memory resource

Two boto3 clients needed:
  bedrock-agentcore-control  → create/get the Memory resource (admin ops)
  bedrock-agentcore          → create_event / list_events (per-turn data ops)

Environment variables needed:
  AGENTCORE_MEMORY_ID   — ID of the pre-created Memory resource (required)
  AWS_REGION            — e.g. ap-south-1
  AWS_ACCESS_KEY_ID     — if running locally (not needed on EC2/Lambda with IAM role)
  AWS_SECRET_ACCESS_KEY — same
  AWS_SESSION_TOKEN     — same (if using STS temporary credentials)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from threading import Lock

from models import SessionState

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS: int      = int(os.getenv("SESSION_TTL_SECONDS", "900"))
SESSION_BACKEND: str         = os.getenv("SESSION_BACKEND", "memory").lower()
SESSION_DB_URL: str      = os.getenv("SESSION_DB_URL", "")
AGENTCORE_MEMORY_ID: str = os.getenv("AGENTCORE_MEMORY_ID", "")
AWS_REGION: str          = os.getenv("AWS_REGION", "ap-south-1")


# =============================================================================
# In-memory backend (default — no dependencies)
# =============================================================================

_store: dict[str, tuple[SessionState, float]] = {}
_lock = Lock()


def _mem_get(conversation_id: str) -> "SessionState | None":
    with _lock:
        entry = _store.get(conversation_id)
        if entry is None:
            return None
        state, saved_at = entry
        if time.time() - saved_at > SESSION_TTL_SECONDS:
            del _store[conversation_id]
            logger.debug("[SESSION:mem] %s expired", conversation_id)
            return None
        return state


def _mem_save(conversation_id: str, state: SessionState) -> None:
    with _lock:
        _store[conversation_id] = (state, time.time())


def _mem_clear(conversation_id: str) -> None:
    with _lock:
        _store.pop(conversation_id, None)


# =============================================================================
# PostgreSQL backend
# =============================================================================

_pg_conn = None
_pg_lock = Lock()


def _get_pg_conn():
    global _pg_conn
    with _pg_lock:
        try:
            import psycopg2
            if _pg_conn is not None:
                try:
                    _pg_conn.isolation_level   # liveness check
                    return _pg_conn
                except Exception:
                    _pg_conn = None
            _pg_conn = psycopg2.connect(SESSION_DB_URL)
            _pg_conn.autocommit = True
            _ensure_pg_tables(_pg_conn)
            logger.info("[SESSION:pg] Connected (%s)", SESSION_DB_URL.split("@")[-1])
            return _pg_conn
        except Exception as exc:
            logger.error("[SESSION:pg] Connection failed: %s", exc)
            raise


def _ensure_pg_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS web_chatbot_sessions (
                conversation_id TEXT PRIMARY KEY,
                state_json      TEXT    NOT NULL,
                updated_at      FLOAT   NOT NULL
            );
        """)


def _pg_get(conversation_id: str) -> SessionState | None:
    try:
        conn = _get_pg_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT state_json, updated_at FROM web_chatbot_sessions WHERE conversation_id = %s",
                (conversation_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        state_json, updated_at = row
        if time.time() - updated_at > SESSION_TTL_SECONDS:
            _pg_clear(conversation_id)
            return None
        return SessionState(**json.loads(state_json))
    except Exception as exc:
        logger.error("[SESSION:pg] get failed: %s — falling back to memory", exc)
        return _mem_get(conversation_id)


def _pg_save(conversation_id: str, state: SessionState) -> None:
    try:
        conn = _get_pg_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO web_chatbot_sessions (conversation_id, state_json, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (conversation_id) DO UPDATE
                  SET state_json = EXCLUDED.state_json,
                      updated_at = EXCLUDED.updated_at
                """,
                (conversation_id, state.model_dump_json(), time.time()),
            )
    except Exception as exc:
        logger.error("[SESSION:pg] save failed: %s — falling back to memory", exc)
        _mem_save(conversation_id, state)


def _pg_clear(conversation_id: str) -> None:
    try:
        conn = _get_pg_conn()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM web_chatbot_sessions WHERE conversation_id = %s",
                (conversation_id,),
            )
    except Exception as exc:
        logger.error("[SESSION:pg] clear failed: %s", exc)


# =============================================================================
# AgentCore Memory backend (short-term memory)
# =============================================================================
#
# Architecture:
#   Memory resource (created once) → identified by AGENTCORE_MEMORY_ID
#     └─ Session (one per conversation_id)
#           └─ Event (one per save_session call — stores full SessionState JSON)
#
# On get_session:
#   list_events(sessionId=conv_id, maxResults=1) → latest event → parse JSON
#
# On save_session:
#   create_event(sessionId=conv_id, payload=[{conversational: {content: {text: json}}}])
#
# On clear_session:
#   We don't delete events (no per-event delete in the API).
#   Instead we write a special "CLEARED" sentinel event.
#   get_session checks for this sentinel and returns None.
#
# TTL: events expire automatically via eventExpiryDuration set on the Memory resource.
#      Set eventExpiryDuration=1 (day) when creating the Memory resource.
#
# Two clients:
#   bedrock-agentcore-control  — admin operations (list_memories, get_memory)
#   bedrock-agentcore          — data operations (create_event, list_events)

_ac_data_client = None
_ac_control_client = None
_ac_lock = Lock()

_AC_SENTINEL = "__CLEARED__"
_AC_ACTOR_ID = "web-chatbot-anonymous"   # all pre-login sessions share this actor


def _get_ac_clients():
    """Return (data_client, control_client), creating them once.
    
    AgentCore Memory lives in the SANDBOX account — use dedicated
    MEMORY_AWS_* credentials if set, otherwise fall back to default env creds.
    """
    global _ac_data_client, _ac_control_client
    with _ac_lock:
        if _ac_data_client is None:
            import boto3

            # Memory resource is in sandbox account — may need separate creds
            # from the Bedrock model credentials (which may be in a different account).
            mem_key    = os.getenv("MEMORY_AWS_ACCESS_KEY_ID")
            mem_secret = os.getenv("MEMORY_AWS_SECRET_ACCESS_KEY")
            mem_token  = os.getenv("MEMORY_AWS_SESSION_TOKEN")

            kwargs = {"region_name": AWS_REGION}
            if mem_key and mem_secret:
                boto_sess = boto3.Session(
                    region_name=AWS_REGION,
                    aws_access_key_id=mem_key,
                    aws_secret_access_key=mem_secret,
                    aws_session_token=mem_token,
                )
                logger.info("[SESSION:agentcore] Using dedicated MEMORY_AWS_* credentials")
            else:
                try:
                    import botocore.session
                    bc_sess = botocore.session.get_session()
                    resolver = bc_sess.get_component("credential_provider")
                    resolver.providers = [p for p in resolver.providers if p.METHOD != "env"]
                    boto_sess = boto3.Session(botocore_session=bc_sess, region_name=AWS_REGION)
                    if boto_sess.get_credentials() is None:
                        boto_sess = boto3.Session(region_name=AWS_REGION)
                except Exception:
                    boto_sess = boto3.Session(region_name=AWS_REGION)

            _ac_data_client    = boto_sess.client("bedrock-agentcore")
            _ac_control_client = boto_sess.client("bedrock-agentcore-control")
            logger.info("[SESSION:agentcore] Clients initialised region=%s memory_id=%s",
                        AWS_REGION, AGENTCORE_MEMORY_ID)
    return _ac_data_client, _ac_control_client


def warmup() -> None:
    """
    Pre-build the AgentCore Memory clients at startup so the first customer
    request doesn't pay the ~1-2s boto3 client-init cost. Safe no-op unless the
    agentcore backend is active. Call once during app/container startup.
    """
    if SESSION_BACKEND != "agentcore":
        return
    try:
        import time as _t
        _t0 = _t.perf_counter()
        _get_ac_clients()
        logger.info("[SESSION:agentcore] warmup complete in %d ms",
                    int((_t.perf_counter() - _t0) * 1000))
    except Exception as exc:
        logger.warning("[SESSION:agentcore] warmup failed (will lazy-init on first request): %s", exc)


def _agentcore_get(conversation_id: str) -> SessionState | None:
    """
    Read session from AgentCore short-term memory.
    Retrieves the most recent event for this session and parses the SessionState JSON.
    Falls back to in-memory on any error.
    """
    try:
        data_client, _ = _get_ac_clients()

        response = data_client.list_events(
            memoryId=AGENTCORE_MEMORY_ID,
            actorId=_AC_ACTOR_ID,
            sessionId=conversation_id,
            maxResults=1,
        )

        events = response.get("events", [])
        if not events:
            logger.debug("[SESSION:agentcore] No events found for %s", conversation_id)
            return None

        # Events are ordered newest first (maxResults=1 gives us the latest)
        latest_event = events[0]
        payload = latest_event.get("payload", [])

        if not payload:
            return None

        # Extract the text content from the first conversational payload item
        content_text = (
            payload[0]
            .get("conversational", {})
            .get("content", {})
            .get("text", "")
        )

        if not content_text or content_text == _AC_SENTINEL:
            logger.debug("[SESSION:agentcore] Session %s cleared or empty", conversation_id)
            return None

        # Parse the SessionState JSON
        state = SessionState(**json.loads(content_text))

        # Check TTL manually (AgentCore expiry is in days, not seconds)
        # We encode the save timestamp inside the state JSON via updated_at
        # Use the event timestamp from AgentCore as the reference
        event_ts = latest_event.get("eventTimestamp")
        if event_ts:
            # Convert to epoch seconds
            if hasattr(event_ts, "timestamp"):
                saved_at = event_ts.timestamp()
            else:
                saved_at = float(event_ts)
            if time.time() - saved_at > SESSION_TTL_SECONDS:
                logger.debug("[SESSION:agentcore] Session %s expired (TTL check)", conversation_id)
                # Write a cleared sentinel so future reads are fast
                _agentcore_clear(conversation_id)
                return None

        logger.debug("[SESSION:agentcore] Loaded session %s flow=%s", conversation_id, state.flow)
        return state

    except Exception as exc:
        logger.error("[SESSION:agentcore] get failed: %s — falling back to memory", exc)
        return _mem_get(conversation_id)


def _agentcore_save(conversation_id: str, state: SessionState) -> None:
    """
    Write session to AgentCore short-term memory as a new event.
    The full SessionState is stored as a JSON string in the conversational content.
    """
    try:
        data_client, _ = _get_ac_clients()

        state_json = state.model_dump_json()

        data_client.create_event(
            memoryId=AGENTCORE_MEMORY_ID,
            actorId=_AC_ACTOR_ID,
            sessionId=conversation_id,
            eventTimestamp=datetime.now(timezone.utc),
            payload=[
                {
                    "conversational": {
                        "content": {"text": state_json},
                        "role": "USER",
                    }
                }
            ],
        )

        logger.debug("[SESSION:agentcore] Saved session %s flow=%s flow_state=%s",
                     conversation_id, state.flow, state.flow_state)

    except Exception as exc:
        logger.error("[SESSION:agentcore] save failed: %s — falling back to memory", exc)
        _mem_save(conversation_id, state)


def _agentcore_clear(conversation_id: str) -> None:
    """
    Mark session as cleared by writing a sentinel event.
    AgentCore does not support per-event deletion — the sentinel approach
    ensures get_session returns None immediately on the next call.
    """
    try:
        data_client, _ = _get_ac_clients()

        data_client.create_event(
            memoryId=AGENTCORE_MEMORY_ID,
            actorId=_AC_ACTOR_ID,
            sessionId=conversation_id,
            eventTimestamp=datetime.now(timezone.utc),
            payload=[
                {
                    "conversational": {
                        "content": {"text": _AC_SENTINEL},
                        "role": "USER",
                    }
                }
            ],
        )

        logger.info("[SESSION:agentcore] Session %s cleared (sentinel written)", conversation_id)

    except Exception as exc:
        logger.error("[SESSION:agentcore] clear failed: %s", exc)
        _mem_clear(conversation_id)


# =============================================================================
# AgentCore Memory resource setup helper
# =============================================================================

def create_agentcore_memory(
    name: str = "asl_web_chatbot_memory",
    region: str | None = None,
    expiry_days: int = 3,
) -> str:
    """
    One-time setup: create the AgentCore Memory resource.
    Returns the memory_id to put in AGENTCORE_MEMORY_ID env var.

    Call this ONCE from a setup script, not at app startup.

    Usage:
        python -c "
        import sys; sys.path.insert(0, 'chatbot_web')
        from src.core.session_store import create_agentcore_memory
        mid = create_agentcore_memory()
        print('Memory ID:', mid)
        print('Add to config/.env: AGENTCORE_MEMORY_ID=' + mid)
        "
    """
    import boto3

    r = region or AWS_REGION
    control = boto3.client("bedrock-agentcore-control", region_name=r)

    response = control.create_memory(
        name=name,
        description="Short-term session memory for ASL Web Chatbot pre-login flows",
        eventExpiryDuration=expiry_days,    # events auto-expire after N days
        # No memoryStrategies = short-term only (no long-term extraction)
        # Add summaryMemoryStrategy here if you want session summaries persisted
    )

    memory_id = response["memory"]["id"]
    logger.info("[SESSION:agentcore] Memory resource created: %s", memory_id)

    # Wait for ACTIVE status
    import time as _time
    for attempt in range(30):
        status_resp = control.get_memory(memoryId=memory_id)
        status = status_resp.get("memory", {}).get("status", "")
        if status == "ACTIVE":
            logger.info("[SESSION:agentcore] Memory is ACTIVE")
            break
        if status == "FAILED":
            raise RuntimeError(f"Memory creation FAILED: {status_resp}")
        logger.info("[SESSION:agentcore] Waiting for ACTIVE... (attempt %d, status=%s)",
                    attempt + 1, status)
        _time.sleep(5)

    return memory_id


# =============================================================================
# Backend selection — runs once at import time
# =============================================================================

_backend = SESSION_BACKEND

if _backend == "postgres":
    if SESSION_DB_URL:
        logger.info("[SESSION_STORE] Backend: PostgreSQL")
    else:
        logger.warning(
            "[SESSION_STORE] SESSION_BACKEND=postgres but SESSION_DB_URL not set "
            "— falling back to in-memory"
        )
        _backend = "memory"

elif _backend == "agentcore":
    if AGENTCORE_MEMORY_ID:
        logger.info("[SESSION_STORE] Backend: AgentCore Memory (memory_id=%s)", AGENTCORE_MEMORY_ID)
    else:
        logger.warning(
            "[SESSION_STORE] SESSION_BACKEND=agentcore but AGENTCORE_MEMORY_ID not set "
            "— falling back to in-memory"
        )
        _backend = "memory"

else:
    logger.info("[SESSION_STORE] Backend: in-memory")


# =============================================================================
# Public API — the only functions flow handlers call
# =============================================================================

def get_session(conversation_id: str) -> SessionState | None:
    """Load session by conversation_id. Returns None if not found or expired."""
    if _backend == "postgres":
        return _pg_get(conversation_id)
    if _backend == "agentcore":
        return _agentcore_get(conversation_id)
    return _mem_get(conversation_id)


def save_session(conversation_id: str, state: SessionState) -> None:
    """Persist session state. Called after every state change."""
    if _backend == "postgres":
        _pg_save(conversation_id, state)
    elif _backend == "agentcore":
        _agentcore_save(conversation_id, state)
    else:
        _mem_save(conversation_id, state)


def clear_session(conversation_id: str) -> None:
    """Delete/invalidate session. Called on end-chat or session timeout."""
    if _backend == "postgres":
        _pg_clear(conversation_id)
    elif _backend == "agentcore":
        _agentcore_clear(conversation_id)
    else:
        _mem_clear(conversation_id)


def get_or_create_session(conversation_id: str) -> SessionState:
    """
    Load existing session or create a fresh one.
    Called at the start of every /api/v1/web/chat turn.
    """
    state = get_session(conversation_id)
    if state is None:
        state = SessionState(conversation_id=conversation_id)
        logger.info("[SESSION] New session: %s", conversation_id)
    return state
