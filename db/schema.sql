-- =====================================================================
-- chatbot_langchain — Aurora Serverless (PostgreSQL) session schema
-- ---------------------------------------------------------------------
-- Scope: this AgentCore repo stores ONLY its own conversation session.
-- The agent is request -> response (eventid 1001/1002); it does NOT do
-- channel redirecting, WxCC handover, OTP, or token management -- those
-- live in SEPARATE microservices / repos. Hence a SINGLE table here.
--
-- This file is schema only (DDL). No application code reads/writes it yet.
-- =====================================================================

CREATE TABLE conversation_session (
    -- Identity of the conversation
    conversation_id    TEXT PRIMARY KEY,           -- echoes Webex "Conversationid"; returned as conversation_id
    channel            TEXT NOT NULL,              -- 'WEB' | 'WHATSAPP' (from request "Channel"); recorded, not routed on

    -- Customer identity captured during the conversation (for the response `customer` object).
    -- Masked values only. Once known, echoed in every subsequent response.
    is_login           BOOLEAN NOT NULL DEFAULT FALSE,
    sub_account_id     TEXT,
    customer_email     TEXT,                       -- masked
    customer_phone     TEXT,                       -- masked (with country code)

    -- The agent's own state (persisted SessionState)
    flow               TEXT,                       -- active flow, e.g. 'statement'
    flow_state         TEXT,                       -- step within the flow, e.g. 'date_range_30'
    collected_data     JSONB NOT NULL DEFAULT '{}',-- pending_intents, picked values, etc.
    history            JSONB NOT NULL DEFAULT '[]',-- recent conversation turns

    -- Escalation SIGNAL only. The agent records that it returned eventid 1002;
    -- the actual WxCC handover is performed by a separate service, not here.
    is_escalated       BOOLEAN NOT NULL DEFAULT FALSE,
    escalation_reason  TEXT,                       -- e.g. 'user_requested_agent'

    -- Message timestamps (sourced from the request "timestamp" field, ISO 8601 UTC)
    first_msg_at       TIMESTAMPTZ,                -- set ONCE on the first message; never updated after
    last_msg_at        TIMESTAMPTZ,                -- updated on EVERY message

    -- Lifecycle / row bookkeeping (server-side clock)
    status             TEXT NOT NULL DEFAULT 'active'
                       CHECK (status IN ('active','escalated','ended','expired')),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),  -- when the row was first written
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),  -- when the row was last written
    expires_at         TIMESTAMPTZ                           -- last_msg_at + session TTL (e.g. 900s)
);

CREATE INDEX idx_cs_subacct   ON conversation_session (sub_account_id);
CREATE INDEX idx_cs_status    ON conversation_session (status);
CREATE INDEX idx_cs_expires   ON conversation_session (expires_at);
CREATE INDEX idx_cs_channel   ON conversation_session (channel, status);
CREATE INDEX idx_cs_first_msg ON conversation_session (first_msg_at);

-- =====================================================================
-- Column notes
-- ---------------------------------------------------------------------
-- first_msg_at vs created_at:
--   first_msg_at / last_msg_at come from the REQUEST timestamp (customer /
--   Webex clock) -> use for conversation duration, idle detection, analytics.
--   created_at / updated_at are server now() at DB-write time -> row auditing.
--
-- Write pattern (for whoever implements the backend later):
--   first_msg_at = COALESCE(first_msg_at, :request_ts)   -- fill once
--   last_msg_at  = :request_ts                           -- every turn
--   updated_at   = now()
--   expires_at   = :request_ts + INTERVAL '900 seconds'  -- or app-chosen TTL
--
-- Ownership: this repo (the agent) writes every column here. Channel-specific
-- plumbing (whatsapp task maps, WxCC OAuth tokens, OTP audit) is intentionally
-- NOT in this schema -- it belongs to the separate channel/handler services.
-- =====================================================================
