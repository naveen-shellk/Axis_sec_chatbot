# ASL Web Chatbot

**Axis Securities Limited — Axis Direct customer-service chatbot (web channel today, WhatsApp later).**

The backend is a channel-agnostic **agent** (the brain) with thin **channel adapters** in front
of it. A channel adapter only translates its transport payload to/from the agent — all routing,
flow logic, LLM calls, Knowledge Base retrieval, and API/gateway calls live in `agent/`.

---

## Project structure

This branch holds **only the deployable agent** — the channel-agnostic brain plus what
AgentCore needs to run it:

```
chatbot_langchain/
├── agent/                     # THE BRAIN (channel-agnostic; what AgentCore deploys)
│   ├── handler.py             # handle_message() — greeting menu, intent routing, flow dispatch
│   ├── models.py              # shared Pydantic models (request/response/session)
│   ├── entrypoint.py          # AgentCore Runtime entry (process_turn; lazy SDK import)
│   ├── core/                  # llm, langchain_agent, kb_retrieval, faq, conversation, session_store, tools
│   ├── flows/                 # one package per flow (statement, brokerage, order_status, closure, ...)
│   ├── gateways/              # API clients (gateway_client, statement_api, order_api, customer_api)
│   ├── shared/                # session_end, shared helpers
│   └── utils/                 # token_logger
├── config/                    # .env.example (template); real .env is local-only (gitignored)
├── requirements.txt
└── README.md
```

**Channel adapters are not in this branch.** The web adapter (`channels/web/`, FastAPI) and the
local `web_test_ui/` live on disk for local testing and in the `version-1` branch, but they are
gitignored here. Each channel (web now, WhatsApp later) will be deployed as its own microservice
that calls the agent — the brain in `agent/` never changes when a channel is added.

**Why this split:** the agent is a single self-contained package AgentCore runs via
`agent/entrypoint.py` (`process_turn(payload) -> Simcomm dict`). It reads all configuration from
environment variables, so AgentCore just needs the `agent/` package, `requirements.txt`, and the
env vars (template in `config/.env.example`).

---

## Run locally

Local testing uses the web channel adapter, which is **not committed to this branch** (it lives
on disk and in the `version-1` branch). From the `chatbot_langchain/` root:

```bash
pip install -r requirements.txt
cp config/.env.example config/.env     # fill in AWS keys, KB creds, WEB_API_TOKEN, etc.

# 1) Backend (web channel adapter — local only)
python channels/web/app.py             # serves http://localhost:8000

# 2) Frontend test UI (static) — serve the web_test_ui folder on port 5500, e.g.
python -m http.server 5500             # run inside web_test_ui/, then open http://localhost:5500
```

- Health check: http://localhost:8000/health
- Swagger: http://localhost:8000/docs
- The test UI has a password gate (`axis@2026`) and a **Client ID** field (the customer's
  Sub-Account ID). Leave Client ID blank for pre-login flows; set it for account flows.

> Backend code changes need an `app.py` restart (no hot reload). Frontend changes just need a
> browser reload. There is no deployment tooling here — AgentCore deployment is handled separately.

---

## Request / response contract (`POST /api/chat`)

Auth: `Authorization: Bearer <WEB_API_TOKEN>`

Request:
```json
{
  "Conversationid": "CSR0122XNRRMENM4",
  "Message": "What are your support hours?",
  "Event": "Incoming message",
  "Channel": "WEB",
  "timestamp": "2026-08-13T09:15:00.000Z",
  "Client_ID": "6033593"
}
```
- `Conversationid` — session id (which chat).
- `Client_ID` — the customer's **Sub-Account ID** (identity). Optional overall, but **required for
  account flows**; when present the session is treated as logged-in. The backend never asks the
  customer for it in chat.

Response (eventid `1001` normal reply):
```json
{
  "eventid": "1001",
  "conversation_id": "CSR0122XNRRMENM4",
  "message": "...",
  "messagetype": "text",
  "quickReplies": {
    "reference": "<flow_state>",
    "options": [{ "type": "quickReplyPostback", "identifier": "...", "title": "...", "payload": {...} }]
  }
}
```
`eventid` `1002` = escalate to a live agent.

End-chat notification: `POST /api/chat/end` with `{ "conversation_id": "...", "Client_ID": "...", "endedBy": "...", "resolution": "..." }` → `200 {"status":"acknowledged"}`.

---

## Flows

**Pre-login (no Client ID):** Bank Query, How To Trade, Edit Profile, Need More Help, plus FAQ and greeting.
**Post-login (Client ID required):** Statement, Order Status, Brokerage & Charges, Account Details, IPO, Login Query, Closure.

Account flows requested without a Client ID return a brief "please sign in" notice + the pre-login menu (no OTP).

---

## Models (Bedrock)

| Purpose | Model | Env var |
|---------|-------|---------|
| Intent classification / routing + slot extraction | Claude 3 Haiku (`anthropic.claude-3-haiku-20240307-v1:0`) | `INTENT_MODEL_ID` |
| Response / FAQ composition | Qwen (`qwen.qwen3-235b-a22b-2507-v1:0`) | `RESPONSE_MODEL_ID` |
| FAQ retrieval | Bedrock Knowledge Base `JTGNRX1HDB` | `INTENT_KB_ID` |

Model calls use the main `AWS_*` credentials; Knowledge Base retrieval can use separate
`KB_AWS_*` credentials (the KB lives in a different account). Both must allow the needed
Bedrock actions (`bedrock:InvokeModel` for models; KB retrieve for the KB).

---

## Session store

| Backend | Use | Config |
|---------|-----|--------|
| `memory` | local dev (default) | `SESSION_BACKEND=memory` |
| `postgres` | staging / multi-instance | `SESSION_BACKEND=postgres` + `SESSION_DB_URL=...` |
| `agentcore` | production | `SESSION_BACKEND=agentcore` |

Session TTL default: 15 min (`SESSION_TTL_SECONDS`). "New chat" = new `Conversationid` = fresh,
unauthenticated session.

---

## Key environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `WEB_API_TOKEN` | yes | Bearer token for `/api/chat` |
| `AWS_REGION` | yes | e.g. `ap-south-1` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` | yes | Bedrock model invocation |
| `KB_AWS_ACCESS_KEY_ID` / `KB_AWS_SECRET_ACCESS_KEY` / `KB_AWS_SESSION_TOKEN` | KB in a different account | Knowledge Base retrieval |
| `INTENT_KB_ID` | yes (FAQ) | Knowledge Base id (default `JTGNRX1HDB`) |
| `RESPONSE_MODEL_ID` / `INTENT_MODEL_ID` | no | override the Qwen defaults |
| `SESSION_BACKEND` | no | `memory` / `postgres` / `agentcore` |
| `PORT` | no | web server port (default 8000) |

Secrets live in `config/.env` (gitignored). Never commit real credentials — only `config/.env.example`.
