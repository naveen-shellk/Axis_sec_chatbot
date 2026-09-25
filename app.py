"""
chatbot_web/app.py
------------------
FastAPI application — local development server.

Run:
    cd chatbot_web
    python app.py
    # or
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Then open:
    http://localhost:8000/docs      ← Swagger UI
    http://localhost:8000/redoc     ← ReDoc
    http://localhost:8000/health    ← Health check

Single public endpoint: POST /api/chat  (Simcomm/WebX integration)
Debug endpoints:        /internal/*     (dev only — no auth)
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "config", ".env"), override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.web_routes import router as web_router
from src.api.internal_routes import router as internal_router
from src.api.oauth_routes import router as oauth_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="ASL Web Chatbot — Axis Direct",
    description="""
## Axis Direct Web Channel Chatbot API

Backend for the Axis Direct web chatbot. Integrates with **Cisco Simcomm/WebX** via a single endpoint.

---

## Single Endpoint

```
POST /api/chat
Authorization: Bearer <customer-issued token>
```

### Request (Simcomm → Our Backend)
```json
{
  "Conversationid": "CSR0122XNRRMENM4",
  "Message": "What are your support hours?",
  "Event": "Incoming message",
  "Channel": "WEB",
  "timestamp": "2026-08-13T09:15:00.000Z"
}
```

### Response — Normal reply (eventid 1001)
```json
{
  "eventid": "1001",
  "conversation_id": "CSR0122XNRRMENM4",
  "message": "Welcome to Axis Direct! How can I help you?",
  "messagetype": "text",
  "timestamp": "2026-08-13T09:15:01.200Z",
  "customer": {"email": "", "phone": ""},
  "quickReplies": {
    "reference": "main_menu",
    "options": [
      {"type": "quickReplyPostback", "identifier": "bank_query", "title": "Bank Query", "imageUrl": "", "payload": {"payload": {"action": "bank_query"}}}
    ]
  }
}
```

### Response — Escalation (eventid 1002)
```json
{
  "eventid": "1002",
  "conversation_id": "CSR0122XNRRMENM4",
  "timestamp": "2026-08-13T09:15:02.000Z",
  "customparam1": "escalation_reason:need_more_help",
  "customer": {"email": "", "phone": ""},
  "context": {"summary": "Customer requested live agent", "customerIntent": "need_more_help"}
}
```

---

## Flows

| Flow | Auth | Description |
|------|------|-------------|
| Bank Query | No | Redirect to Axis Bank contact info |
| How To Trade | No | Step-by-step trading instructions |
| Need More Help | No | Escalate → eventid 1002 |
| Edit Profile | No | Show portal deeplink |
| Statement | Yes | Generate and email statements |
| IPO | Yes | IPO application deeplink |
| Account Details | Yes | Show account info |
| Brokerage & Charges | Yes | Show charges / send DP bill |
| Login Query | Yes | FTL / troubleshooting / deactivation |
| Order Status | Yes | Today's orders / order history |

---

## Local Testing

Use the `/internal/message` endpoint (no auth needed) with any conversation_id.
Pass `sub_account_id` to test post-login flows without real authentication.
""",
    version="1.0.0",
    contact={"name": "ASL Platform Team", "email": "platform@axisdirect.in"},
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Error handlers ────────────────────────────────────────────────────────────
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logging.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(oauth_router)        # POST /oauth/token (client_credentials)
app.include_router(web_router)          # POST /api/chat  (public — Simcomm)
app.include_router(internal_router)     # /internal/*     (dev only)


# ── Startup: warm the AgentCore Memory clients ────────────────────────────────
# Pre-builds the boto3 clients so the first customer request doesn't pay the
# ~1-2s client-init cost (moves it into server startup instead).
@app.on_event("startup")
def _warmup_memory():
    try:
        from src.core.session_store import warmup as _mem_warmup
        _mem_warmup()
    except Exception as exc:
        logging.warning("memory warmup skipped: %s", exc)
    try:
        from src.core.strands_agent import warmup as _agent_warmup
        _agent_warmup()
    except Exception as exc:
        logging.warning("agent warmup skipped: %s", exc)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
