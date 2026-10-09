"""
chatbot_web/tools_proxy.py
--------------------------
Local HTTP proxy that exposes ALL chatbot tools/APIs as public endpoints, so an
AgentCore Gateway target (in the sandbox gateway) can reach the internal APIs
via an ngrok tunnel:

    AgentCore Gateway  →  https://<ngrok>.ngrok-free.app  →  this proxy  →  internal APIs

Each endpoint simply forwards to the existing gateway_client functions (which
already know how to call the real internal APIs), so behaviour matches the app.

Run:
    cd chatbot_web
    python tools_proxy.py            # serves on :8090
    ngrok http 8090                  # public HTTPS URL

Auth:
    Every request must send header  X-Proxy-Key: <TOOLS_PROXY_KEY>
    (set TOOLS_PROXY_KEY in config/.env). This stops anyone with the ngrok URL
    from hitting internal APIs.

OpenAPI spec for the Gateway target:
    http://localhost:8090/openapi.json   (or the ngrok URL + /openapi.json)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "config", ".env"), override=True)

# CRITICAL: this process IS the tools proxy. It must talk to the internal APIs
# DIRECTLY, never through TOOLS_PROXY_URL — otherwise every request loops back
# out through ngrok to this same proxy (proxy → ngrok → proxy → …) until it
# times out (~45s) and only then falls through to direct. The shared config/.env
# now sets TOOLS_PROXY_URL for the deployed runtime, so we must blank it HERE,
# before gateway_client reads it at import time.
os.environ["TOOLS_PROXY_URL"] = ""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Reuse the existing, working internal-API clients.
from src.gateways.gateway_client import (
    get_customer_profile,
    create_closure_request,
)
from src.gateways.statement_api import (
    request_statement_fireandforget,
    get_ledger,
    send_dp_bill,
)
from src.gateways.order_api import (
    get_todays_orders,
    send_order_history_email,
)

_PROXY_KEY = os.getenv("TOOLS_PROXY_KEY", "change-me-proxy-key")

app = FastAPI(
    title="ASL Chatbot Tools Proxy",
    description="Public HTTP surface for the chatbot's internal APIs, for AgentCore Gateway via ngrok.",
    version="1.0.0",
)


# Auth via middleware so X-Proxy-Key is NOT declared as an OpenAPI parameter.
# (The AgentCore Gateway injects it via its credential provider; declaring it as
# a tool parameter too causes a "conflicts with api key credential provider" error.)
@app.middleware("http")
async def _proxy_auth(request: Request, call_next):
    # Allow unauthenticated access to health + the OpenAPI/docs endpoints.
    open_paths = ("/health", "/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect")
    if request.url.path in open_paths:
        return await call_next(request)
    # TEMP DIAGNOSTIC: set TOOLS_PROXY_OPEN=1 to bypass the key check (to isolate
    # whether the gateway credential injection is the failure). Remove for prod.
    if os.getenv("TOOLS_PROXY_OPEN") == "1":
        if request.headers.get("x-proxy-key") != _PROXY_KEY:
            logger_warn = getattr(app, "_warned", False)
            if not logger_warn:
                print("[tools_proxy] WARNING: TOOLS_PROXY_OPEN=1 — key check bypassed")
                app._warned = True
        return await call_next(request)
    if request.headers.get("x-proxy-key") != _PROXY_KEY:
        return JSONResponse(status_code=401,
                            content={"detail": "Invalid or missing X-Proxy-Key"})
    return await call_next(request)


# ── Request models ────────────────────────────────────────────────────────────

class SubAccountReq(BaseModel):
    sub_account_id: str


class StatementReq(BaseModel):
    sub_account_id: str
    report_name: str
    start_date: str          # DD-MM-YYYY
    end_date: str            # DD-MM-YYYY
    endpoint: str = "exports"
    dp_account_no: str = ""


class LedgerReq(BaseModel):
    sub_account_id: str
    start_date: str
    end_date: str


class DpBillReq(BaseModel):
    sub_account_id: str
    start_date: str
    end_date: str


class OrdersReq(BaseModel):
    sub_account_id: str
    segment: str


class OrderHistoryReq(BaseModel):
    sub_account_id: str
    segment: str
    date_str: str            # DD-MM-YYYY


class ClosureReq(BaseModel):
    sub_account_id: str
    email: str = ""
    name: str = ""
    type_of_account_closure: str = "demat"
    dp_account_no: str = ""


# ── Endpoints (one per tool) ──────────────────────────────────────────────────

@app.get("/health", summary="Liveness probe")
def health():
    return {"status": "ok", "service": "tools-proxy"}


@app.post("/get_customer_profile", summary="Fetch full customer profile",
          operation_id="get_customer_profile")
def ep_get_customer_profile(body: SubAccountReq):
    return get_customer_profile(body.sub_account_id)


@app.post("/request_statement", summary="Submit a statement request (emailed to customer)",
          operation_id="request_statement")
def ep_request_statement(body: StatementReq):
    r = request_statement_fireandforget(
        sub_account_id=body.sub_account_id,
        api_jobname=body.report_name,
        endpoint=body.endpoint,
        start_date=body.start_date,
        end_date=body.end_date,
        dp_account_no=body.dp_account_no,
    )
    return {"success": r.success, "masked_email": r.masked_email,
            "report_id": r.report_id, "error": r.error_message or None}


@app.post("/get_ledger_balance", summary="Fetch ledger balance for a date range",
          operation_id="get_ledger_balance")
def ep_get_ledger(body: LedgerReq):
    return get_ledger(body.sub_account_id, body.start_date, body.end_date)


@app.post("/send_dp_bill", summary="Email the DP charges bill to the customer",
          operation_id="send_dp_bill")
def ep_send_dp_bill(body: DpBillReq):
    return send_dp_bill(body.sub_account_id, body.start_date, body.end_date)


@app.post("/get_todays_orders", summary="Fetch today's executed orders for a segment",
          operation_id="get_todays_orders")
def ep_get_todays_orders(body: OrdersReq):
    return get_todays_orders(body.sub_account_id, body.segment)


@app.post("/send_order_history_email", summary="Email order history for a date",
          operation_id="send_order_history_email")
def ep_send_order_history(body: OrderHistoryReq):
    return send_order_history_email(body.sub_account_id, body.segment, body.date_str)


@app.post("/create_account_closure", summary="Submit an account-closure request",
          operation_id="create_account_closure")
def ep_create_closure(body: ClosureReq):
    return create_closure_request(
        body.sub_account_id, body.email, body.name,
        type_of_account_closure=body.type_of_account_closure,
        dp_account_no=body.dp_account_no,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("TOOLS_PROXY_PORT", "8090"))
    uvicorn.run("tools_proxy:app", host="0.0.0.0", port=port, reload=False)
