"""
chatbot_web/create_gateway.py
-----------------------------
Create a NEW AgentCore Gateway in the DEPLOYED account (106611079163, the
MEMORY_AWS_* credentials) and register the ngrok tools proxy as an OpenAPI
target on it.

Why: the existing gateway (pgywge4cf0) lives in the client/model account
(625867133907). The deployed runtime runs in 106611079163, so its gateway +
tool target must be created there, using the MEMORY_AWS_* credentials.

Prereqs:
  1. python tools_proxy.py          (serves tools on :8090)
  2. ngrok http 8090                (public HTTPS URL)
  3. config/.env has MEMORY_AWS_* (deployed acct), TOOLS_PROXY_KEY,
     and GATEWAY_ROLE_ARN (an IAM role in 106611079163 the gateway assumes)

Usage:
  python create_gateway.py https://<id>.ngrok-free.dev

Output: the new gateway id + MCP URL to set as AGENTCORE_GATEWAY_URL in the
runtime env (via _create_runtime.py) on the next deploy.
"""
from __future__ import annotations

import os
import sys
import json
import time
import urllib.request

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "config", ".env"), override=True)

import boto3

REGION        = os.getenv("AWS_REGION", "ap-south-1")
PROXY_KEY     = os.getenv("TOOLS_PROXY_KEY", "change-me-proxy-key")
GATEWAY_NAME  = os.getenv("NEW_GATEWAY_NAME", "asl-web-chatbot-gateway")
TARGET_NAME   = os.getenv("TOOLS_TARGET_NAME", "chatbot-tools-ngrok")
# IAM role (in the deployed account) the gateway uses. Defaults to the runtime
# execution role created by deploy_agentcore.ps1.
ROLE_ARN      = os.getenv(
    "GATEWAY_ROLE_ARN",
    "arn:aws:iam::106611079163:role/AgentCoreRuntimeExecutionRole",
)


def _deployed_client(service: str):
    """boto3 client using the DEPLOYED account (MEMORY_AWS_*) credentials."""
    k = os.getenv("MEMORY_AWS_ACCESS_KEY_ID")
    s = os.getenv("MEMORY_AWS_SECRET_ACCESS_KEY")
    t = os.getenv("MEMORY_AWS_SESSION_TOKEN")
    if not (k and s):
        print("ERROR: MEMORY_AWS_* credentials not set in config/.env")
        sys.exit(1)
    return boto3.client(service, region_name=REGION,
                        aws_access_key_id=k, aws_secret_access_key=s,
                        aws_session_token=t)


def _fetch_openapi(ngrok_url: str) -> dict:
    url = ngrok_url.rstrip("/") + "/openapi.json"
    print(f"Fetching OpenAPI spec from {url} ...")
    req = urllib.request.Request(url, headers={"ngrok-skip-browser-warning": "1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        spec = json.loads(resp.read().decode("utf-8"))
    spec["servers"] = [{"url": ngrok_url.rstrip("/")}]
    print(f"  OK — {len(spec.get('paths', {}))} paths")
    return spec


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_gateway.py https://<id>.ngrok-free.dev")
        sys.exit(1)
    ngrok_url = sys.argv[1].rstrip("/")

    spec   = _fetch_openapi(ngrok_url)
    client = _deployed_client("bedrock-agentcore-control")

    # Confirm which account we're operating in.
    ident = _deployed_client("sts").get_caller_identity()
    print(f"Operating in account: {ident['Account']}  (should be 106611079163)")

    # ── 1. Create the gateway (MCP, IAM inbound auth) ─────────────────────────
    print(f"\nCreating gateway '{GATEWAY_NAME}' ...")
    try:
        gw = client.create_gateway(
            name=GATEWAY_NAME,
            description="ASL web chatbot — MCP tool gateway (deployed account)",
            roleArn=ROLE_ARN,
            protocolType="MCP",
            protocolConfiguration={
                "mcp": {
                    "instructions": "Gateway for the Axis Direct web chatbot tools.",
                    "searchType": "SEMANTIC",
                    "supportedVersions": ["2025-03-26"],
                }
            },
            authorizerType="AWS_IAM",
        )
        gateway_id  = gw["gatewayId"]
        gateway_url = gw.get("gatewayUrl", "")
        print(f"  gatewayId  = {gateway_id}")
        print(f"  gatewayUrl = {gateway_url}")
    except client.exceptions.ConflictException:
        gws = client.list_gateways().get("items", [])
        match = next((g for g in gws if g.get("name") == GATEWAY_NAME), None)
        if not match:
            raise
        gateway_id  = match["gatewayId"]
        gateway_url = match.get("gatewayUrl", "")
        print(f"  reusing existing gatewayId = {gateway_id}")

    # Wait until the gateway is READY before adding a target.
    print("  waiting for gateway to be READY ...")
    for _ in range(30):
        st = client.get_gateway(gatewayIdentifier=gateway_id).get("status")
        if st == "READY":
            break
        time.sleep(2)
    print(f"  status = {st}")

    # ── 2. API-key credential provider (X-Proxy-Key) ──────────────────────────
    print("\nCreating API-key credential provider ...")
    cred_name = f"{TARGET_NAME}-key"

    def _find_existing_cred() -> str:
        provs = client.list_api_key_credential_providers()
        items = provs.get("credentialProviders") or provs.get("items") or []
        for p in items:
            if p.get("name") == cred_name:
                return p.get("credentialProviderArn") or p.get("arn")
        raise RuntimeError(f"Credential provider {cred_name} exists but was not found in list")

    try:
        cred = client.create_api_key_credential_provider(name=cred_name, apiKey=PROXY_KEY)
        cred_arn = cred["credentialProviderArn"]
    except Exception as exc:
        # Already exists (ConflictException OR ValidationException "already exists")
        if "already exists" in str(exc) or "Conflict" in type(exc).__name__:
            cred_arn = _find_existing_cred()
            print("  reusing existing credential provider")
        else:
            raise
    print(f"  credentialProviderArn = {cred_arn}")

    # ── 3. OpenAPI target (create, or UPDATE if it already exists) ────────────
    target_config = {
        "mcp": {"openApiSchema": {"inlinePayload": json.dumps(spec)}}
    }
    cred_cfg = [
        {
            "credentialProviderType": "API_KEY",
            "credentialProvider": {
                "apiKeyCredentialProvider": {
                    "providerArn": cred_arn,
                    "credentialLocation": "HEADER",
                    "credentialParameterName": "X-Proxy-Key",
                }
            },
        }
    ]

    print("\nCreating gateway target ...")
    try:
        resp = client.create_gateway_target(
            gatewayIdentifier=gateway_id,
            name=TARGET_NAME,
            description="ngrok-tunnelled local chatbot tools proxy",
            targetConfiguration=target_config,
            credentialProviderConfigurations=cred_cfg,
        )
        target_id = resp.get("targetId")
    except client.exceptions.ConflictException:
        # Target already exists — find it and UPDATE with the fresh spec.
        print("  target exists — updating with the latest OpenAPI spec ...")
        items = client.list_gateway_targets(gatewayIdentifier=gateway_id).get("items", [])
        target_id = next(t["targetId"] for t in items if t["name"] == TARGET_NAME)
        resp = client.update_gateway_target(
            gatewayIdentifier=gateway_id,
            targetId=target_id,
            name=TARGET_NAME,
            description="ngrok-tunnelled local chatbot tools proxy",
            targetConfiguration=target_config,
            credentialProviderConfigurations=cred_cfg,
        )

    print(f"\nSUCCESS")
    print(f"  gatewayId = {gateway_id}")
    print(f"  targetId  = {target_id}  status={resp.get('status')}")
    print(f"\nSet this as AGENTCORE_GATEWAY_URL in the runtime env (next deploy):")
    print(f"  {gateway_url or '(fetch via get_gateway)'}")


if __name__ == "__main__":
    main()
