"""
chatbot_web/create_gateway_target.py
-------------------------------------
Register the ngrok-tunnelled tools proxy as an OpenAPI target in the SANDBOX
AgentCore Gateway, with an API-key credential provider (X-Proxy-Key header).

Prereqs:
  1. python tools_proxy.py          (serves the tools on :8090)
  2. ngrok http 8090                (gives https://<id>.ngrok-free.app)
  3. Set NGROK_URL below / env, and TOOLS_PROXY_KEY in config/.env

Usage:
  python create_gateway_target.py https://<id>.ngrok-free.app

What it does:
  - Fetches the OpenAPI spec from <ngrok>/openapi.json
  - Creates an API-key credential provider (sends X-Proxy-Key)
  - Creates a gateway target of type openApiSchema pointing at the spec

NOTE: Free ngrok URLs change on restart — re-run this with the new URL, or
delete the old target first. Confirm the gateway id/region below.
"""
from __future__ import annotations

import os
import sys
import json
import urllib.request

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "config", ".env"), override=True)

import boto3

REGION       = os.getenv("AWS_REGION", "ap-south-1")
GATEWAY_ID   = os.getenv("AGENTCORE_GATEWAY_ID",
                         "asl-aws-dev-orion-bot-agentcore-gateway-pgywge4cf0")
PROXY_KEY    = os.getenv("TOOLS_PROXY_KEY", "change-me-proxy-key")
TARGET_NAME  = os.getenv("TOOLS_TARGET_NAME", "chatbot-tools-ngrok")


def _fetch_openapi(ngrok_url: str) -> dict:
    url = ngrok_url.rstrip("/") + "/openapi.json"
    print(f"Fetching OpenAPI spec from {url} ...")
    req = urllib.request.Request(url, headers={"ngrok-skip-browser-warning": "1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        spec = json.loads(resp.read().decode("utf-8"))
    # Ensure the spec advertises the ngrok server URL so the gateway calls the tunnel.
    spec["servers"] = [{"url": ngrok_url.rstrip("/")}]
    print(f"  OK — {len(spec.get('paths', {}))} paths")
    return spec


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_gateway_target.py https://<id>.ngrok-free.app")
        sys.exit(1)
    ngrok_url = sys.argv[1].rstrip("/")

    spec = _fetch_openapi(ngrok_url)

    # Use the DEPLOYED AgentCore account credentials (MEMORY_AWS_*), which is
    # where the runtime + gateway live (106611079163) — NOT the client/model
    # account (AWS_* → 625867133907). Fall back to default creds if not set.
    mem_key = os.getenv("MEMORY_AWS_ACCESS_KEY_ID")
    mem_sec = os.getenv("MEMORY_AWS_SECRET_ACCESS_KEY")
    mem_tok = os.getenv("MEMORY_AWS_SESSION_TOKEN")
    if mem_key and mem_sec:
        client = boto3.client(
            "bedrock-agentcore-control", region_name=REGION,
            aws_access_key_id=mem_key, aws_secret_access_key=mem_sec,
            aws_session_token=mem_tok,
        )
        print("Using MEMORY_AWS_* credentials (deployed AgentCore account).")
    else:
        client = boto3.client("bedrock-agentcore-control", region_name=REGION)
        print("MEMORY_AWS_* not set — using default AWS credentials.")

    # ── 1. API-key credential provider (X-Proxy-Key header) ───────────────────
    print("\nCreating API-key credential provider ...")
    try:
        cred = client.create_api_key_credential_provider(
            name=f"{TARGET_NAME}-key",
            apiKey=PROXY_KEY,
        )
        cred_arn = cred["credentialProviderArn"]
        print(f"  credentialProviderArn = {cred_arn}")
    except client.exceptions.ConflictException:
        # Already exists — look it up.
        provs = client.list_api_key_credential_providers()
        cred_arn = next(
            p["credentialProviderArn"] for p in provs.get("credentialProviders", [])
            if p["name"] == f"{TARGET_NAME}-key"
        )
        print(f"  reusing existing credentialProviderArn = {cred_arn}")

    # ── 2. Gateway target (OpenAPI schema, inline) ────────────────────────────
    print("\nCreating gateway target ...")
    resp = client.create_gateway_target(
        gatewayIdentifier=GATEWAY_ID,
        name=TARGET_NAME,
        description="ngrok-tunnelled local chatbot tools proxy",
        targetConfiguration={
            "mcp": {
                "openApiSchema": {
                    "inlinePayload": json.dumps(spec),
                }
            }
        },
        credentialProviderConfigurations=[
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
        ],
    )
    print(f"\nSUCCESS")
    print(f"  targetId = {resp.get('targetId')}")
    print(f"  status   = {resp.get('status')}")
    print(f"\nGateway {GATEWAY_ID} now routes the tools to {ngrok_url}")
    print("Remember to set the ngrok URL in the AgentCore runtime env if the "
          "runtime calls the proxy directly.")


if __name__ == "__main__":
    main()
