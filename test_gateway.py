"""
chatbot_web/test_gateway.py
---------------------------
Test whether the new AgentCore Gateway is reachable and its tools work,
by making SigV4-signed MCP calls (same mechanism the deployed runtime uses),
signed with the DEPLOYED-account credentials (MEMORY_AWS_*).

    1. tools/list  — confirms the gateway is reachable + target is wired
    2. tools/call  — invokes get_customer_profile through gateway → ngrok → proxy

Usage:
    python test_gateway.py                # lists tools
    python test_gateway.py 6033593        # also calls get_customer_profile
"""
from __future__ import annotations

import os
import sys
import json
import urllib.request
import urllib.error

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "config", ".env"), override=True)

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

REGION   = "ap-south-1"
SERVICE  = "bedrock-agentcore"
# Always target the NEW gateway (env in .env still points at the old one).
GW_URL   = "https://asl-web-chatbot-gateway-y9c3ekf6p8.gateway.bedrock-agentcore.ap-south-1.amazonaws.com/mcp"

_creds = Credentials(
    access_key=os.getenv("MEMORY_AWS_ACCESS_KEY_ID"),
    secret_key=os.getenv("MEMORY_AWS_SECRET_ACCESS_KEY"),
    token=os.getenv("MEMORY_AWS_SESSION_TOKEN"),
)


def _rpc(method: str, params: dict | None = None) -> dict:
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id":      f"test-{method}",
        "method":  method,
        "params":  params or {},
    }).encode("utf-8")

    aws_req = AWSRequest(
        method="POST", url=GW_URL, data=payload,
        headers={"Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream"},
    )
    SigV4Auth(_creds, SERVICE, REGION).add_auth(aws_req)
    req = urllib.request.Request(url=GW_URL, data=payload,
                                 headers=dict(aws_req.headers), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        return {"_http_error": exc.code, "_body": body}
    except urllib.error.URLError as exc:
        return {"_network_error": str(exc.reason)}

    # SSE responses may prefix with "data:"
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw[:500]}


def main():
    print(f"Gateway URL: {GW_URL}\n")

    print("=== 1. tools/list (is the gateway reachable + target wired?) ===")
    r = _rpc("tools/list")
    print(json.dumps(r, indent=2)[:1500])

    if "_http_error" in r:
        print(f"\n>>> HTTP {r['_http_error']} — likely IAM permission (403) or gateway config.")
        print("    If 403: the signing identity lacks bedrock-agentcore gateway-invoke permission.")
        return

    if len(sys.argv) > 1:
        sub = sys.argv[1]
        print(f"\n=== 2. tools/call get_customer_profile (sub={sub}) ===")
        r2 = _rpc("tools/call", {"name": "get_customer_profile",
                                 "arguments": {"sub_account_id": sub}})
        print(json.dumps(r2, indent=2)[:1500])


if __name__ == "__main__":
    main()
