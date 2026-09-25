"""
chatbot_web/src/gateways/mcp_gateway.py
-----------------------------------------
AgentCore Gateway MCP client — DISABLED for local testing.

The Gateway is not reachable in local_test environment.
call_tool() immediately raises MCPGatewayError so gateway_client.py
falls through to direct HTTP on every call — zero latency overhead.

To re-enable when Gateway is available:
  Set ENABLE_AGENTCORE_GATEWAY=true in config/.env
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_ENABLE_GATEWAY = os.getenv("ENABLE_AGENTCORE_GATEWAY", "false").lower() == "true"


class MCPGatewayError(Exception):
    """Raised on MCP protocol or HTTP errors."""


def call_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Call a single MCP tool via AgentCore Gateway.
    Disabled until Gateway is available — raises MCPGatewayError immediately
    so gateway_client.py falls through to direct HTTP.
    """
    if not _ENABLE_GATEWAY:
        raise MCPGatewayError(
            f"Gateway disabled — ENABLE_AGENTCORE_GATEWAY=false. "
            f"Tool '{tool_name}' will use direct HTTP fallback."
        )

    # ── Full Gateway implementation (re-enabled when ENABLE_AGENTCORE_GATEWAY=true) ──
    import json
    import urllib.error
    import urllib.request
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.session import Session as BotocoreSession

    _GATEWAY_URL = os.getenv(
        "AGENTCORE_GATEWAY_URL",
        "https://asl-aws-dev-orion-bot-agentcore-gateway-pgywge4cf0"
        ".gateway.bedrock-agentcore.ap-south-1.amazonaws.com/mcp",
    )
    _REGION  = os.getenv("AWS_REGION", "ap-south-1")
    _SERVICE = "bedrock-agentcore"
    _TIMEOUT = 10

    # Resolve credentials
    session  = BotocoreSession()
    resolver = session.get_component("credential_provider")
    creds    = resolver.load_credentials()
    if creds is None:
        raise MCPGatewayError("No AWS credentials found.")

    # Build JSON-RPC payload
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id":      f"chatbot-{tool_name}",
        "method":  "tools/call",
        "params":  {"name": tool_name, "arguments": arguments},
    }).encode("utf-8")

    # SigV4 sign
    aws_req = AWSRequest(
        method="POST", url=_GATEWAY_URL, data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
    )
    SigV4Auth(creds, _SERVICE, _REGION).add_auth(aws_req)
    req = urllib.request.Request(url=_GATEWAY_URL, data=payload, headers=dict(aws_req.headers), method="POST")

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise MCPGatewayError(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc
    except urllib.error.URLError as exc:
        raise MCPGatewayError(f"Network error: {exc.reason}") from exc

    data    = json.loads(raw)
    if "error" in data:
        err = data["error"]
        raise MCPGatewayError(f"MCP error {err.get('code')}: {err.get('message')}")

    result  = data.get("result", {})
    content = result.get("content", []) if isinstance(result, dict) else []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            try:
                return json.loads(block["text"])
            except (json.JSONDecodeError, KeyError):
                return {"text": block.get("text", "")}
    return result or {}


def list_tools() -> list[dict[str, Any]]:
    """List available Gateway tools — only works when Gateway is enabled."""
    if not _ENABLE_GATEWAY:
        logger.info("[MCP] Gateway disabled — list_tools returns []")
        return []
    return []


def get_gateway_url() -> str:
    return os.getenv("AGENTCORE_GATEWAY_URL", "")
