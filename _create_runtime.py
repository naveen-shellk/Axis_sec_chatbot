"""
chatbot_web/_create_runtime.py
--------------------------------
Creates or updates the AgentCore Runtime using boto3.
Called by deploy_agentcore.ps1 after docker push.

Usage:
  python _create_runtime.py <account_id> <region> <image_uri> <runtime_name> <role_arn> <web_token>
"""

from __future__ import annotations

import sys
import os
import json


from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "config", ".env"), override=False)


def main():
    if len(sys.argv) < 7:
        print("Usage: python _create_runtime.py <account_id> <region> <image_uri> <runtime_name> <role_arn> <web_token>")
        sys.exit(1)

    account_id   = sys.argv[1]
    region       = sys.argv[2]
    image_uri    = sys.argv[3]
    runtime_name = sys.argv[4]
    role_arn     = sys.argv[5]
    web_token    = sys.argv[6]

    gateway_url = (
        f"https://asl-aws-dev-orion-bot-agentcore-gateway-pgywge4cf0"
        f".gateway.bedrock-agentcore.{region}.amazonaws.com/mcp"
    )

    memory_id       = os.getenv("AGENTCORE_MEMORY_ID", "asl_web_chatbot_memory-HOlLFy7mrf")
    session_backend = os.getenv("SESSION_BACKEND", "agentcore")

    # NOTE: Do NOT include AWS_ACCESS_KEY_ID / SECRET / SESSION_TOKEN here.
    # The runtime container must use its own IAM role (AslWebChatbotRuntimeRole)
    # via IMDS — not the deployer's SSO credentials.
    env_vars = {
        "AWS_REGION":            region,
        "SESSION_BACKEND":       session_backend,
        "AGENTCORE_MEMORY_ID":   memory_id,
        "SESSION_TTL_SECONDS":   "900",
        "INTENT_MODEL_ID":       "anthropic.claude-3-haiku-20240307-v1:0",
        "RESPONSE_MODEL_ID":     "qwen.qwen3-235b-a22b-2507-v1:0",
        "AGENTCORE_GATEWAY_URL": gateway_url,
        "WEB_API_TOKEN":         web_token,
        "LOG_LEVEL":             "INFO",
        "ENVIRONMENT":           "uat",
    }

    mem_key = os.getenv("MEMORY_AWS_ACCESS_KEY_ID")
    mem_sec = os.getenv("MEMORY_AWS_SECRET_ACCESS_KEY")
    mem_tok = os.getenv("MEMORY_AWS_SESSION_TOKEN")
    if mem_key and mem_sec:
        env_vars["MEMORY_AWS_ACCESS_KEY_ID"] = mem_key
        env_vars["MEMORY_AWS_SECRET_ACCESS_KEY"] = mem_sec
        if mem_tok:
            env_vars["MEMORY_AWS_SESSION_TOKEN"] = mem_tok

    # chatbot_web only handles pre-login flows — no private API calls needed.
    # PUBLIC mode is sufficient. Switch to VPC later if private APIs are added.
    network_config = {
        "networkMode": "PUBLIC"
    }

    import boto3
    boto_kwargs = {"region_name": region}
    mem_key = os.getenv("MEMORY_AWS_ACCESS_KEY_ID")
    mem_sec = os.getenv("MEMORY_AWS_SECRET_ACCESS_KEY")
    mem_tok = os.getenv("MEMORY_AWS_SESSION_TOKEN")
    if mem_key and mem_sec:
        boto_kwargs["aws_access_key_id"] = mem_key
        boto_kwargs["aws_secret_access_key"] = mem_sec
        if mem_tok:
            boto_kwargs["aws_session_token"] = mem_tok

    client = boto3.client("bedrock-agentcore-control", **boto_kwargs)

    def do_create():
        print(f"Creating runtime '{runtime_name}'...")
        return client.create_agent_runtime(
            agentRuntimeName=runtime_name,
            description="Axis Direct web channel pre-login chatbot",
            roleArn=role_arn,
            agentRuntimeArtifact={
                "containerConfiguration": {"containerUri": image_uri}
            },
            networkConfiguration=network_config,
            environmentVariables=env_vars,
        )

    def do_update():
        print(f"Runtime already exists — finding runtimeId for '{runtime_name}'...")
        runtimes = client.list_agent_runtimes().get("agentRuntimes", [])
        rt_match = next((r for r in runtimes if r.get("agentRuntimeName") == runtime_name), None)
        if not rt_match:
            raise RuntimeError(f"Could not find agentRuntimeId for existing runtime: {runtime_name}")
        rt_id = rt_match["agentRuntimeId"]
        print(f"Updating '{runtime_name}' ({rt_id})...")
        return client.update_agent_runtime(
            agentRuntimeId=rt_id,
            agentRuntimeArtifact={
                "containerConfiguration": {"containerUri": image_uri}
            },
            roleArn=role_arn,
            networkConfiguration=network_config,
            environmentVariables=env_vars,
        )

    try:
        try:
            resp = do_create()
        except client.exceptions.ConflictException:
            resp = do_update()

        rt_id  = resp.get("agentRuntimeId", resp.get("agentRuntimeArn", "unknown"))
        status = resp.get("status", "UPDATING")
        print(f"SUCCESS")
        print(f"RuntimeId : {rt_id}")
        print(f"Status    : {status}")
        print(json.dumps({"agentRuntimeId": rt_id, "status": status}))

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
