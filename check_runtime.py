"""
Diagnose AgentCore Runtime configuration and test AgentCore Memory connectivity.
"""
import os
import boto3
import json
from dotenv import dotenv_values

# Single source of truth: config/.env (same file the deployed container and
# _create_runtime.py use). Falls back to the known-good values.
_cfg_path = os.path.join(os.path.dirname(__file__), "config", ".env")
_cfg = dotenv_values(_cfg_path)

REGION = _cfg.get("AWS_REGION", "ap-south-1")
RUNTIME_ID = "Asl_Web_chatbot_runtime-ev3kF2E2bY"
MEMORY_ID = _cfg.get("AGENTCORE_MEMORY_ID", "asl_web_chatbot_memory-HOlLFy7mrf")

print("\n=== 1. Checking AgentCore Runtime on AWS ===")
control = boto3.client("bedrock-agentcore-control", region_name=REGION)
try:
    rt = control.get_agent_runtime(agentRuntimeId=RUNTIME_ID)
    print(f"Status             : {rt.get('status')}")
    print(f"RoleArn            : {rt.get('roleArn')}")
    print(f"ContainerUri       : {rt.get('agentRuntimeArtifact', {}).get('containerConfiguration', {}).get('containerUri')}")
    print("Environment Variables:")
    for k, v in rt.get("environmentVariables", {}).items():
        print(f"  {k} = {v}")
except Exception as e:
    print(f"Error getting runtime: {e}")

print("\n=== 2. Testing Direct AgentCore Memory Read/Write ===")
data = boto3.client("bedrock-agentcore", region_name=REGION)
test_conv = "diag_test_conv"
test_state = json.dumps({"test": "hello", "flow": "statement", "flow_state": "statement_category"})

try:
    from datetime import datetime, timezone
    print(f"Attempting create_event on Memory [{MEMORY_ID}]...")
    resp = data.create_event(
        memoryId=MEMORY_ID,
        actorId="web-chatbot-anonymous",
        sessionId=test_conv,
        eventTimestamp=datetime.now(timezone.utc),
        payload=[{"conversational": {"content": {"text": test_state}, "role": "USER"}}],
    )
    print("SUCCESS: create_event wrote to AgentCore Memory!")
    
    print("Attempting list_events from Memory...")
    read_resp = data.list_events(
        memoryId=MEMORY_ID,
        actorId="web-chatbot-anonymous",
        sessionId=test_conv,
        maxResults=1,
    )
    events = read_resp.get("events", [])
    if events:
        text = events[0].get("payload", [])[0].get("conversational", {}).get("content", {}).get("text", "")
        print(f"SUCCESS: Read back event from AgentCore Memory: {text}")
    else:
        print("WARNING: list_events returned 0 events!")
except Exception as e:
    print(f"FAILED: AgentCore Memory operation failed with error:\n  {type(e).__name__}: {e}")
