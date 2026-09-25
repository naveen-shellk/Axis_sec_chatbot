"""
chatbot_web/setup_memory.py
-----------------------------
One-time script: creates the AgentCore Memory resource and prints
the memory_id to add to config/.env.

Run ONCE before starting the server with AgentCore backend:

    cd chatbot_web
    python setup_memory.py

Prerequisites:
  - AWS credentials in config/.env (or environment)
  - boto3 installed

Output:
  Memory ID: mem-xxxxxxxxxxxxxxxx
  
  Add this to chatbot_web/config/.env:
    SESSION_BACKEND=agentcore
    AGENTCORE_MEMORY_ID=mem-xxxxxxxxxxxxxxxx
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "config", ".env"), override=True)

import boto3

AWS_REGION  = os.getenv("AWS_REGION", "ap-south-1")
MEMORY_NAME = os.getenv("AGENTCORE_MEMORY_NAME", "asl_web_chatbot_memory")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def main():
    print(f"\n{BOLD}{CYAN}AgentCore Memory Setup{RESET}")
    print(f"Region : {AWS_REGION}")
    print(f"Name   : {MEMORY_NAME}\n")

    kwargs = {"region_name": AWS_REGION}
    mem_key = os.getenv("MEMORY_AWS_ACCESS_KEY_ID")
    mem_secret = os.getenv("MEMORY_AWS_SECRET_ACCESS_KEY")
    mem_token = os.getenv("MEMORY_AWS_SESSION_TOKEN")
    if mem_key and mem_secret:
        kwargs["aws_access_key_id"] = mem_key
        kwargs["aws_secret_access_key"] = mem_secret
        if mem_token:
            kwargs["aws_session_token"] = mem_token

    sts = boto3.client("sts", **kwargs)
    try:
        caller = sts.get_caller_identity()
        print(f"Target AWS Account: {BOLD}{caller.get('Account')}{RESET} ({caller.get('Arn')})\n")
    except Exception as exc:
        print(f"{YELLOW}Could not verify caller identity: {exc}{RESET}\n")

    control = boto3.client("bedrock-agentcore-control", **kwargs)

    # ── Check if a memory with this name already exists ───────────────────────
    print("Checking for existing memory resources...")
    try:
        list_resp = control.list_memories()
        existing = [
            m for m in list_resp.get("memories", [])
            if m.get("name") == MEMORY_NAME
        ]
        if existing:
            memory_id = existing[0]["id"]
            status    = existing[0].get("status", "UNKNOWN")
            print(f"{YELLOW}Memory already exists:{RESET}")
            print(f"  Name   : {MEMORY_NAME}")
            print(f"  ID     : {memory_id}")
            print(f"  Status : {status}")
            _print_result(memory_id)
            return
    except Exception as exc:
        print(f"{YELLOW}Could not list memories: {exc} — proceeding to create{RESET}")

    # ── Create the Memory resource ────────────────────────────────────────────
    print(f"Creating memory resource '{MEMORY_NAME}'...")
    try:
        response = control.create_memory(
            name=MEMORY_NAME,
            description="Short-term session memory for ASL Web Chatbot pre-login flows",
            eventExpiryDuration=3,   # AWS minimum is 3 days
            # No memoryStrategies = pure short-term memory (fast, no async extraction)
            # Uncomment below to also enable session summarisation (long-term memory):
            # memoryStrategies=[
            #     {
            #         "summaryMemoryStrategy": {
            #             "name": "SessionSummarizer",
            #             "namespaceTemplates": ["/summaries/{actorId}/{sessionId}/"]
            #         }
            #     }
            # ]
        )
        memory_id = response["memory"]["id"]
        print(f"  Created. ID: {memory_id}")
    except Exception as exc:
        print(f"{RED}Failed to create memory: {exc}{RESET}")
        sys.exit(1)

    # ── Wait for ACTIVE status ────────────────────────────────────────────────
    print("Waiting for memory to become ACTIVE", end="", flush=True)
    for attempt in range(30):
        try:
            status_resp = control.get_memory(memoryId=memory_id)
            status = status_resp.get("memory", {}).get("status", "")
            if status == "ACTIVE":
                print(f" {GREEN}ACTIVE{RESET}")
                break
            if status == "FAILED":
                print(f" {RED}FAILED{RESET}")
                print(f"{RED}Memory creation failed: {status_resp}{RESET}")
                sys.exit(1)
            print(".", end="", flush=True)
            time.sleep(5)
        except Exception as exc:
            print(f"\n{YELLOW}Status check error: {exc}{RESET}")
            time.sleep(5)
    else:
        print(f"\n{YELLOW}Timed out waiting for ACTIVE status. Check AWS console.{RESET}")

    _print_result(memory_id)


def _print_result(memory_id: str):
    print(f"\n{BOLD}{'='*55}{RESET}")
    print(f"{GREEN}Memory ready!{RESET}")
    print(f"\nAdd these lines to {BOLD}chatbot_web/config/.env{RESET}:\n")
    print(f"  SESSION_BACKEND=agentcore")
    print(f"  AGENTCORE_MEMORY_ID={memory_id}")
    print(f"\n{CYAN}Then restart the server:{RESET}")
    print(f"  cd chatbot_web && python app.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
