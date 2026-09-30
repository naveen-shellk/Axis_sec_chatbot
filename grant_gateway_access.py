"""
chatbot_web/grant_gateway_access.py
-----------------------------------
Grant the deployed runtime role permission to INVOKE the new AgentCore Gateway.

The runtime (AslWebChatbotRuntimeRole) calls the gateway via SigV4 using its own
IAM role. The gateway uses IAM inbound auth, so the role needs an explicit
bedrock-agentcore invoke permission on the gateway ARN — otherwise the call
gets 403 AccessDenied and falls back to (unreachable) direct HTTP.

Run:
    python grant_gateway_access.py
"""
from __future__ import annotations

import os
import json

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "config", ".env"), override=True)

import boto3

ROLE_NAME   = os.getenv("RUNTIME_ROLE_NAME", "AslWebChatbotRuntimeRole")
GATEWAY_ARN = os.getenv(
    "GATEWAY_ARN",
    "arn:aws:bedrock-agentcore:ap-south-1:106611079163:gateway/asl-web-chatbot-gateway-y9c3ekf6p8",
)
POLICY_NAME = "AgentCoreGatewayInvoke"


def _iam():
    return boto3.client(
        "iam",
        aws_access_key_id=os.getenv("MEMORY_AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("MEMORY_AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("MEMORY_AWS_SESSION_TOKEN"),
    )


def main():
    iam = _iam()
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeChatbotGateway",
                "Effect": "Allow",
                "Action": [
                    "bedrock-agentcore:InvokeGateway",
                    "bedrock-agentcore:GetGateway",
                    "bedrock-agentcore:ListGatewayTargets",
                    "bedrock-agentcore:GetGatewayTarget",
                ],
                # Cover the gateway and its targets (target ARNs are children).
                "Resource": [
                    GATEWAY_ARN,
                    GATEWAY_ARN + "/target/*",
                ],
            }
        ],
    }

    print(f"Attaching inline policy '{POLICY_NAME}' to role '{ROLE_NAME}' ...")
    print(json.dumps(policy, indent=2))
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName=POLICY_NAME,
        PolicyDocument=json.dumps(policy),
    )
    print("\nSUCCESS — runtime role can now invoke the gateway.")
    print("Note: IAM changes are picked up by the running container within a few "
          "minutes; a redeploy guarantees it.")


if __name__ == "__main__":
    main()
