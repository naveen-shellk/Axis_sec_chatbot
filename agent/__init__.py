"""
agent — the channel-agnostic brain of the Axis Direct chatbot.

Exposes the single orchestration entry point, handle_message(), which every
channel adapter (web /api/chat today, WhatsApp later) calls. All routing,
flow state-machines, LLM calls, Knowledge Base retrieval, gateway/API clients,
and session handling live under this package. Channel adapters only translate
their transport payloads to/from handle_message().
"""
