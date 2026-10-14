"""
channels — thin transport adapters that sit in front of the agent brain.

Each channel (web today, WhatsApp later) only translates its own request/response
format to and from agent.handler.handle_message(). No business logic lives here.
"""
