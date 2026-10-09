"""
chatbot_langchain/src/core/langchain_agent.py
----------------------------------------------
LangChain agent for the web channel — the LangChain port of chatbot_web's
Strands agent (src/core/strands_agent.py).

Public surface is IDENTICAL to the Strands version so the flow handlers and
startup hooks are unchanged (only the import path differs):
  - run_tool(tool_name, /, **kwargs) -> dict | None   (deterministic tool bridge)
  - run_agent_turn(prompt) -> dict                     (fully-agentic ReAct turn)
  - get_profile(sub_account_id) -> CustomerProfile     (typed profile fetch)
  - warmup() -> None                                   (startup pre-init)
  - escalate(reason) / check_account(customer_id)      (legacy helpers, no callers)

Framework mapping (Strands -> LangChain):
  strands.Agent + strands.models.BedrockModel  ->  langchain_aws.ChatBedrockConverse
  @tool + _agent.tool.<name>()                 ->  plain funcs in tools.TOOL_FUNCS
  _agent(prompt) ReAct loop                    ->  manual bind_tools + tool loop here

The conversational (deterministic) LLM path in src/core/llm.py is untouched —
this module is only the agentic layer, same as in chatbot_web.

AgentCore deployment:
  ChatBedrockConverse uses the ambient boto3 credentials — explicit AWS keys in
  local dev (.env), and the AgentCore Runtime IAM role in production (no keys).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_aws import ChatBedrockConverse

from src.core.tools import ALL_TOOLS, TOOL_FUNCS

logger = logging.getLogger(__name__)

_MODEL_ID    = os.getenv("RESPONSE_MODEL_ID", "qwen.qwen3-235b-a22b-2507-v1:0")  # Qwen for responses
_REGION      = os.getenv("AWS_REGION", "ap-south-1")
_MAX_TOKENS  = int(os.getenv("CHATBOT_MAX_TOKENS", "512"))
_TEMPERATURE = float(os.getenv("CHATBOT_TEMPERATURE", "0.2"))
_MAX_TOOL_ITERS = int(os.getenv("AGENT_MAX_TOOL_ITERS", "6"))

_AGENT_SYSTEM_PROMPT = (
    "You are the Axis Direct virtual assistant for the web (pre-login) channel. "
    "You help customers with: bank queries, how to trade, editing profile, account "
    "statements, IPO, account details, brokerage & charges, login queries, order "
    "status, and account closure.\n\n"
    "You have tools to fetch customer profiles, request statements, get orders, "
    "fetch ledger balances, send DP bills, create account-closure requests, "
    "request authentication, and escalate to a live agent. DECIDE YOURSELF which "
    "tools to call and in what order to fulfil the customer's request. Call "
    "get_customer_profile_full first when you need account data.\n\n"
    "Rules:\n"
    "- Account-specific actions (statement, order status, account details, "
    "brokerage & charges, login help, account closure, IPO) require a VERIFIED "
    "Sub-Account ID. If the context says the customer is NOT authenticated / no "
    "Sub-Account ID is provided, call request_authentication FIRST — do NOT ask "
    "for the account number yourself and do NOT call any account tool. The system "
    "will run secure OTP verification and resolve the account.\n"
    "- If a verified Sub-Account ID IS provided in the context, use it directly; "
    "never invent one.\n"
    "- General questions (how to trade, bank queries, support hours, editing "
    "profile guidance) do NOT need authentication — answer them directly.\n"
    "- If the customer explicitly asks for a human/live agent, call escalate_to_agent.\n"
    "- Be warm, concise, and professional.\n"
    "- Do not fabricate data — only state what the tools return.\n\n"
    "OUTPUT FORMAT — your FINAL reply to the customer MUST be a single JSON object:\n"
    '  {"message": "<your reply text>", "quick_reply_set": "<set name or empty>"}\n'
    "The buttons shown to the customer come from a FIXED catalogue. You do NOT "
    "write button labels — you only name WHICH set to show via quick_reply_set. "
    "Do NOT list the options in the message body; just ask the question and name "
    "the set. Allowed quick_reply_set values:\n"
    "  \"statement_categories\" — when asking which statement/report category "
    "(Tax / Demat / Trading reports).\n"
    "  \"segments\"             — when asking which market segment.\n"
    "  \"order_types\"          — when asking Today Order Status vs Order History.\n"
    "  \"closure_types\"        — when asking which account to close (Trading/Demat/Both).\n"
    "  \"yes_no\"               — for a yes/no question.\n"
    "  \"main_menu\"            — to show the full main menu.\n"
    "  \"\"                     — (empty) when there are no choices to offer.\n"
    "Output ONLY the JSON object — no markdown fences, no extra text."
)


# ── Lazy singletons ───────────────────────────────────────────────────────────
_llm: Any = None            # base ChatBedrockConverse
_llm_with_tools: Any = None  # model.bind_tools(ALL_TOOLS)


def _get_llm():
    """Build (once) the ChatBedrockConverse client and its tool-bound variant."""
    global _llm, _llm_with_tools
    if _llm is None:
        _llm = ChatBedrockConverse(
            model=_MODEL_ID,
            region_name=_REGION,
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
        )
        _llm_with_tools = _llm.bind_tools(ALL_TOOLS)
        logger.info("[LC] ChatBedrockConverse initialised model=%s region=%s", _MODEL_ID, _REGION)
    return _llm, _llm_with_tools


# ── Legacy helpers (kept for parity — no external callers) ────────────────────

def escalate(reason: str) -> dict[str, Any]:
    """Signal live agent escalation. (Legacy — no callers.)"""
    logger.info("[LC] escalate: %s", reason)
    return {"escalate": True, "reason": reason, "eventid": "1002"}


def check_account(customer_id: str) -> dict[str, Any]:
    """Check account status via a direct tool call. (Legacy — no callers.)"""
    logger.info("[LC] check_account: %s", customer_id)
    try:
        result = TOOL_FUNCS["get_account_status"](sub_account_id=customer_id)
        status = result.get("status", "active") if isinstance(result, dict) else "active"
        return {
            "action": "proceed" if status == "active" else "blocked",
            "tool_results": {"get_account_status": {"status": status}},
        }
    except Exception as exc:
        logger.error("[LC] check_account failed: %s — defaulting active", exc)
        return {
            "action": "proceed",
            "tool_results": {"get_account_status": {"status": "active"}},
        }


# ── Generic tool bridge ───────────────────────────────────────────────────────
# Flow handlers call run_tool() to invoke a named tool. In the LangChain variant
# this calls the raw Python function directly (reliable, no framework unwrap),
# preserving the same signature/return contract the Strands version exposed.

def run_tool(tool_name: str, /, **kwargs: Any) -> dict[str, Any] | None:
    """
    Invoke a registered tool by name.
    Returns the tool's JSON dict, or None if it failed (caller then uses its own
    direct fallback — same contract as the Strands version).
    """
    logger.info("[LC] run_tool %s args=%s", tool_name, list(kwargs))
    fn = TOOL_FUNCS.get(tool_name)
    if fn is None:
        logger.error("[LC] run_tool: unknown tool %r", tool_name)
        return None
    try:
        result = fn(**kwargs)
        return result if isinstance(result, dict) else None
    except Exception as exc:
        logger.error("[LC] run_tool %s failed: %s — caller will fall back", tool_name, exc)
        return None


def _parse_agent_output(text: str) -> tuple[str, str]:
    """
    Parse the agent's final answer against the JSON output contract
      {"message": str, "quick_reply_set": str}.
    Returns (message, quick_reply_set_name). The handler maps the set name to a
    fixed, complete button list. Robust to markdown fences / non-JSON output.
    """
    import json as _json
    import re as _re

    if not text:
        return "", ""
    clean = text.strip()
    # Strip ```json fences if present.
    if "```" in clean:
        m = _re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
        if m:
            clean = m.group(1).strip()
    obj = None
    try:
        obj = _json.loads(clean)
    except Exception:
        m = _re.search(r"\{[\s\S]*\}", clean)
        if m:
            try:
                obj = _json.loads(m.group(0))
            except Exception:
                obj = None
    if isinstance(obj, dict) and "message" in obj:
        msg = str(obj.get("message", "")).strip()
        qset = str(obj.get("quick_reply_set", "") or "").strip()
        return msg, qset

    # Salvage: prose with an inline `quick_reply_set: name` fragment instead of
    # clean JSON. Extract the set name and strip that fragment from the message.
    m = _re.search(r'quick[_ ]?reply[_ ]?set\s*[:=]\s*"?([a-z_]+)"?', clean, _re.IGNORECASE)
    if m:
        qset = m.group(1).strip()
        msg = clean[:m.start()].rstrip().rstrip("{").rstrip(",").strip()
        msg = _re.sub(r'^\{?\s*"?message"?\s*[:=]\s*"?', "", msg).strip().strip('"').strip()
        return (msg or clean), qset

    # Not JSON — treat the whole thing as the message.
    return clean, ""


def run_agent_turn(prompt: str) -> dict[str, Any]:
    """
    Fully agentic turn: hand the prompt to the LangChain model (with tools bound)
    and let IT decide which tools to call, in what order, and when it's done
    (ReAct-style loop, implemented here over ChatBedrockConverse.bind_tools).

    Returns:
        {"message": str, "escalate": bool, "input_tokens": int, "output_tokens": int}
    """
    from langchain_core.messages import (
        SystemMessage, HumanMessage, AIMessage, ToolMessage,
    )
    import json as _json

    logger.info("[LC] run_agent_turn (LLM decides tools) prompt_len=%d", len(prompt))
    _, llm_tools = _get_llm()

    messages: list[Any] = [
        SystemMessage(content=_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    in_tok = out_tok = 0
    escalate_flag = False
    final_text = ""

    try:
        for _iter in range(_MAX_TOOL_ITERS):
            ai: AIMessage = llm_tools.invoke(messages)
            messages.append(ai)

            # Accumulate token usage across every model call in the loop.
            um = getattr(ai, "usage_metadata", None) or {}
            in_tok  += int(um.get("input_tokens", 0) or 0)
            out_tok += int(um.get("output_tokens", 0) or 0)

            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls:
                # No more tools requested — this is the final answer.
                # `.text` is a property in langchain-core 1.x; fall back to
                # `.content` (which may be a list of blocks) for older versions.
                txt = getattr(ai, "text", None)
                if callable(txt):          # older langchain-core: .text() method
                    txt = txt()
                if not txt:
                    c = ai.content
                    txt = c if isinstance(c, str) else " ".join(
                        b.get("text", "") for b in c if isinstance(b, dict)
                    ) if isinstance(c, list) else str(c)
                final_text = txt
                break

            # Execute each requested tool and feed results back to the model.
            for tc in tool_calls:
                name = tc.get("name")
                args = tc.get("args", {}) or {}
                call_id = tc.get("id")
                # Auth gate: if the agent asks to authenticate the customer,
                # short-circuit the whole turn — the handler starts the secure
                # OTP phone flow instead of letting the agent continue.
                if name == "request_authentication":
                    logger.info("[LC] agent requested authentication → needs_auth")
                    return {"message": "", "escalate": False, "needs_auth": True,
                            "input_tokens": in_tok, "output_tokens": out_tok}
                if name == "escalate_to_agent":
                    escalate_flag = True
                fn = TOOL_FUNCS.get(name)
                if fn is None:
                    result = {"error": f"unknown tool {name}"}
                else:
                    try:
                        result = fn(**args)
                    except Exception as exc:
                        logger.error("[LC] agent tool %s failed: %s", name, exc)
                        result = {"error": str(exc)}
                messages.append(ToolMessage(
                    content=_json.dumps(result, default=str),
                    tool_call_id=call_id,
                ))
        else:
            # Loop exhausted without a tool-free reply — use the last text we have.
            logger.warning("[LC] run_agent_turn hit max iterations (%d)", _MAX_TOOL_ITERS)

        final_text = (final_text or "").strip()

        # Parse the agent's JSON output contract {message, quick_reply_set}.
        message, quick_reply_set = _parse_agent_output(final_text)

        # Safety net: if the model narrated an auth need WITHOUT calling
        # request_authentication, still trigger the auth gate so it never leaks.
        _t = message.lower()
        if ("verify your identity" in _t or "verify your account" in _t
                or "otp verification" in _t or "authentication process" in _t):
            logger.info("[LC] agent narrated auth need → needs_auth (safety net)")
            return {"message": "", "escalate": False, "needs_auth": True,
                    "input_tokens": in_tok, "output_tokens": out_tok}

        if not escalate_flag:
            escalate_flag = "1002" in message or "escalate to a live agent" in message.lower()

        # ── Token accounting — push into the per-turn accumulator so log.txt
        # records correct totals (same bookkeeping the Strands version did).
        try:
            from src.core.conversation import _turn_tokens
            _turn_tokens["input_tokens"]           += in_tok
            _turn_tokens["output_tokens"]          += out_tok
            _turn_tokens["response_input_tokens"]  += in_tok
            _turn_tokens["response_output_tokens"] += out_tok
            _turn_tokens["llm_call_count"]         += 1
        except Exception as exc:
            logger.warning("[LC] token accounting failed: %s", exc)

        return {"message": message, "quick_reply_set": quick_reply_set,
                "escalate": escalate_flag,
                "input_tokens": in_tok, "output_tokens": out_tok}
    except Exception as exc:
        logger.error("[LC] run_agent_turn failed: %s", exc)
        return {"message": "", "escalate": False, "error": str(exc)}


# ── Agent-as-decision-maker: routing decision ─────────────────────────────────
# Option (a): the AGENT decides the top-level intent/flow + control action for a
# turn. It does NOT write the customer reply — the chosen deterministic flow
# emits the hardcoded messages/quick-replies. This replaces the plain Haiku
# classifier so routing is an agent decision (context-aware: mid-flow, auth
# state), while responses stay hardcoded (and flows still use Qwen where they
# already do).

_ROUTER_DECISION_SYSTEM = (
    "You route turns for the Axis Direct chatbot. You ONLY classify intent — you "
    "do NOT write replies, and you NEVER output dates, amounts, periods, report "
    "names, or segments (a separate extractor does that). Use recent history for "
    "context (customer may be mid-flow). Classify by the customer's ACTUAL goal, "
    "not isolated keywords.\n\n"
    "Intents:\n"
    "  bank_query       — Axis BANK products (loan, credit card, savings, branch)\n"
    "  how_to_trade     — HOW DO I / HOW TO place/buy/sell/execute a trade (wants the steps)\n"
    "  edit_profile     — update profile (email, mobile, address)\n"
    "  statement        — wants a DOCUMENT: statement/report/contract note/trade book/CML, "
    "or 'email/send/download me my ...'\n"
    "  ipo              — apply for or check IPO\n"
    "  account_details  — static info: demat number, trading id, DP id, account status\n"
    "  brokerage        — VIEW charges/fees: 'show/see/check my charges', 'why was I "
    "charged', DP/AMC/brokerage amounts (NO document word)\n"
    "  login_query      — can't login, forgot password, FTL, account locked\n"
    "  order_status     — status/history of orders or trades placed\n"
    "  closure          — close demat/trading account\n"
    "  faq              — INFORMATIONAL/conceptual: 'what is X', 'how does X work', "
    "'explain/define X', eligibility/permission/rules questions. The verb test: "
    "'Can I / Is it possible / Am I allowed / Will there be / What happens if / Do I "
    "get' + any topic = faq (NOT how_to_trade, NOT statement). Also account-opening, "
    "eligibility, 2FA/security, 'what margin is charged' = faq.\n"
    "  greeting         — PURE greeting, no embedded intent\n"
    "  need_more_help   — EXPLICIT live-agent/human request\n"
    "  escalate_to_human— ONLY fraud/disputes/grievances no self-service flow handles\n"
    "  unknown          — anything not clearly above\n\n"
    "Key rules:\n"
    "- Multi-intent: if the message clearly covers >1 topic, list all (max 3) in the "
    "order raised; else a single-element list.\n"
    "- statement needs an explicit DOCUMENT word (statement/report/email/send/download). "
    "'show/see/check my charges' without one = brokerage ALONE (not multi-intent).\n"
    "- A statement ABOUT charges ('brokerage charges statement') = statement, not brokerage.\n"
    "- A product word (Encash, GTDT, intraday, contract, margin) does NOT force "
    "how_to_trade/statement — permission/explanation question = faq. Only 'how to "
    "sell/buy/place' = how_to_trade.\n"
    "- Conceptual terms (MTM, margin, demat, settlement) = faq, NOT bank_query.\n"
    "- A greeting that embeds an intent ('hi, I want my statement') → the embedded intent.\n"
    "- ALWAYS prefer a serviceable intent over escalation. Length, politeness, or "
    "multiple date ranges do NOT make a request escalate_to_human.\n"
    "- When unsure → ['unknown'].\n"
    "- complaint=true ONLY for a genuine grievance about something WRONG ('complaint', "
    "'wrong charges', 'overcharged', 'not received', 'dispute', 'why was I charged'); a "
    "message can be both a serviceable intent AND a complaint. 'wrong year/date' is NOT "
    "a complaint. Neutral view/download requests = false.\n\n"
    "Return ONLY JSON — no markdown, no prose:\n"
    "{\n"
    '  "intents":        ["<intent1>", "<intent2>"],\n'
    '  "confidence":     <0.0-1.0>,\n'
    '  "complaint":      <true|false>,\n'
    '  "reasoning":      "<one sentence>",\n'
    '  "sub_account_id": "<5-10 digit account id if present, else null>"\n'
    "}"
)


def run_router_decision(
    customer_message: str,
    recent_history: list[dict] | None = None,
    authenticated: bool = False,
    active_flow: str | None = None,
) -> dict[str, Any]:
    """
    Agent routing DECISION (Option a). Uses HAIKU (fast, cheap classifier) to
    decide intent/flow + control for this turn — routing is a bounded
    classification task Haiku handles well, so we don't pay Qwen prices for it.
    Qwen is reserved for response wording inside flows.

    Returns:
      {"intents": [...], "confidence": float, "reasoning": str,
       "sub_account_id": str|None, "input_tokens": int, "output_tokens": int}
    On any failure, returns intents=["unknown"] with confidence 0.0 so the
    caller falls back safely.
    """
    from src.core.llm import call_intent_llm

    ctx_lines = []
    if authenticated:
        ctx_lines.append("Customer is already authenticated.")
    if active_flow:
        ctx_lines.append(f"Customer is currently in the '{active_flow}' flow.")
    ctx = ("\n".join(ctx_lines) + "\n\n") if ctx_lines else ""

    # Bedrock Converse message list (role/content-blocks), user-first.
    messages: list[dict[str, Any]] = []
    for h in (recent_history or [])[-4:]:
        role, content = h.get("role"), h.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": [{"text": content}]})
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    messages.append({"role": "user",
                     "content": [{"text": f"{ctx}Customer's latest message: {customer_message}"}]})

    try:
        result = call_intent_llm(_ROUTER_DECISION_SYSTEM, messages)  # Haiku
        obj = result.get("parsed")
        if not isinstance(obj, dict):
            raise ValueError(f"router decision not JSON: {result.get('text','')[:200]}")
        intents = obj.get("intents") or ([obj["intent"]] if obj.get("intent") else [])

        # ── Deterministic guard: statement requires a DOCUMENT keyword ────────
        # Haiku tends to add 'statement' for any "charges" wording (e.g. "see my
        # charges"), producing a spurious multi-intent brokerage+statement. Drop
        # 'statement' when the message has no document cue (statement/report/
        # email/send/download/document/slip/certificate) AND another serviceable
        # intent is present — the customer wants to VIEW data, not get a report.
        if "statement" in intents and len(intents) > 1:
            _doc_cues = ("statement", "report", "email", "e-mail", "send",
                         "download", "document", "slip", "certificate", "copy")
            if not any(c in customer_message.lower() for c in _doc_cues):
                intents = [i for i in intents if i != "statement"]
                logger.info("[LC] router guard: dropped spurious 'statement' "
                            "(no document keyword) → intents=%r", intents)

        logger.info("[LC] router_decision (haiku) intents=%r conf=%s", intents, obj.get("confidence"))
        return {
            "intents":        intents or ["unknown"],
            "confidence":     float(obj.get("confidence", 0.0) or 0.0),
            "complaint":      bool(obj.get("complaint", False)),
            "reasoning":      obj.get("reasoning", ""),
            "sub_account_id": obj.get("sub_account_id"),
            "input_tokens":   int(result.get("input_tokens", 0) or 0),
            "output_tokens":  int(result.get("output_tokens", 0) or 0),
        }
    except Exception as exc:
        logger.error("[LC] run_router_decision failed: %s — defaulting unknown", exc)
        return {"intents": ["unknown"], "confidence": 0.0, "complaint": False,
                "reasoning": "", "sub_account_id": None,
                "input_tokens": 0, "output_tokens": 0}


# ── Agentic API decision ──────────────────────────────────────────────────────
# The AGENT decides WHICH API tool to call (and its routing args) at a flow's
# data/dispatch step — instead of the flow hardcoding e.g. request_statement vs
# send_dp_bill vs get_ledger, or which endpoint/jobname a report maps to. Flow
# navigation + the customer-facing messages stay hardcoded. Bounded choice, so
# Haiku. The caller ALWAYS passes a deterministic fallback, so a bad/failed
# decision never breaks the flow.

_API_DECISION_SYSTEM = (
    "You are the API-routing decision-maker for the Axis Direct chatbot. Given "
    "the flow, what the customer asked for, and a list of CANDIDATE API options, "
    "choose the SINGLE best option to fulfil the request. You do NOT write any "
    "customer reply — you only pick the option.\n\n"
    "You are given candidates as a JSON list, each: "
    '{"id": "<opt id>", "when": "<when to use it>"}.\n'
    "Return ONLY JSON: {\"choice\": \"<the id of the best candidate>\", "
    "\"reasoning\": \"<one short sentence>\"}. "
    "If none clearly fits, choose the first candidate's id."
)


def run_api_decision(
    flow: str,
    request_summary: str,
    candidates: list[dict],
) -> dict[str, Any]:
    """
    Agent picks which API option to use for a flow's data/dispatch step.

    Args:
        flow:            e.g. "statement", "charges".
        request_summary: short description of what the customer wants + params
                         (e.g. "report=Tax Statement, range FY 2024-25").
        candidates:      [{"id": "...", "when": "..."}] — the allowed choices.

    Returns:
        {"choice": "<candidate id>", "reasoning": str, "input_tokens": int,
         "output_tokens": int}. On failure, choice = first candidate id (so the
        caller's deterministic default is used).
    """
    from src.core.llm import call_intent_llm
    import json as _json

    default_id = candidates[0]["id"] if candidates else ""
    if len(candidates) <= 1:
        # No real choice to make — skip the LLM call.
        return {"choice": default_id, "reasoning": "single candidate",
                "input_tokens": 0, "output_tokens": 0}

    user = (
        f"Flow: {flow}\n"
        f"Customer request: {request_summary}\n\n"
        f"Candidates:\n{_json.dumps(candidates, ensure_ascii=False)}"
    )
    messages = [{"role": "user", "content": [{"text": user}]}]
    try:
        result = call_intent_llm(_API_DECISION_SYSTEM, messages)  # Haiku
        obj = result.get("parsed") or {}
        choice = obj.get("choice")
        valid = {c["id"] for c in candidates}
        if choice not in valid:
            logger.warning("[LC] api_decision invalid choice %r — default %r", choice, default_id)
            choice = default_id
        logger.info("[LC] api_decision flow=%s choice=%s", flow, choice)
        return {
            "choice":        choice,
            "reasoning":     obj.get("reasoning", ""),
            "input_tokens":  int(result.get("input_tokens", 0) or 0),
            "output_tokens": int(result.get("output_tokens", 0) or 0),
        }
    except Exception as exc:
        logger.error("[LC] run_api_decision failed: %s — default %r", exc, default_id)
        return {"choice": default_id, "reasoning": "", "input_tokens": 0, "output_tokens": 0}


# ── Multi-intent response merge (Option C) ────────────────────────────────────
# When a turn covers multiple intents, each flow produces its OWN hardcoded
# first-step message. Instead of concatenating them with a "---" separator, ask
# Qwen to blend them into ONE cohesive reply. This changes WORDING only — the
# quick-reply buttons + flow_state still come from the primary flow (Qwen must
# not invent options or drop any information).

_MULTI_MERGE_SYSTEM = (
    "You are the Axis Direct web assistant. The customer asked about MULTIPLE "
    "things in one message. Below are the separate, already-correct replies our "
    "system produced for each topic. Combine them into ONE cohesive, warm, "
    "concise reply.\n\n"
    "STRICT RULES:\n"
    "- Preserve ALL information, questions, dates, and links from every section. "
    "Do not drop or shorten away any detail.\n"
    "- Do NOT invent new options, buttons, numbers, or facts. Only rephrase and "
    "connect what is given.\n"
    "- Do NOT list button labels in the text — the buttons are shown separately.\n"
    "- Each section is a SEPARATE topic. Keep each section's question DISTINCT "
    "and clearly labelled — do NOT merge two different questions into one. The "
    "customer must still be able to answer each topic separately.\n"
    "- ORDER MATTERS: keep the sections in the SAME order given. Put the "
    "informational/answer sections FIRST, and the FINAL section (which asks the "
    "customer to choose something) LAST — its question must be the very last "
    "line, because the on-screen buttons appear right below it and must clearly "
    "belong to THAT question.\n"
    "- Keep it to a short, readable message (a few sentences or short lines).\n"
    "- Plain text only. No markdown headings, no '---' separators.\n\n"
    "Return ONLY the combined reply text — no JSON, no preamble."
)


def run_multi_intent_merge(sections: list[dict]) -> str:
    """
    Blend multiple flow replies into one cohesive message (wording only).

    Args:
        sections: [{"intent": "statement", "message": "<flow reply>"}, ...]
                  in the order they should appear.

    Returns:
        A single combined message string. On any failure (or <2 sections),
        falls back to the plain "\n\n---\n\n" join so the turn never breaks.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    parts = [s.get("message", "") for s in sections if s.get("message")]
    if len(parts) < 2:
        return "\n\n---\n\n".join(parts)

    blocks = []
    for s in sections:
        if s.get("message"):
            label = str(s.get("intent", "")).replace("_", " ").title() or "Topic"
            blocks.append(f"[{label}]\n{s['message']}")
    user_content = (
        "Combine these replies into one cohesive message:\n\n" + "\n\n".join(blocks)
    )

    try:
        llm, _ = _get_llm()
        resp = llm.invoke([
            SystemMessage(content=_MULTI_MERGE_SYSTEM),
            HumanMessage(content=user_content),
        ])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        text = (text or "").strip()
        # Guard: if the model returned nothing usable, fall back to the join.
        if not text or len(text) < 10:
            logger.warning("[LC] multi-intent merge empty — using plain join")
            return "\n\n---\n\n".join(parts)
        logger.info("[LC] multi-intent merge OK (%d sections)", len(parts))
        return text
    except Exception as exc:
        logger.error("[LC] run_multi_intent_merge failed: %s — plain join", exc)
        return "\n\n---\n\n".join(parts)


def warmup() -> None:
    """
    Pre-initialise the ChatBedrockConverse client at startup so the first tool
    dispatch / agent turn doesn't pay the one-time client-init cost on a
    customer's request. run_tool("escalate_to_agent") makes NO external call.
    """
    try:
        import time as _t
        _t0 = _t.perf_counter()
        _get_llm()  # build the model + tool binding
        run_tool("escalate_to_agent", reason="__warmup__")  # no external call
        logger.info("[LC] agent warmup complete in %d ms",
                    int((_t.perf_counter() - _t0) * 1000))
    except Exception as exc:
        logger.warning("[LC] agent warmup failed: %s", exc)


# ── Short-lived profile cache ─────────────────────────────────────────────────
# Multiple flows can call get_profile(sub_id) within the SAME turn (e.g. parallel
# multi-intent: statement + brokerage both check account status at their start
# step). Cache the CustomerProfile briefly, keyed by sub_account_id, so the
# profile API is hit ONCE and reused — instead of a duplicate call per flow.
import threading as _threading
import time as _time_mod

_PROFILE_CACHE: dict[str, tuple[float, Any]] = {}
_PROFILE_CACHE_TTL = float(os.getenv("PROFILE_CACHE_TTL_SEC", "30"))
_PROFILE_CACHE_LOCK = _threading.Lock()


def get_profile(sub_account_id: str):
    """
    Cached wrapper: returns the profile from a short-lived per-account cache when
    fresh (so parallel/repeated calls within a turn share ONE API hit), else
    fetches once and caches. TTL via PROFILE_CACHE_TTL_SEC (default 30s).
    """
    key = str(sub_account_id or "")
    now = _time_mod.time()
    with _PROFILE_CACHE_LOCK:
        hit = _PROFILE_CACHE.get(key)
        if hit and (now - hit[0]) < _PROFILE_CACHE_TTL:
            logger.info("[LC] get_profile cache HIT sub=%s", key)
            return hit[1]
    profile = _fetch_profile_uncached(sub_account_id)
    with _PROFILE_CACHE_LOCK:
        _PROFILE_CACHE[key] = (now, profile)
    return profile


def invalidate_profile_cache(sub_account_id: str | None = None) -> None:
    """Drop cached profile(s) — call after a profile-changing action."""
    with _PROFILE_CACHE_LOCK:
        if sub_account_id is None:
            _PROFILE_CACHE.clear()
        else:
            _PROFILE_CACHE.pop(str(sub_account_id), None)


def _fetch_profile_uncached(sub_account_id: str):
    """
    Fetch the full customer profile via the tool layer and return it as a
    CustomerProfile object (same type the flows expect). Falls back to the direct
    typed gateway call if the tool path fails, so flows never break.
    """
    from src.gateways.customer_api import CustomerProfile, get_customer_profile as _direct
    import time as _t
    _t0 = _t.perf_counter()

    data = run_tool("get_customer_profile_full", sub_account_id=sub_account_id)
    logger.info("[TIMING] profile_fetch_ms=%d", int((_t.perf_counter() - _t0) * 1000))
    if data and not data.get("error"):
        try:
            return CustomerProfile(
                sub_account_id       = data.get("sub_account_id", sub_account_id),
                account_status       = data.get("account_status", "active"),
                name                 = data.get("name", ""),
                registered_email     = data.get("registered_email", ""),
                phone                = data.get("phone", ""),
                account_opening_date = data.get("account_opening_date", ""),
                portal_status        = data.get("portal_status", 0),
                deactivation_code    = data.get("deactivation_code", ""),
                deactivation_reason  = data.get("deactivation_reason", ""),
                demat_account_no     = data.get("demat_account_no", ""),
                dp_id                = data.get("dp_id", ""),
                trading_account_no   = data.get("trading_account_no", ""),
                # segmentsEnabled isn't always surfaced as a top-level field by
                # the tool layer — fall back to the raw profile so the order-
                # status segment-active gate has the data it needs.
                segments_enabled     = data.get("segments_enabled")
                                       or (data.get("raw", {}) or {}).get("segmentsEnabled")
                                       or {},
                raw                  = data.get("raw", {}) or {},
            )
        except Exception as exc:
            logger.warning("[LC] get_profile rebuild failed: %s — direct fallback", exc)
    return _direct(sub_account_id)
