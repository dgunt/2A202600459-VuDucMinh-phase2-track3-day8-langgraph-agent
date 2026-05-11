"""Nodes for the LangGraph workflow.

Each function is small, testable, and returns a partial state update. Nodes do not
mutate the input state in place.
"""

from __future__ import annotations

import os
import re
import string

from .state import AgentState, ApprovalDecision, Route, make_event

RISKY_KEYWORDS = {"refund", "delete", "send", "cancel", "remove", "revoke"}
TOOL_KEYWORDS = {"status", "order", "lookup", "check", "track", "find", "search"}
ERROR_KEYWORDS = {"timeout", "fail", "failure", "error", "crash", "unavailable"}
VAGUE_WORDS = {"it", "this", "that", "thing", "issue", "problem"}


def _normalized_words(query: str) -> list[str]:
    normalized = query.lower().translate(str.maketrans("", "", string.punctuation))
    return re.findall(r"\b\w+\b", normalized)


def _contains_phrase(query: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", query) is not None


def intake_node(state: AgentState) -> dict:
    """Normalize raw query into state fields."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using the lab's priority policy."""
    query = state.get("query", "")
    normalized_query = query.lower()
    clean_words = _normalized_words(query)
    word_set = set(clean_words)
    route = Route.SIMPLE
    risk_level = "low"

    if word_set & RISKY_KEYWORDS:
        route = Route.RISKY
        risk_level = "high"
    elif word_set & TOOL_KEYWORDS:
        route = Route.TOOL
    elif len(clean_words) < 5 and word_set & VAGUE_WORDS:
        route = Route.MISSING_INFO
    elif (word_set & ERROR_KEYWORDS) or _contains_phrase(normalized_query, "cannot recover"):
        route = Route.ERROR

    return {
        "route": route.value,
        "actual_route": route.value,
        "risk_level": risk_level,
        "events": [make_event("classify", "completed", f"route={route.value}")],
    }


def clarify_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating or calling a tool."""
    question = "Could you share the missing details so I can help with the right request?"
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "missing information requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for approval without executing it."""
    return {
        "proposed_action": "prepare requested risky support action; approval required",
        "requires_approval": True,
        "events": [make_event("risky_action", "pending_approval", "approval required")],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval step with optional LangGraph interrupt().

    Set LANGGRAPH_INTERRUPT=true to use real interrupt() for HITL demos. Default
    uses a mock decision so tests and CI run offline.
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt(
            {
                "proposed_action": state.get("proposed_action"),
                "risk_level": state.get("risk_level"),
            }
        )
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        decision = ApprovalDecision(approved=True, comment="mock approval for lab")

    return {
        "approved": decision.approved,
        "approval": decision.model_dump(),
        "events": [
            make_event(
                "approval",
                "hitl_approval",
                f"approved={decision.approved}",
                interrupt=True,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Call a deterministic mock tool."""
    route = state.get("actual_route") or state.get("route")
    attempt = int(state.get("attempts", state.get("attempt", 0)))
    max_attempts = int(state.get("max_attempts", 3))
    query = state.get("query", "")

    if route == Route.ERROR.value and attempt <= 1 and attempt < max_attempts:
        result = f"ERROR: transient failure attempt={attempt}"
    else:
        result = f"OK: mock tool processed route={route or Route.TOOL.value} query={query[:60]}"

    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", f"tool executed attempt={attempt}")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results, the done check that enables retry loops."""
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""
    normalized = latest.upper()

    if normalized.startswith("ERROR") or normalized.startswith("FAILURE"):
        return {
            "evaluation_result": "needs_retry",
            "events": [
                make_event("evaluate", "completed", "tool result indicates failure, retry needed")
            ],
        }

    return {
        "evaluation_result": "success",
        "events": [make_event("evaluate", "completed", "tool result satisfactory")],
    }


def retry_node(state: AgentState) -> dict:
    """Record one bounded retry attempt."""
    attempt = int(state.get("attempts", state.get("attempt", 0))) + 1
    max_attempts = int(state.get("max_attempts", 3))

    return {
        "attempt": attempt,
        "attempts": attempt,
        "retry_count": attempt,
        "events": [
            make_event(
                "retry",
                "completed",
                "retry attempt recorded",
                attempt=attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Log unresolvable failures for manual review."""
    attempt = int(state.get("attempts", state.get("attempt", 0)))
    message = f"Request escalated for manual review after {attempt} retry attempt(s)."
    return {
        "errors": [message],
        "final_answer": message,
        "events": [
            make_event("dead_letter", "completed", f"max retries exceeded, attempt={attempt}")
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response grounded in the route and mock tool output."""
    route = state.get("actual_route") or state.get("route") or Route.SIMPLE.value
    latest_tool_result = (state.get("tool_results") or [""])[-1]

    if route == Route.SIMPLE.value:
        answer = "Here are the steps I recommend for this support request."
    elif route == Route.RISKY.value:
        approval_note = " after approval" if state.get("approved") else ""
        answer = f"The requested risky action was processed{approval_note}: {latest_tool_result}"
    elif latest_tool_result:
        answer = f"I checked the relevant system and found: {latest_tool_result}"
    else:
        answer = "I completed the request using the available support workflow."

    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "answer generated")],
    }


def finalize_node(state: AgentState) -> dict:
    """Finalize the run and emit a final audit event."""
    final_answer = state.get("final_answer") or "Workflow finished without additional output."
    return {
        "final_answer": final_answer,
        "events": [make_event("finalize", "completed", "workflow finished")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Backward-compatible alias for older tests and imports."""
    return clarify_node(state)


def retry_or_fallback_node(state: AgentState) -> dict:
    """Backward-compatible alias for older tests and imports."""
    return retry_node(state)
