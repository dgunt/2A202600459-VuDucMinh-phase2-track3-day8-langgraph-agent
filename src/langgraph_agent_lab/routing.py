"""Routing functions for conditional edges."""

from __future__ import annotations

from .state import AgentState, Route


def route_after_classify(state: AgentState) -> str:
    """Map classified route to the next graph node."""
    route = state.get("actual_route") or state.get("route", Route.SIMPLE.value)
    mapping = {
        Route.SIMPLE.value: "answer",
        Route.TOOL.value: "tool",
        Route.MISSING_INFO.value: "clarify",
        Route.RISKY.value: "risky_action",
        Route.ERROR.value: "retry",
    }
    return mapping.get(route, "clarify")


def route_after_evaluate(state: AgentState) -> str:
    """Route failed tool results back to retry, otherwise answer."""
    if state.get("evaluation_result") == "needs_retry":
        return "retry"
    return "answer"


def route_after_retry(state: AgentState) -> str:
    """Route to the tool while retry budget remains, else dead-letter."""
    attempts = int(state.get("attempts", state.get("attempt", 0)))
    max_attempts = int(state.get("max_attempts", 3))
    return "tool" if attempts < max_attempts else "dead_letter"


def route_after_approval(state: AgentState) -> str:
    """Continue only if approved."""
    approval = state.get("approval") or {}
    return "tool" if approval.get("approved") else "clarify"
