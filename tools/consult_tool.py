#!/usr/bin/env python3
"""Peer Consultation Tools — consult_agent, consult_panel, consult_team.

Registered tools:
- ``consult_agent``  — 1-on-1 peer consultation with a specific agent
- ``consult_panel``  — parallel consultation with an explicit list of agents
- ``consult_team``   — fan-out consultation to all active members of a team

All three tools persist consultation rows via ConsultationManager, enforce
depth / cycle / cost-cap guards, and invoke a Claude Code sub-agent to
produce the actual response.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

_MAX_PANEL_CONCURRENT = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_manager():
    """Lazy import to avoid circular imports at module load time."""
    from agent.consultation_manager import get_consultation_manager
    return get_consultation_manager()


def _get_agent_manager():
    from agent.agent_manager import get_agent_manager
    return get_agent_manager()


def _get_team_manager():
    from agent.team_manager import get_team_manager
    return get_team_manager()


def _invoke_agent(target_agent_id: str, question: str, context_summary: str) -> str:
    """Call a Claude sub-agent representing *target_agent_id*.

    Returns the text response string, or raises on failure.
    """
    from tools.claude_subagent_tool import claude_subagent

    prompt = (
        f"You are agent '{target_agent_id}'. "
        f"Question: {question}\n\nContext: {context_summary}"
    )
    raw = claude_subagent(goal=prompt)

    # claude_subagent returns a JSON string; extract the 'output' field.
    try:
        data = json.loads(raw)
        return data.get("output") or data.get("error") or str(data)
    except (json.JSONDecodeError, TypeError):
        return str(raw)


def _run_single_consult(
    *,
    caller_session_id: str,
    target_agent_id: str,
    question: str,
    context_summary: str,
    caller_agent_id: Optional[str] = None,
    parent_consultation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create + execute one consultation. Returns a result dict (never raises)."""
    manager = _get_manager()

    try:
        consultation = manager.create_consultation(
            caller_session_id=caller_session_id,
            target_agent_id=target_agent_id,
            question=question,
            context_summary=context_summary,
            caller_agent_id=caller_agent_id,
            parent_consultation_id=parent_consultation_id,
        )
    except ValueError as exc:
        # Guard failed before we even created the row.
        return {
            "agent_id": target_agent_id,
            "consultation_id": None,
            "response": None,
            "status": "failed",
            "error": str(exc),
        }

    consultation_id = consultation["id"]

    try:
        response = _invoke_agent(target_agent_id, question, context_summary)
        updated = manager.complete_consultation(consultation_id, response)
        return {
            "agent_id": target_agent_id,
            "consultation_id": consultation_id,
            "response": response,
            "status": updated["status"],
            "depth": updated["depth"],
        }
    except Exception as exc:
        logger.exception("consult_agent: sub-agent invocation failed for %s", target_agent_id)
        try:
            manager.fail_consultation(consultation_id, str(exc))
        except Exception:
            pass
        return {
            "agent_id": target_agent_id,
            "consultation_id": consultation_id,
            "response": None,
            "status": "failed",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def consult_agent_handler(
    target_agent_id: str,
    question: str,
    context_summary: str,
    *,
    session_id: Optional[str] = None,
    caller_agent_id: Optional[str] = None,
    **kwargs,
) -> str:
    """1-on-1 peer consultation.

    Args:
        target_agent_id: The agent to consult.
        question: The question to ask.
        context_summary: LLM-summarised context (not raw history).
        session_id: Injected by the tool dispatch layer (kwarg).
        caller_agent_id: Injected by the tool dispatch layer (kwarg).

    Returns:
        JSON tool result.
    """
    # Validate target exists before creating the consultation record.
    agent_manager = _get_agent_manager()
    agent = agent_manager.get_agent(target_agent_id)
    if agent is None:
        return tool_error(f"Target agent '{target_agent_id}' not found", success=False)
    if agent.get("status") != "active":
        return tool_error(
            f"Target agent '{target_agent_id}' is not active "
            f"(status='{agent.get('status')}')",
            success=False,
        )

    caller_session_id = session_id or "default-session"

    result = _run_single_consult(
        caller_session_id=caller_session_id,
        target_agent_id=target_agent_id,
        question=question,
        context_summary=context_summary,
        caller_agent_id=caller_agent_id,
    )

    if result.get("status") == "failed":
        return tool_error(
            result.get("error", "Consultation failed"),
            consultation_id=result.get("consultation_id"),
            success=False,
        )

    return tool_result(
        consultation_id=result["consultation_id"],
        agent_id=target_agent_id,
        response=result["response"],
        depth=result.get("depth", 0),
        status="done",
    )


def consult_panel_handler(
    target_agent_ids: List[str],
    question: str,
    context_summary: str,
    *,
    session_id: Optional[str] = None,
    caller_agent_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Parallel consultation with an explicit list of agents.

    Args:
        target_agent_ids: Agents to consult (list of agent id strings).
        question: The question to ask each agent.
        context_summary: LLM-summarised context.
        session_id: Injected by the tool dispatch layer.
        caller_agent_id: Injected by the tool dispatch layer.

    Returns:
        JSON tool result with ``consultations`` list.
    """
    if not target_agent_ids or not isinstance(target_agent_ids, list):
        return tool_error("'target_agent_ids' must be a non-empty list", success=False)

    caller_session_id = session_id or "default-session"

    def _run(agent_id: str) -> Dict[str, Any]:
        return _run_single_consult(
            caller_session_id=caller_session_id,
            target_agent_id=agent_id,
            question=question,
            context_summary=context_summary,
            caller_agent_id=caller_agent_id,
            parent_consultation_id=None,
        )

    results: List[Dict[str, Any]] = [{}] * len(target_agent_ids)

    with ThreadPoolExecutor(max_workers=_MAX_PANEL_CONCURRENT) as executor:
        futures = {executor.submit(_run, aid): i for i, aid in enumerate(target_agent_ids)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = {
                    "agent_id": target_agent_ids[idx],
                    "consultation_id": None,
                    "response": None,
                    "status": "failed",
                    "error": str(exc),
                }

    return tool_result(consultations=results)


def consult_team_handler(
    team_id: str,
    question: str,
    context_summary: str,
    include_lead: bool = True,
    aggregate: bool = False,
    *,
    session_id: Optional[str] = None,
    caller_agent_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Fan-out consultation to all active members of a team.

    Args:
        team_id: The team to consult.
        question: The question to ask each member.
        context_summary: LLM-summarised context.
        include_lead: If True, include the team lead in the fan-out (default True).
        aggregate: Reserved for M2; ignored in M1.5 — always returns list.
        session_id: Injected by the tool dispatch layer.
        caller_agent_id: Injected by the tool dispatch layer.

    Returns:
        JSON tool result with ``team_id`` and ``consultations`` list.
    """
    team_manager = _get_team_manager()

    # Verify team exists.
    team = team_manager.get_team(team_id)
    if team is None:
        return tool_error(f"Team '{team_id}' not found", success=False)

    if include_lead:
        members = team_manager.get_members(team_id, status="active")
    else:
        members = team_manager.get_members_excluding_lead(team_id)

    if not members:
        return tool_result(
            team_id=team_id,
            consultations=[],
            message="No active members to consult.",
        )

    caller_session_id = session_id or "default-session"
    target_agent_ids = [m["agent_id"] for m in members]

    def _run(agent_id: str) -> Dict[str, Any]:
        return _run_single_consult(
            caller_session_id=caller_session_id,
            target_agent_id=agent_id,
            question=question,
            context_summary=context_summary,
            caller_agent_id=caller_agent_id,
            parent_consultation_id=None,
        )

    results: List[Dict[str, Any]] = [{}] * len(target_agent_ids)

    with ThreadPoolExecutor(max_workers=_MAX_PANEL_CONCURRENT) as executor:
        futures = {executor.submit(_run, aid): i for i, aid in enumerate(target_agent_ids)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = {
                    "agent_id": target_agent_ids[idx],
                    "consultation_id": None,
                    "response": None,
                    "status": "failed",
                    "error": str(exc),
                }

    return tool_result(team_id=team_id, consultations=results)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

CONSULT_AGENT_SCHEMA = {
    "name": "consult_agent",
    "description": (
        "Ask a specific agent for their perspective on a question. "
        "Records the consultation chain for auditing. "
        "Enforces depth limit (max 2), cycle detection, and per-session cost cap (max 5)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target_agent_id": {
                "type": "string",
                "description": "ID of the agent to consult (e.g. 'alice', 'dev-1').",
            },
            "question": {
                "type": "string",
                "description": "The question to ask the target agent.",
            },
            "context_summary": {
                "type": "string",
                "description": (
                    "LLM-summarised context for the target agent. "
                    "Never pass raw conversation history — summarise it first."
                ),
            },
        },
        "required": ["target_agent_id", "question", "context_summary"],
    },
}

CONSULT_PANEL_SCHEMA = {
    "name": "consult_panel",
    "description": (
        "Consult multiple agents in parallel and collect all responses. "
        "All consultations run concurrently (max 3 at a time)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target_agent_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of agent IDs to consult.",
            },
            "question": {
                "type": "string",
                "description": "The question to ask each agent.",
            },
            "context_summary": {
                "type": "string",
                "description": "LLM-summarised context (not raw history).",
            },
        },
        "required": ["target_agent_ids", "question", "context_summary"],
    },
}

CONSULT_TEAM_SCHEMA = {
    "name": "consult_team",
    "description": (
        "Fan out a question to all active members of a team in parallel. "
        "Use include_lead=false to exclude the team lead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "team_id": {
                "type": "string",
                "description": "ID of the team to consult.",
            },
            "question": {
                "type": "string",
                "description": "The question to ask team members.",
            },
            "context_summary": {
                "type": "string",
                "description": "LLM-summarised context (not raw history).",
            },
            "include_lead": {
                "type": "boolean",
                "default": True,
                "description": "Include the team lead in the fan-out (default true).",
            },
            "aggregate": {
                "type": "boolean",
                "default": False,
                "description": "Reserved for M2 — aggregation is not yet implemented.",
            },
        },
        "required": ["team_id", "question", "context_summary"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

registry.register(
    name="consult_agent",
    toolset="consult",
    schema=CONSULT_AGENT_SCHEMA,
    handler=lambda args, **kw: consult_agent_handler(
        target_agent_id=args["target_agent_id"],
        question=args["question"],
        context_summary=args["context_summary"],
        session_id=kw.get("session_id"),
        caller_agent_id=kw.get("caller_agent_id"),
    ),
    description="1-on-1 peer consultation with a specific agent",
)

registry.register(
    name="consult_panel",
    toolset="consult",
    schema=CONSULT_PANEL_SCHEMA,
    handler=lambda args, **kw: consult_panel_handler(
        target_agent_ids=args["target_agent_ids"],
        question=args["question"],
        context_summary=args["context_summary"],
        session_id=kw.get("session_id"),
        caller_agent_id=kw.get("caller_agent_id"),
    ),
    description="Parallel consultation with an explicit list of agents",
)

registry.register(
    name="consult_team",
    toolset="consult",
    schema=CONSULT_TEAM_SCHEMA,
    handler=lambda args, **kw: consult_team_handler(
        team_id=args["team_id"],
        question=args["question"],
        context_summary=args["context_summary"],
        include_lead=args.get("include_lead", True),
        aggregate=args.get("aggregate", False),
        session_id=kw.get("session_id"),
        caller_agent_id=kw.get("caller_agent_id"),
    ),
    description="Fan-out consultation to all active members of a team",
)
