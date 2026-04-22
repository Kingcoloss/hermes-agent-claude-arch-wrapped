#!/usr/bin/env python3
"""Claude Code Sub-Agent Tool — spawn Claude Code sub-agents via CLI.

Registered tools:
- ``claude_subagent``       — spawn a single Claude Code sub-agent
- ``claude_subagent_batch`` — spawn multiple in parallel

Uses ``claude --print`` for non-interactive, headless execution.
Supports custom agents, tool restriction, model selection, and
permission bypass for automated workflows.

Requirements:
- ``claude`` CLI installed and in PATH
- ANTHROPIC_API_KEY or valid Claude Code auth
"""

import json
import logging
import os
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300
MAX_BATCH_SIZE = 5
_BATCH_POLL_INTERVAL = 0.5


def _resolve_claude_command() -> str:
    """Resolve the ``claude`` CLI command path."""
    env_cmd = os.getenv("CLAUDE_CLI_PATH", "").strip()
    if env_cmd:
        return env_cmd
    # Try common install locations
    for candidate in ("claude", "/usr/local/bin/claude", os.path.expanduser("~/.local/bin/claude")):
        try:
            result = subprocess.run([candidate, "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return candidate
        except Exception:
            continue
    return "claude"


def _check_claude_available() -> bool:
    """Check if Claude Code CLI is available."""
    try:
        result = subprocess.run([_resolve_claude_command(), "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def _build_claude_args(
    goal: str,
    context: Optional[str] = None,
    agent_name: Optional[str] = None,
    custom_agent_prompt: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None,
    model: Optional[str] = None,
    permission_mode: str = "bypassPermissions",
    bare: bool = True,
    add_dirs: Optional[List[str]] = None,
    allowed_dirs: Optional[List[str]] = None,
) -> List[str]:
    """Build CLI argument list for ``claude --print``."""
    cmd = [_resolve_claude_command(), "--print"]

    if bare:
        cmd.append("--bare")

    # Permission mode for automation
    valid_modes = {"acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan"}
    if permission_mode in valid_modes:
        cmd.extend(["--permission-mode", permission_mode])

    # Model override
    if model:
        cmd.extend(["--model", model])

    # Agent selection
    if agent_name:
        cmd.extend(["--agent", agent_name])

    # Custom agents definition
    if custom_agent_prompt:
        agents_json = json.dumps({
            "custom": {
                "description": "Custom sub-agent",
                "prompt": custom_agent_prompt,
            }
        })
        cmd.extend(["--agents", agents_json])
        if not agent_name:
            cmd.extend(["--agent", "custom"])

    # Tool restrictions
    if allowed_tools:
        formatted = ", ".join(allowed_tools)
        cmd.extend(["--allowed-tools", formatted])

    # Additional directories
    if add_dirs:
        for d in add_dirs:
            cmd.extend(["--add-dir", d])

    # Build prompt
    prompt_parts = [f"TASK:\n{goal}"]
    if context:
        prompt_parts.append(f"\nCONTEXT:\n{context}")
    prompt_parts.append(
        "\nComplete this task thoroughly. Return a clear, concise summary of what you did, "
        "what you found, any files created or modified, and any issues encountered."
    )
    prompt = "\n".join(prompt_parts)

    cmd.append(prompt)
    return cmd


def _run_claude_subprocess(
    cmd: List[str],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    task_index: int = 0,
    task_count: int = 1,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """Run a Claude subprocess and capture output."""
    start_time = time.monotonic()
    prefix = f"[{task_index + 1}/{task_count}] " if task_count > 1 else ""

    if progress_callback:
        try:
            progress_callback("subagent.start", preview=cmd[-1] if cmd else "")
        except Exception as e:
            logger.debug("Progress callback start failed: %s", e)

    try:
        logger.debug("Running Claude subagent: %s", shlex.join(cmd[:10]) + " ...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "CLAUDE_CODE_SIMPLE": "1"} if "--bare" in cmd else os.environ,
        )
        duration = round(time.monotonic() - start_time, 2)

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # Claude --print writes everything to stdout
        # stderr usually contains progress/warnings
        output = stdout.strip()

        if result.returncode != 0:
            error_msg = stderr.strip() or output[:500] or f"Claude exited with code {result.returncode}"
            if progress_callback:
                try:
                    progress_callback("subagent.complete", preview=error_msg, status="failed")
                except Exception:
                    pass
            return {
                "status": "error",
                "error": error_msg,
                "returncode": result.returncode,
                "duration_seconds": duration,
                "stdout": stdout[:2000],
                "stderr": stderr[:2000],
            }

        if progress_callback:
            try:
                progress_callback(
                    "subagent.complete",
                    preview=output[:200] if output else "Completed",
                    status="completed",
                    duration_seconds=duration,
                )
            except Exception:
                pass

        return {
            "status": "completed",
            "output": output,
            "duration_seconds": duration,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired as exc:
        duration = round(time.monotonic() - start_time, 2)
        stdout = exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        if progress_callback:
            try:
                progress_callback("subagent.complete", preview=f"Timeout after {timeout}s", status="timeout")
            except Exception:
                pass
        return {
            "status": "timeout",
            "error": f"Claude subagent timed out after {timeout} seconds",
            "duration_seconds": duration,
            "stdout": stdout[:2000],
            "stderr": stderr[:2000],
        }

    except Exception as exc:
        duration = round(time.monotonic() - start_time, 2)
        if progress_callback:
            try:
                progress_callback("subagent.complete", preview=str(exc), status="error")
            except Exception:
                pass
        return {
            "status": "error",
            "error": str(exc),
            "duration_seconds": duration,
        }


# ---------------------------------------------------------------------------
# claude_subagent (single)
# ---------------------------------------------------------------------------


def claude_subagent(
    goal: Optional[str] = None,
    context: Optional[str] = None,
    agent_name: Optional[str] = None,
    custom_agent_prompt: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None,
    model: Optional[str] = None,
    permission_mode: str = "bypassPermissions",
    bare: bool = True,
    add_dirs: Optional[List[str]] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    parent_agent=None,
) -> str:
    """Spawn a single Claude Code sub-agent via CLI.

    Args:
        goal: The task description for the sub-agent.
        context: Additional background information.
        agent_name: Name of a pre-configured agent to use.
        custom_agent_prompt: Custom system prompt for a one-off agent.
        allowed_tools: Restrict which tools the sub-agent can use.
        model: Model override (e.g., 'claude-sonnet-4-6').
        permission_mode: Permission mode for the session.
        bare: Use minimal mode (skip hooks, auto-memory, etc.).
        add_dirs: Additional directories to allow access to.
        timeout: Max execution time in seconds.
        parent_agent: Parent agent instance (for progress callbacks).

    Returns:
        JSON result string.
    """
    if not goal or not goal.strip():
        return tool_error("'goal' is required and must not be empty", success=False)

    if not _check_claude_available():
        return tool_error(
            "Claude Code CLI not found. Install from https://claude.ai/code "
            "or set CLAUDE_CLI_PATH environment variable.",
            success=False,
        )

    cmd = _build_claude_args(
        goal=goal,
        context=context,
        agent_name=agent_name,
        custom_agent_prompt=custom_agent_prompt,
        allowed_tools=allowed_tools,
        model=model,
        permission_mode=permission_mode,
        bare=bare,
        add_dirs=add_dirs,
    )

    # Progress callback from parent
    progress_cb = getattr(parent_agent, "tool_progress_callback", None) if parent_agent else None

    result = _run_claude_subprocess(
        cmd=cmd,
        timeout=timeout,
        progress_callback=progress_cb,
    )

    if result.get("status") == "completed":
        return tool_result(
            success=True,
            output=result["output"],
            duration_seconds=result["duration_seconds"],
            returncode=result.get("returncode", 0),
        )
    else:
        return tool_error(
            result.get("error", "Unknown error"),
            success=False,
            status=result.get("status"),
            duration_seconds=result.get("duration_seconds", 0),
            stdout=result.get("stdout", "")[:1000],
            stderr=result.get("stderr", "")[:1000],
        )


# ---------------------------------------------------------------------------
# claude_subagent_batch (parallel)
# ---------------------------------------------------------------------------


def claude_subagent_batch(
    tasks: Optional[List[Dict[str, Any]]] = None,
    max_concurrent: int = 3,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    permission_mode: str = "bypassPermissions",
    bare: bool = True,
    model: Optional[str] = None,
    parent_agent=None,
) -> str:
    """Spawn multiple Claude Code sub-agents in parallel.

    Args:
        tasks: List of task dicts, each with keys: goal, context,
               agent_name, custom_agent_prompt, allowed_tools, add_dirs.
        max_concurrent: Max parallel sub-agents (default 3, max 5).
        timeout: Max execution time per sub-agent.
        permission_mode: Permission mode for all sub-agents.
        bare: Use minimal mode for all sub-agents.
        model: Model override for all sub-agents.
        parent_agent: Parent agent instance (for progress callbacks).

    Returns:
        JSON result string with array of results.
    """
    if not tasks or not isinstance(tasks, list):
        return tool_error("'tasks' is required and must be a non-empty list", success=False)

    if len(tasks) > MAX_BATCH_SIZE:
        return tool_error(
            f"Too many tasks: {len(tasks)} (max {MAX_BATCH_SIZE}). "
            "Split into multiple batch calls.",
            success=False,
        )

    if not _check_claude_available():
        return tool_error(
            "Claude Code CLI not found. Install from https://claude.ai/code "
            "or set CLAUDE_CLI_PATH environment variable.",
            success=False,
        )

    max_concurrent = max(1, min(max_concurrent, MAX_BATCH_SIZE))
    n_tasks = len(tasks)
    progress_cb = getattr(parent_agent, "tool_progress_callback", None) if parent_agent else None

    def _run_task(i: int, task: Dict[str, Any]) -> Dict[str, Any]:
        cmd = _build_claude_args(
            goal=task.get("goal", ""),
            context=task.get("context"),
            agent_name=task.get("agent_name"),
            custom_agent_prompt=task.get("custom_agent_prompt"),
            allowed_tools=task.get("allowed_tools"),
            model=task.get("model") or model,
            permission_mode=task.get("permission_mode") or permission_mode,
            bare=task.get("bare") if task.get("bare") is not None else bare,
            add_dirs=task.get("add_dirs"),
        )
        return _run_claude_subprocess(
            cmd=cmd,
            timeout=task.get("timeout") or timeout,
            task_index=i,
            task_count=n_tasks,
            progress_callback=progress_cb,
        )

    overall_start = time.monotonic()
    results: List[Dict[str, Any]] = [{}] * n_tasks

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(_run_task, i, t): i for i, t in enumerate(tasks)}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = {
                    "status": "error",
                    "error": str(exc),
                    "duration_seconds": round(time.monotonic() - overall_start, 2),
                }

    total_duration = round(time.monotonic() - overall_start, 2)
    completed = sum(1 for r in results if r.get("status") == "completed")
    failed = sum(1 for r in results if r.get("status") != "completed")

    return tool_result(
        success=failed == 0,
        total_tasks=n_tasks,
        completed=completed,
        failed=failed,
        total_duration_seconds=total_duration,
        results=results,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


CLAUDE_SUBAGENT_SCHEMA = {
    "name": "claude_subagent",
    "description": (
        "Spawn a Claude Code sub-agent to work on a task using the Claude CLI. "
        "The sub-agent runs in an isolated non-interactive session and returns a summary.\n\n"
        "WHEN TO USE:\n"
        "- Tasks requiring Claude Code's specific capabilities (file editing, code review, refactoring)\n"
        "- Work that benefits from Claude Code's built-in tools and agent framework\n"
        "- Delegating to a different model or agent personality\n\n"
        "REQUIREMENTS:\n"
        "- Claude Code CLI must be installed (`claude --version`)\n"
        "- ANTHROPIC_API_KEY or valid Claude Code auth\n\n"
        "NOTES:\n"
        "- Sub-agents run with --bare mode by default (no auto-memory, no CLAUDE.md discovery)\n"
        "- Use --permission-mode bypassPermissions for fully automated execution\n"
        "- Restrict tools with allowed_tools to limit scope"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "The task description. Be specific and self-contained.",
            },
            "context": {
                "type": "string",
                "description": "Additional background: file paths, constraints, prior errors.",
            },
            "agent_name": {
                "type": "string",
                "description": "Pre-configured agent name (e.g., 'reviewer'). Use claude_subagent_list_agents to discover.",
            },
            "custom_agent_prompt": {
                "type": "string",
                "description": "Custom system prompt for a one-off agent. Overrides agent_name.",
            },
            "allowed_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Restrict which tools the sub-agent can use. "
                    "Examples: ['Bash', 'Edit', 'Read'], ['Bash(git *)']"
                ),
            },
            "model": {
                "type": "string",
                "description": "Model override, e.g. 'claude-sonnet-4-6', 'claude-opus-4-7'.",
            },
            "permission_mode": {
                "type": "string",
                "enum": ["acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan"],
                "default": "bypassPermissions",
                "description": "Permission mode. Use 'bypassPermissions' for automated workflows.",
            },
            "bare": {
                "type": "boolean",
                "default": True,
                "description": "Use minimal mode (skip hooks, auto-memory, LSP, etc.).",
            },
            "add_dirs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional directories to grant the sub-agent access to.",
            },
            "timeout": {
                "type": "integer",
                "default": 300,
                "description": "Max execution time in seconds (default 300).",
            },
        },
        "required": ["goal"],
    },
}


CLAUDE_SUBAGENT_BATCH_SCHEMA = {
    "name": "claude_subagent_batch",
    "description": (
        "Spawn multiple Claude Code sub-agents in parallel. "
        "Each task runs in its own isolated session. "
        "Max 5 tasks per batch, max 3 concurrent by default."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "Task goal"},
                        "context": {"type": "string", "description": "Task context"},
                        "agent_name": {"type": "string", "description": "Agent name"},
                        "custom_agent_prompt": {"type": "string", "description": "Custom prompt"},
                        "allowed_tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Restricted tools for this task",
                        },
                        "model": {"type": "string", "description": "Model override"},
                        "add_dirs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Additional directories",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Task-specific timeout override",
                        },
                    },
                    "required": ["goal"],
                },
                "description": "Tasks to run in parallel (max 5)",
            },
            "max_concurrent": {
                "type": "integer",
                "default": 3,
                "description": "Max parallel sub-agents (1-5)",
            },
            "timeout": {
                "type": "integer",
                "default": 300,
                "description": "Default timeout per task in seconds",
            },
            "permission_mode": {
                "type": "string",
                "enum": ["acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan"],
                "default": "bypassPermissions",
                "description": "Permission mode for all tasks",
            },
            "bare": {
                "type": "boolean",
                "default": True,
                "description": "Use bare mode for all tasks",
            },
            "model": {
                "type": "string",
                "description": "Default model for all tasks",
            },
        },
        "required": ["tasks"],
    },
}


registry.register(
    name="claude_subagent",
    toolset="claude",
    schema=CLAUDE_SUBAGENT_SCHEMA,
    handler=lambda args, **kw: claude_subagent(
        goal=args.get("goal"),
        context=args.get("context"),
        agent_name=args.get("agent_name"),
        custom_agent_prompt=args.get("custom_agent_prompt"),
        allowed_tools=args.get("allowed_tools"),
        model=args.get("model"),
        permission_mode=args.get("permission_mode", "bypassPermissions"),
        bare=args.get("bare", True),
        add_dirs=args.get("add_dirs"),
        timeout=args.get("timeout", DEFAULT_TIMEOUT_SECONDS),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=_check_claude_available,
    description="Spawn a Claude Code sub-agent via CLI",
    emoji="🧠",
)

registry.register(
    name="claude_subagent_batch",
    toolset="claude",
    schema=CLAUDE_SUBAGENT_BATCH_SCHEMA,
    handler=lambda args, **kw: claude_subagent_batch(
        tasks=args.get("tasks"),
        max_concurrent=args.get("max_concurrent", 3),
        timeout=args.get("timeout", DEFAULT_TIMEOUT_SECONDS),
        permission_mode=args.get("permission_mode", "bypassPermissions"),
        bare=args.get("bare", True),
        model=args.get("model"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=_check_claude_available,
    description="Spawn multiple Claude Code sub-agents in parallel",
    emoji="🧠",
)
