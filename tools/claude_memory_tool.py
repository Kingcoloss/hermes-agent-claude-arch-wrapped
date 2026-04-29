#!/usr/bin/env python3
"""Claude Code Memory Tools — self-registering wrappers around ClaudeMemoryManager.

Tools:
- ``claude_memory_list``          — list memory files
- ``claude_memory_read``          — read a memory file
- ``claude_memory_write``         — write/update a memory file
- ``claude_memory_delete``        — delete a memory file
- ``claude_memory_read_index``    — read MEMORY.md index
- ``claude_memory_read_claude_md`` — read CLAUDE.md
- ``claude_memory_write_claude_md`` — write CLAUDE.md
- ``claude_memory_sync_to_hermes``   — sync memories to Hermes
- ``claude_memory_sync_from_hermes`` — sync memories from Hermes
"""

import json
import os
from typing import Any, Dict, Optional

from tools.registry import registry, tool_error, tool_result
from agent.claude_memory_manager import (
    ClaudeMemoryManager,
    claude_memory_delete as _delete,
    claude_memory_list as _list,
    claude_memory_read as _read,
    claude_memory_read_claude_md as _read_claude_md,
    claude_memory_read_index as _read_index,
    claude_memory_sync_from_hermes as _sync_from,
    claude_memory_sync_to_hermes as _sync_to,
    claude_memory_write as _write,
    claude_memory_write_claude_md as _write_claude_md,
)


def _check_claude_memory_available() -> bool:
    """Check if Claude Code memory directory exists."""
    claude_projects = os.path.expanduser("~/.claude/projects")
    return os.path.isdir(claude_projects)


# ---------------------------------------------------------------------------
# claude_memory_list
# ---------------------------------------------------------------------------

registry.register(
    name="claude_memory_list",
    toolset="claude",
    schema={
        "name": "claude_memory_list",
        "description": "List all Claude Code memory files for the current or specified project.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_path": {
                    "type": "string",
                    "description": "Project path. Defaults to current working directory.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _list(args.get("project_path")),
    check_fn=_check_claude_memory_available,
    description="List Claude Code memory files",
    emoji="🧠",
)


# ---------------------------------------------------------------------------
# claude_memory_read
# ---------------------------------------------------------------------------

registry.register(
    name="claude_memory_read",
    toolset="claude",
    schema={
        "name": "claude_memory_read",
        "description": "Read a specific Claude Code memory file by name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Memory file name (with or without .md suffix).",
                },
                "project_path": {
                    "type": "string",
                    "description": "Project path. Defaults to current working directory.",
                },
            },
            "required": ["name"],
        },
    },
    handler=lambda args, **kw: _read(args.get("name"), args.get("project_path")),
    check_fn=_check_claude_memory_available,
    description="Read a Claude Code memory file",
    emoji="🧠",
)


# ---------------------------------------------------------------------------
# claude_memory_write
# ---------------------------------------------------------------------------

registry.register(
    name="claude_memory_write",
    toolset="claude",
    schema={
        "name": "claude_memory_write",
        "description": "Write or update a Claude Code memory file.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Memory name (without .md suffix). Used as filename.",
                },
                "body": {
                    "type": "string",
                    "description": "Markdown body content.",
                },
                "mem_type": {
                    "type": "string",
                    "enum": ["user", "feedback", "project", "reference"],
                    "default": "project",
                    "description": "Memory type.",
                },
                "description": {
                    "type": "string",
                    "default": "",
                    "description": "One-line description for MEMORY.md index.",
                },
                "project_path": {
                    "type": "string",
                    "description": "Project path. Defaults to current working directory.",
                },
                "origin_session_id": {
                    "type": "string",
                    "description": "Optional originating session ID.",
                },
            },
            "required": ["name", "body"],
        },
    },
    handler=lambda args, **kw: _write(
        name=args.get("name"),
        body=args.get("body"),
        mem_type=args.get("mem_type", "project"),
        description=args.get("description", ""),
        project_path=args.get("project_path"),
        origin_session_id=args.get("origin_session_id"),
    ),
    check_fn=_check_claude_memory_available,
    description="Write/update a Claude Code memory file",
    emoji="🧠",
)


# ---------------------------------------------------------------------------
# claude_memory_delete
# ---------------------------------------------------------------------------

registry.register(
    name="claude_memory_delete",
    toolset="claude",
    schema={
        "name": "claude_memory_delete",
        "description": "Delete a Claude Code memory file.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Memory file name (with or without .md suffix).",
                },
                "project_path": {
                    "type": "string",
                    "description": "Project path. Defaults to current working directory.",
                },
            },
            "required": ["name"],
        },
    },
    handler=lambda args, **kw: _delete(args.get("name"), args.get("project_path")),
    check_fn=_check_claude_memory_available,
    description="Delete a Claude Code memory file",
    emoji="🧠",
)


# ---------------------------------------------------------------------------
# claude_memory_read_index
# ---------------------------------------------------------------------------

registry.register(
    name="claude_memory_read_index",
    toolset="claude",
    schema={
        "name": "claude_memory_read_index",
        "description": "Read the MEMORY.md index for a project.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_path": {
                    "type": "string",
                    "description": "Project path. Defaults to current working directory.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _read_index(args.get("project_path")),
    check_fn=_check_claude_memory_available,
    description="Read MEMORY.md index",
    emoji="🧠",
)


# ---------------------------------------------------------------------------
# claude_memory_read_claude_md
# ---------------------------------------------------------------------------

registry.register(
    name="claude_memory_read_claude_md",
    toolset="claude",
    schema={
        "name": "claude_memory_read_claude_md",
        "description": "Read the project's CLAUDE.md file.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_path": {
                    "type": "string",
                    "description": "Project path. Defaults to current working directory.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _read_claude_md(args.get("project_path")),
    check_fn=lambda: True,
    description="Read project's CLAUDE.md",
    emoji="🧠",
)


# ---------------------------------------------------------------------------
# claude_memory_write_claude_md
# ---------------------------------------------------------------------------

registry.register(
    name="claude_memory_write_claude_md",
    toolset="claude",
    schema={
        "name": "claude_memory_write_claude_md",
        "description": "Write or overwrite the project's CLAUDE.md file.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Full CLAUDE.md content to write.",
                },
                "project_path": {
                    "type": "string",
                    "description": "Project path. Defaults to current working directory.",
                },
            },
            "required": ["content"],
        },
    },
    handler=lambda args, **kw: _write_claude_md(args.get("content"), args.get("project_path")),
    check_fn=lambda: True,
    description="Write project's CLAUDE.md",
    emoji="🧠",
)


# ---------------------------------------------------------------------------
# claude_memory_sync_to_hermes
# ---------------------------------------------------------------------------

registry.register(
    name="claude_memory_sync_to_hermes",
    toolset="claude",
    schema={
        "name": "claude_memory_sync_to_hermes",
        "description": "Sync Claude Code memories to Hermes memory store.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_path": {
                    "type": "string",
                    "description": "Project path. Defaults to current working directory.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Target Hermes session ID. Defaults to auto-generated.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _sync_to(args.get("project_path"), args.get("session_id")),
    check_fn=_check_claude_memory_available,
    description="Sync Claude memories to Hermes",
    emoji="🧠",
)


# ---------------------------------------------------------------------------
# claude_memory_sync_from_hermes
# ---------------------------------------------------------------------------

registry.register(
    name="claude_memory_sync_from_hermes",
    toolset="claude",
    schema={
        "name": "claude_memory_sync_from_hermes",
        "description": "Sync Hermes memories into Claude Code memory files.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_path": {
                    "type": "string",
                    "description": "Project path. Defaults to current working directory.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Filter by Hermes session ID. Sync all if omitted.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _sync_from(args.get("project_path"), args.get("session_id")),
    check_fn=lambda: True,
    description="Sync Hermes memories to Claude Code",
    emoji="🧠",
)
