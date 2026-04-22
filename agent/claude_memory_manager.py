#!/usr/bin/env python3
"""Claude Code Memory Manager — read/write/manage Claude Code's persistent memory.

Integrates with the `.claude/projects/<id>/memory/` file-based memory system
and optionally syncs with Hermes's own memory store.

Supported operations:
- Read / write / delete memory files (user, feedback, project, reference)
- Manage MEMORY.md index
- Read / update CLAUDE.md
- Sync between Claude Code memory and Hermes memory
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_MEMORY_TYPES = {"user", "feedback", "project", "reference"}
_DEFAULT_MEMORY_DIR = Path.home() / ".claude" / "projects"


def _resolve_project_id(project_path: Optional[str] = None) -> str:
    """Resolve Claude Code project ID from a path.

    Claude Code encodes paths by replacing / with - and removing leading /.
    """
    if not project_path:
        project_path = os.getcwd()
    abs_path = os.path.abspath(os.path.expanduser(project_path))
    # Strip leading /
    encoded = abs_path.lstrip("/").replace("/", "-")
    return encoded


def _resolve_memory_dir(project_path: Optional[str] = None) -> Path:
    """Resolve the memory directory for a project."""
    project_id = _resolve_project_id(project_path)
    return _DEFAULT_MEMORY_DIR / project_id / "memory"


def _resolve_claude_md_path(project_path: Optional[str] = None) -> Path:
    """Resolve the CLAUDE.md path for a project."""
    if not project_path:
        project_path = os.getcwd()
    return Path(project_path) / ".claude" / "CLAUDE.md"


def _parse_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Returns (frontmatter_dict, body_text).
    """
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    fm_text = parts[1].strip()
    body = parts[2].strip()

    fm: Dict[str, Any] = {}
    for line in fm_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            # Remove quotes
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            elif val.startswith("'") and val.endswith("'"):
                val = val[1:-1]
            fm[key] = val

    return fm, body


def _build_frontmatter(fm: Dict[str, Any]) -> str:
    """Build YAML frontmatter string from dict."""
    lines = ["---"]
    for key, val in fm.items():
        if isinstance(val, str):
            # Escape quotes
            if '"' in val:
                lines.append(f"{key}: '{val}'")
            else:
                lines.append(f'{key}: "{val}"')
        else:
            lines.append(f"{key}: {val}")
    lines.append("---")
    return "\n".join(lines)


def _build_memory_file(
    name: str,
    description: str,
    mem_type: str,
    body: str,
    origin_session_id: Optional[str] = None,
) -> str:
    """Build full memory file content with frontmatter."""
    fm = {
        "name": name,
        "description": description,
        "type": mem_type,
    }
    if origin_session_id:
        fm["originSessionId"] = origin_session_id
    return _build_frontmatter(fm) + "\n\n" + body.strip() + "\n"


# ============================================================================
# Public API
# ============================================================================


class ClaudeMemoryManager:
    """Manager for Claude Code's persistent memory files."""

    def __init__(self, project_path: Optional[str] = None):
        self.project_path = project_path or os.getcwd()
        self.memory_dir = _resolve_memory_dir(self.project_path)
        self.claude_md_path = _resolve_claude_md_path(self.project_path)
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create memory directory if needed."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Memory file CRUD
    # ------------------------------------------------------------------

    def list_memories(self) -> List[Dict[str, Any]]:
        """List all memory files with metadata."""
        results: List[Dict[str, Any]] = []
        if not self.memory_dir.exists():
            return results

        for f in sorted(self.memory_dir.iterdir()):
            if not f.is_file() or f.suffix != ".md" or f.name == "MEMORY.md":
                continue
            try:
                content = f.read_text(encoding="utf-8")
                fm, body = _parse_frontmatter(content)
                results.append({
                    "filename": f.name,
                    "name": fm.get("name", f.stem),
                    "description": fm.get("description", ""),
                    "type": fm.get("type", "unknown"),
                    "originSessionId": fm.get("originSessionId"),
                    "body_preview": body[:200] + "..." if len(body) > 200 else body,
                })
            except Exception as exc:
                logger.debug("Failed to parse memory file %s: %s", f, exc)
                results.append({
                    "filename": f.name,
                    "name": f.stem,
                    "description": "",
                    "type": "unknown",
                    "error": str(exc),
                })
        return results

    def read_memory(self, name: str) -> Optional[Dict[str, Any]]:
        """Read a memory file by name (with or without .md suffix)."""
        filename = name if name.endswith(".md") else f"{name}.md"
        filepath = self.memory_dir / filename
        if not filepath.exists():
            return None

        content = filepath.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(content)
        return {
            "filename": filename,
            "name": fm.get("name", filepath.stem),
            "description": fm.get("description", ""),
            "type": fm.get("type", "unknown"),
            "originSessionId": fm.get("originSessionId"),
            "body": body,
            "frontmatter": fm,
        }

    def write_memory(
        self,
        name: str,
        body: str,
        mem_type: str = "project",
        description: str = "",
        origin_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write or update a memory file.

        Args:
            name: Memory name (without .md suffix).
            body: Markdown body content.
            mem_type: One of user, feedback, project, reference.
            description: One-line description for the index.
            origin_session_id: Optional originating session ID.

        Returns:
            Dict with success status and file path.
        """
        if mem_type not in _MEMORY_TYPES:
            raise ValueError(f"Invalid memory type '{mem_type}'. Must be one of: {_MEMORY_TYPES}")

        filename = f"{name}.md"
        filepath = self.memory_dir / filename
        content = _build_memory_file(name, description, mem_type, body, origin_session_id)
        filepath.write_text(content, encoding="utf-8")

        # Auto-update MEMORY.md index
        self._update_memory_index(name, description)

        return {
            "success": True,
            "filename": filename,
            "path": str(filepath),
            "type": mem_type,
        }

    def delete_memory(self, name: str) -> Dict[str, Any]:
        """Delete a memory file."""
        filename = name if name.endswith(".md") else f"{name}.md"
        filepath = self.memory_dir / filename
        if not filepath.exists():
            return {"success": False, "error": f"Memory file not found: {filename}"}

        filepath.unlink()
        self._remove_from_index(name)
        return {"success": True, "filename": filename}

    # ------------------------------------------------------------------
    # MEMORY.md index
    # ------------------------------------------------------------------

    def read_memory_index(self) -> List[Dict[str, Any]]:
        """Read and parse MEMORY.md index."""
        index_path = self.memory_dir / "MEMORY.md"
        if not index_path.exists():
            return []

        content = index_path.read_text(encoding="utf-8")
        entries: List[Dict[str, Any]] = []
        for line in content.splitlines():
            line = line.strip()
            if not line.startswith("-"):
                continue
            # Parse: - [Title](file.md) — description
            match = re.match(r"- \[(.+?)\]\((.+?)\)\s*—\s*(.+)", line)
            if match:
                entries.append({
                    "title": match.group(1),
                    "file": match.group(2),
                    "description": match.group(3),
                })
        return entries

    def _update_memory_index(self, name: str, description: str) -> None:
        """Add or update an entry in MEMORY.md."""
        index_path = self.memory_dir / "MEMORY.md"
        entries = self.read_memory_index()

        # Update existing or append
        updated = False
        for entry in entries:
            if entry["file"] == f"{name}.md":
                entry["title"] = name
                entry["description"] = description
                updated = True
                break

        if not updated:
            entries.append({
                "title": name,
                "file": f"{name}.md",
                "description": description or "No description",
            })

        lines = [f"- [{e['title']}]({e['file']}) — {e['description']}" for e in entries]
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _remove_from_index(self, name: str) -> None:
        """Remove an entry from MEMORY.md."""
        index_path = self.memory_dir / "MEMORY.md"
        entries = self.read_memory_index()
        entries = [e for e in entries if e["file"] != f"{name}.md"]
        if entries:
            lines = [f"- [{e['title']}]({e['file']}) — {e['description']}" for e in entries]
            index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            if index_path.exists():
                index_path.unlink()

    # ------------------------------------------------------------------
    # CLAUDE.md
    # ------------------------------------------------------------------

    def read_claude_md(self) -> Optional[str]:
        """Read the project's CLAUDE.md if it exists."""
        if self.claude_md_path.exists():
            return self.claude_md_path.read_text(encoding="utf-8")
        return None

    def write_claude_md(self, content: str) -> Dict[str, Any]:
        """Write or overwrite CLAUDE.md."""
        self.claude_md_path.parent.mkdir(parents=True, exist_ok=True)
        self.claude_md_path.write_text(content, encoding="utf-8")
        return {"success": True, "path": str(self.claude_md_path)}

    def update_claude_md(self, append_content: str) -> Dict[str, Any]:
        """Append content to CLAUDE.md (creates if missing)."""
        existing = self.read_claude_md() or ""
        new_content = existing + "\n\n" + append_content.strip() + "\n"
        return self.write_claude_md(new_content)

    # ------------------------------------------------------------------
    # Sync with Hermes memory
    # ------------------------------------------------------------------

    def sync_to_hermes(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Sync Claude Code memories to Hermes memory store.

        Writes each memory as a JSONL entry in ~/.hermes/memories/.
        """
        from agent.memory_extractor import MemoryExtractor

        hermes_memory_dir = get_hermes_home() / "memories"
        hermes_memory_dir.mkdir(parents=True, exist_ok=True)

        memories = self.list_memories()
        synced = 0
        skipped = 0

        extractor = MemoryExtractor(memory_dir=hermes_memory_dir)
        target_session = session_id or f"claude-sync-{int(time.time())}"

        for mem in memories:
            filename = mem.get("filename", "")
            if filename == "MEMORY.md":
                continue

            full = self.read_memory(filename.replace(".md", ""))
            if not full:
                continue

            body = full.get("body", "")
            if not body.strip():
                skipped += 1
                continue

            # Treat the memory body as a "fact" for Hermes
            try:
                extractor.extract_and_store(
                    session_id=target_session,
                    user_message=f"Claude Code memory: {mem.get('name', '')}",
                    assistant_response=body,
                )
                synced += 1
            except Exception as exc:
                logger.debug("Failed to sync memory %s: %s", filename, exc)
                skipped += 1

        return {
            "success": True,
            "synced": synced,
            "skipped": skipped,
            "target_session": target_session,
        }

    def sync_from_hermes(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Sync Hermes memories into Claude Code memory files.

        Reads JSONL from ~/.hermes/memories/ and creates/updates
        Claude Code memory markdown files.
        """
        hermes_memory_dir = get_hermes_home() / "memories"
        if not hermes_memory_dir.exists():
            return {"success": False, "error": "Hermes memory directory not found", "synced": 0}

        synced = 0
        for f in sorted(hermes_memory_dir.iterdir()):
            if not f.suffix == ".jsonl":
                continue
            if session_id and not f.name.startswith(session_id):
                continue

            try:
                with open(f, "r", encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        fact = entry.get("fact", {})
                        if not fact:
                            continue

                        # Derive memory name from session_id + hash
                        sid = entry.get("session_id", "unknown")
                        fact_hash = entry.get("fact_hash", "")[:8]
                        mem_name = f"hermes-{sid}-{fact_hash}"

                        body = fact if isinstance(fact, str) else json.dumps(fact, ensure_ascii=False, indent=2)
                        self.write_memory(
                            name=mem_name,
                            body=body,
                            mem_type="project",
                            description=f"Synced from Hermes session {sid}",
                            origin_session_id=sid,
                        )
                        synced += 1
            except Exception as exc:
                logger.debug("Failed to sync from %s: %s", f, exc)

        return {"success": True, "synced": synced}


# ============================================================================
# Tool wrappers
# ============================================================================


def claude_memory_list(project_path: Optional[str] = None) -> str:
    """List all Claude Code memory files for a project."""
    try:
        manager = ClaudeMemoryManager(project_path)
        memories = manager.list_memories()
        return json.dumps({"success": True, "memories": memories, "count": len(memories)}, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


def claude_memory_read(name: str, project_path: Optional[str] = None) -> str:
    """Read a specific Claude Code memory file."""
    try:
        manager = ClaudeMemoryManager(project_path)
        mem = manager.read_memory(name)
        if not mem:
            return json.dumps({"success": False, "error": f"Memory '{name}' not found"}, ensure_ascii=False)
        return json.dumps({"success": True, "memory": mem}, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


def claude_memory_write(
    name: str,
    body: str,
    mem_type: str = "project",
    description: str = "",
    project_path: Optional[str] = None,
    origin_session_id: Optional[str] = None,
) -> str:
    """Write or update a Claude Code memory file."""
    try:
        manager = ClaudeMemoryManager(project_path)
        result = manager.write_memory(name, body, mem_type, description, origin_session_id)
        return json.dumps({"success": True, **result}, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


def claude_memory_delete(name: str, project_path: Optional[str] = None) -> str:
    """Delete a Claude Code memory file."""
    try:
        manager = ClaudeMemoryManager(project_path)
        result = manager.delete_memory(name)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


def claude_memory_read_index(project_path: Optional[str] = None) -> str:
    """Read the MEMORY.md index."""
    try:
        manager = ClaudeMemoryManager(project_path)
        entries = manager.read_memory_index()
        return json.dumps({"success": True, "entries": entries, "count": len(entries)}, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


def claude_memory_read_claude_md(project_path: Optional[str] = None) -> str:
    """Read the project's CLAUDE.md file."""
    try:
        manager = ClaudeMemoryManager(project_path)
        content = manager.read_claude_md()
        if content is None:
            return json.dumps({"success": False, "error": "CLAUDE.md not found"}, ensure_ascii=False)
        return json.dumps({"success": True, "content": content}, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


def claude_memory_write_claude_md(content: str, project_path: Optional[str] = None) -> str:
    """Write or overwrite the project's CLAUDE.md file."""
    try:
        manager = ClaudeMemoryManager(project_path)
        result = manager.write_claude_md(content)
        return json.dumps({"success": True, **result}, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


def claude_memory_sync_to_hermes(project_path: Optional[str] = None, session_id: Optional[str] = None) -> str:
    """Sync Claude Code memories to Hermes memory store."""
    try:
        manager = ClaudeMemoryManager(project_path)
        result = manager.sync_to_hermes(session_id)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


def claude_memory_sync_from_hermes(project_path: Optional[str] = None, session_id: Optional[str] = None) -> str:
    """Sync Hermes memories to Claude Code memory store."""
    try:
        manager = ClaudeMemoryManager(project_path)
        result = manager.sync_from_hermes(session_id)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)
