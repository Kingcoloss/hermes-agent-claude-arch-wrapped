"""Per-agent personal journal vault on disk.

Each agent gets a directory under ~/.hermes/vaults/agents/<agent_id>/
containing:
  - journal.md  — append-only session log
  - INDEX.md    — auto-rebuilt summary table

All paths resolved via ``get_hermes_home()`` — never hardcode ~/.hermes.
"""

import datetime
import re
from pathlib import Path

from hermes_constants import get_hermes_home

#: Valid agent_id format: lowercase alphanumeric, hyphens, underscores.
_AGENT_ID_RE = re.compile(r"^[a-z0-9_-]+$")


def _validate_agent_id(agent_id: str) -> None:
    """Raise ValueError if agent_id contains path-traversal characters or is invalid."""
    if not _AGENT_ID_RE.match(agent_id):
        raise ValueError(
            f"Invalid agent_id '{agent_id}': must match ^[a-z0-9_-]+$"
        )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_vault_dir(agent_id: str) -> Path:
    """Return ``~/.hermes/vaults/agents/<agent_id>/``, creating dirs if missing."""
    _validate_agent_id(agent_id)
    vault_dir = get_hermes_home() / "vaults" / "agents" / agent_id
    vault_dir.mkdir(parents=True, exist_ok=True)
    return vault_dir


# ---------------------------------------------------------------------------
# Journal entry formatting
# ---------------------------------------------------------------------------

def _format_kpi(kpi_summary: dict[str, float] | None) -> str:
    if not kpi_summary:
        return "—"
    parts = [f"{k}={v}" for k, v in kpi_summary.items()]
    return " ".join(parts) if parts else "—"


def _format_xp(
    xp_delta: float | None,
    level_before: int | None,
    level_after: int | None,
) -> str:
    if xp_delta is None:
        return "—"
    delta_str = f"+{int(xp_delta)}" if xp_delta == int(xp_delta) else f"+{xp_delta}"
    if level_before is not None and level_after is not None and level_before != level_after:
        return f"{delta_str} (level {level_before}→{level_after})"
    level = level_after if level_after is not None else level_before
    if level is not None:
        return f"{delta_str} (level {level})"
    return delta_str


def _format_actions(actions: list[str] | None) -> str:
    if not actions:
        return "—"
    return ", ".join(actions)


def _format_entry(
    timestamp: float,
    project_name: str | None,
    role: str,
    session_id: str,
    kpi_summary: dict[str, float] | None,
    xp_delta: float | None,
    level_before: int | None,
    level_after: int | None,
    actions: list[str] | None,
) -> str:
    dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    date_str = dt.strftime("%Y-%m-%d %H:%M")
    project_tag = f"[{project_name}]" if project_name else "[—]"
    kpi_str = _format_kpi(kpi_summary)
    xp_str = _format_xp(xp_delta, level_before, level_after)
    actions_str = _format_actions(actions)

    return (
        f"## {date_str} {project_tag} {role}\n"
        f"session: {session_id} | KPI: {kpi_str} | XP: {xp_str}\n"
        f"actions: {actions_str}\n"
    )


# ---------------------------------------------------------------------------
# INDEX.md generation
# ---------------------------------------------------------------------------

def _parse_journal(journal_text: str) -> list[dict]:
    """Parse journal.md into a list of entry dicts using simple string ops."""
    entries: list[dict] = []
    # Split on "## " section markers
    sections = journal_text.split("## ")
    for section in sections:
        section = section.strip()
        if not section:
            continue
        lines = section.split("\n")
        if not lines:
            continue
        header = lines[0]
        meta = lines[1] if len(lines) > 1 else ""
        # actions line is lines[2] if present

        # Parse header: "2026-04-30 14:32 [hermes-v2] fullstack-dev"
        parts = header.split(" ", 2)
        date_str = f"{parts[0]} {parts[1]}" if len(parts) >= 2 else "—"
        remainder = parts[2] if len(parts) >= 3 else ""
        # Extract [project] and role from remainder
        project = "—"
        role = "—"
        if remainder.startswith("["):
            close = remainder.find("]")
            if close != -1:
                project = remainder[1:close]
                role = remainder[close + 1:].strip()
            else:
                role = remainder
        else:
            role = remainder

        # Parse meta: "session: abc123 | KPI: tasks=3 ok=3 | XP: +15 (level 2→2)"
        session_id = "—"
        kpi = "—"
        xp = "—"
        if meta.startswith("session:"):
            segments = meta.split("|")
            for seg in segments:
                seg = seg.strip()
                if seg.startswith("session:"):
                    session_id = seg[len("session:"):].strip()
                elif seg.startswith("KPI:"):
                    kpi = seg[len("KPI:"):].strip()
                elif seg.startswith("XP:"):
                    xp = seg[len("XP:"):].strip()

        entries.append({
            "date": date_str,
            "project": project,
            "role": role,
            "kpi": kpi,
            "xp": xp,
        })
    return entries


def _build_index(agent_id: str, entries: list[dict]) -> str:
    """Build INDEX.md content from parsed entries."""
    lines = [f"# Journal Index — {agent_id}", ""]
    lines.append("| Date | Project | Role | KPI | XP |")
    lines.append("|---|---|---|---|---|")
    for entry in entries:
        lines.append(
            f"| {entry['date']} | {entry['project']} | {entry['role']} "
            f"| {entry['kpi']} | {entry['xp']} |"
        )
    lines.append("")
    lines.append(f"Total entries: {len(entries)}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def append_journal(
    agent_id: str,
    session_id: str,
    role: str,
    project_name: str | None,
    kpi_summary: dict[str, float] | None,
    xp_delta: float | None,
    level_before: int | None,
    level_after: int | None,
    actions: list[str] | None,
    timestamp: float | None = None,
) -> Path:
    """Append one entry to journal.md and rebuild INDEX.md.

    Returns the Path to journal.md.
    None for any optional field renders '—' or is skipped.
    """
    _validate_agent_id(agent_id)
    if timestamp is None:
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).timestamp()

    vault_dir = get_vault_dir(agent_id)
    journal_path = vault_dir / "journal.md"

    entry_text = _format_entry(
        timestamp=timestamp,
        project_name=project_name,
        role=role,
        session_id=session_id,
        kpi_summary=kpi_summary,
        xp_delta=xp_delta,
        level_before=level_before,
        level_after=level_after,
        actions=actions,
    )

    # Atomic append on POSIX when using 'a' mode
    with open(journal_path, "a", encoding="utf-8") as fh:
        fh.write("\n" + entry_text)

    rebuild_index(agent_id)
    return journal_path


def rebuild_index(agent_id: str) -> Path:
    """Re-scan journal.md, regenerate INDEX.md with one row per entry."""
    _validate_agent_id(agent_id)
    vault_dir = get_vault_dir(agent_id)
    index_path = vault_dir / "INDEX.md"

    journal_text = read_journal(agent_id)
    entries = _parse_journal(journal_text)
    index_content = _build_index(agent_id, entries)

    index_path.write_text(index_content, encoding="utf-8")
    return index_path


def read_journal(agent_id: str) -> str:
    """Return raw journal.md content, or ``''`` if missing."""
    _validate_agent_id(agent_id)
    vault_dir = get_vault_dir(agent_id)
    journal_path = vault_dir / "journal.md"
    if not journal_path.exists():
        return ""
    return journal_path.read_text(encoding="utf-8")


def list_agents_with_vaults() -> list[str]:
    """Scan ``~/.hermes/vaults/agents/`` and return sorted list of agent_ids.

    Directory names that do not match ``^[a-z0-9_-]+$`` are silently skipped.
    """
    vaults_root = get_hermes_home() / "vaults" / "agents"
    if not vaults_root.exists():
        return []
    return sorted(
        d.name
        for d in vaults_root.iterdir()
        if d.is_dir() and _AGENT_ID_RE.match(d.name)
    )