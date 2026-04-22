"""Role-based agent management for multimodal agent platform.

Roles define toolset presets, default models, skins, and KPI weightings
for specific personas (e.g. quant-trader, fullstack-dev, content-creator).
Roles are loaded from ``~/.hermes/roles/*.yaml`` and can be switched only
at session boundaries to preserve prompt-caching integrity.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from hermes_constants import get_hermes_home
from toolsets import get_toolset, resolve_toolset, validate_toolset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default role definitions (shipped in-repo, user roles override these)
# ---------------------------------------------------------------------------

DEFAULT_ROLES: Dict[str, Dict[str, Any]] = {
    "devops": {
        "description": "Infrastructure automation, CI/CD, container orchestration, and cloud operations.",
        "toolsets": ["devops"],
        "default_model": None,
        "skin": None,
        "kpi_weights": {
            "task_success_rate": 1.0,
            "avg_tokens_per_task": 0.5,
            "tool_diversity_score": 1.2,
            "error_recovery_rate": 1.5,
            "role_proficiency_score": 1.0,
        },
        "system_prompt_extra": (
            "You are a DevOps engineer. Prioritize infrastructure-as-code, automation, "
            "observability, and safe production practices. Always verify before applying "
            "changes to shared environments."
        ),
    },
    "quant-trader": {
        "description": "Statistical arbitrage, backtesting, options pricing, and portfolio optimization.",
        "toolsets": ["quant-trader"],
        "default_model": None,
        "skin": None,
        "kpi_weights": {
            "task_success_rate": 1.2,
            "avg_tokens_per_task": 0.3,
            "tool_diversity_score": 0.8,
            "error_recovery_rate": 1.0,
            "role_proficiency_score": 1.5,
        },
        "system_prompt_extra": (
            "You are a quantitative trader. Prioritize precision, statistical rigor, "
            "and reproducible analysis. Always validate assumptions and include confidence "
            "intervals where applicable."
        ),
    },
    "propfirm-trader": {
        "description": "High-frequency execution, risk management, trade journaling, and market analysis.",
        "toolsets": ["propfirm-trader"],
        "default_model": None,
        "skin": None,
        "kpi_weights": {
            "task_success_rate": 1.5,
            "avg_tokens_per_task": 0.3,
            "tool_diversity_score": 0.6,
            "error_recovery_rate": 1.2,
            "role_proficiency_score": 1.5,
        },
        "system_prompt_extra": (
            "You are a propfirm trader focused on capital preservation and consistent returns. "
            "Prioritize risk management, position sizing, and disciplined execution. "
            "Journal every trade with rationale and outcome."
        ),
    },
    "content-creator": {
        "description": "Writing, editing, media generation, social media, and SEO research.",
        "toolsets": ["content-creator"],
        "default_model": None,
        "skin": None,
        "kpi_weights": {
            "task_success_rate": 1.0,
            "avg_tokens_per_task": 1.0,
            "tool_diversity_score": 1.2,
            "error_recovery_rate": 0.8,
            "role_proficiency_score": 1.0,
        },
        "system_prompt_extra": (
            "You are a content creator. Prioritize engaging storytelling, audience-centric "
            "messaging, and platform-appropriate formatting. Use media tools (image generation, "
            "TTS) when they enhance the content."
        ),
    },
    "fullstack-dev": {
        "description": "Frontend, backend, database, API, testing, and deployment.",
        "toolsets": ["fullstack-dev"],
        "default_model": None,
        "skin": None,
        "kpi_weights": {
            "task_success_rate": 1.2,
            "avg_tokens_per_task": 0.8,
            "tool_diversity_score": 1.0,
            "error_recovery_rate": 1.2,
            "role_proficiency_score": 1.2,
        },
        "system_prompt_extra": (
            "You are a full-stack developer. Prioritize clean architecture, type safety, "
            "test coverage, and pragmatic trade-offs. Use delegation for isolated subsystems."
        ),
    },
    "system-engineer": {
        "description": "OS internals, networking, security, performance tuning, and troubleshooting.",
        "toolsets": ["system-engineer"],
        "default_model": None,
        "skin": None,
        "kpi_weights": {
            "task_success_rate": 1.0,
            "avg_tokens_per_task": 0.6,
            "tool_diversity_score": 1.0,
            "error_recovery_rate": 1.5,
            "role_proficiency_score": 1.2,
        },
        "system_prompt_extra": (
            "You are a system engineer. Prioritize deep understanding of stack layers, "
            "security-first thinking, and measurable performance improvements. Document "
            "assumptions and edge cases explicitly."
        ),
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoleProfile:
    """Immutable definition of an agent role."""

    name: str
    description: str
    toolsets: List[str] = field(default_factory=list)
    default_model: Optional[str] = None
    skin: Optional[str] = None
    kpi_weights: Dict[str, float] = field(default_factory=dict)
    system_prompt_extra: str = ""

    @property
    def resolved_tools(self) -> List[str]:
        """Return the full list of tool names for this role."""
        from toolsets import resolve_multiple_toolsets
        return resolve_multiple_toolsets(self.toolsets)


# ---------------------------------------------------------------------------
# Role loading
# ---------------------------------------------------------------------------

def _roles_dir() -> Path:
    return get_hermes_home() / "roles"


def _load_yaml_role(path: Path) -> Optional[RoleProfile]:
    """Load a single role from a YAML file."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse role file %s: %s", path, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Role file %s is not a dict", path)
        return None

    name = data.get("name") or path.stem
    return RoleProfile(
        name=name,
        description=data.get("description", ""),
        toolsets=list(data.get("toolsets", [])),
        default_model=data.get("default_model"),
        skin=data.get("skin"),
        kpi_weights=dict(data.get("kpi_weights", {})),
        system_prompt_extra=data.get("system_prompt_extra", ""),
    )


def _load_user_roles() -> Dict[str, RoleProfile]:
    """Scan ``~/.hermes/roles/*.yaml`` for user-defined roles."""
    roles_dir = _roles_dir()
    if not roles_dir.exists():
        return {}

    result: Dict[str, RoleProfile] = {}
    for path in sorted(roles_dir.glob("*.yaml")):
        role = _load_yaml_role(path)
        if role:
            result[role.name] = role
    return result


def _load_default_roles() -> Dict[str, RoleProfile]:
    """Convert built-in DEFAULT_ROLES dicts to RoleProfile objects."""
    result: Dict[str, RoleProfile] = {}
    for name, data in DEFAULT_ROLES.items():
        result[name] = RoleProfile(
            name=name,
            description=data.get("description", ""),
            toolsets=list(data.get("toolsets", [])),
            default_model=data.get("default_model"),
            skin=data.get("skin"),
            kpi_weights=dict(data.get("kpi_weights", {})),
            system_prompt_extra=data.get("system_prompt_extra", ""),
        )
    return result


# ---------------------------------------------------------------------------
# Role validation
# ---------------------------------------------------------------------------

def _validate_role(role: RoleProfile) -> List[str]:
    """Return a list of validation errors for *role*."""
    errors: List[str] = []
    for ts in role.toolsets:
        if not validate_toolset(ts):
            errors.append(f"Unknown toolset '{ts}'")
    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RoleManager:
    """Central registry for agent roles.

    User-defined roles (``~/.hermes/roles/*.yaml``) override built-in defaults.
    Roles are immutable once loaded; the manager caches them for the process
    lifetime.
    """

    def __init__(self) -> None:
        self._roles: Optional[Dict[str, RoleProfile]] = None

    def _ensure_loaded(self) -> None:
        if self._roles is not None:
            return
        roles = _load_default_roles()
        user_roles = _load_user_roles()
        # User roles override defaults
        roles.update(user_roles)
        # Validate every role (log warnings, don't crash)
        for name, role in list(roles.items()):
            errors = _validate_role(role)
            if errors:
                logger.warning("Role '%s' validation errors: %s", name, errors)
        self._roles = roles

    def list_roles(self) -> List[str]:
        """Return sorted list of available role names."""
        self._ensure_loaded()
        return sorted(self._roles.keys())

    def get_role(self, name: str) -> Optional[RoleProfile]:
        """Look up a role by name (case-sensitive)."""
        self._ensure_loaded()
        return self._roles.get(name)

    def reload(self) -> None:
        """Force reload from disk (useful after role file edits)."""
        self._roles = None
        self._ensure_loaded()

    def build_role_system_prompt(self, name: str) -> str:
        """Build a system-prompt snippet for *name*.

        Returns an empty string when the role doesn't exist or has no
        ``system_prompt_extra``.
        """
        role = self.get_role(name)
        if not role:
            return ""
        lines: List[str] = []
        lines.append(f"# Role: {role.name}")
        if role.description:
            lines.append(role.description)
        if role.system_prompt_extra:
            lines.append(role.system_prompt_extra)
        if role.resolved_tools:
            lines.append(f"Available tools ({len(role.resolved_tools)}): {', '.join(role.resolved_tools)}")
        return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_role_manager: Optional[RoleManager] = None


def get_role_manager() -> RoleManager:
    """Return the process-global RoleManager singleton."""
    global _role_manager
    if _role_manager is None:
        _role_manager = RoleManager()
    return _role_manager
