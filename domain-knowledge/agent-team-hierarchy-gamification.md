---
scope: Agent Team Hierarchy + RPG Gamification System for Hermes Agent
audience: Implementation agent (next session)
verified_via: codebase analysis (agent/role_manager.py, agent/gamification.py, hermes_state.py, hermes_cli/commands.py, tools/delegate_tool.py) + integration tests
generated: 2026-04-30
status: planning-complete
---

# Agent Team Hierarchy + RPG Gamification System

## 1. Executive Summary

Design and implement a **Deep Hierarchy Agent Team** system where a Team Lead (orchestrator) can spawn Sub-Leads and Workers in a multi-level tree structure. Each agent node has a **Role** (from `RoleManager`) and participates in an **RPG-style Gamification** system with Team XP Pool, Individual XP/Levels, Achievements, and Skill Trees that unlock new capabilities on level-up.

**Key concept:** Teams are persistent organizational structures in the database. When a user runs `/team create`, a team is defined once and can be re-used across sessions. Agents spawned via `delegate_task` inherit their team context, role prompt, and toolsets automatically.

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                        TEAM                              │
│                   Team Level: 5                          │
│                   Team XP: 450                            │
│                   Achievements: ["First Deploy", "10x"]  │
│                                                          │
│   Team Lead (orchestrator, role: fullstack-dev, Lv.7)   │
│   ├── Dev Sub-Lead (orchestrator, role: fullstack-dev)  │
│   │   ├── Frontend Worker (leaf, role: fullstack-dev)   │
│   │   └── Backend Worker (leaf, role: system-engineer)  │
│   └── Ops Sub-Lead (orchestrator, role: devops, Lv.6)  │
│       └── SRE Worker (leaf, role: devops, Lv.3)         │
│                                                          │
│   XP flows UP: Workers → Sub-Leads → Lead → Team       │
└──────────────────────────────────────────────────────────┘
```

### 2.1 Hierarchy Rules

- **Max depth:** Controlled by `delegation.max_spawn_depth` config (default 2)
- **Root:** Always a Team Lead (orchestrator role)
- **Intermediate nodes:** Sub-Leads (orchestrator role, can delegate further)
- **Leaf nodes:** Workers (leaf role, cannot delegate)
- **parent_id:** Each member references its parent in the tree (NULL for Team Lead)

## 3. Data Model

### 3.1 New Tables (added to hermes_state.py as schema v12)

```sql
CREATE TABLE IF NOT EXISTS teams (
    id TEXT PRIMARY KEY,                -- UUID
    name TEXT NOT NULL UNIQUE,          -- Human-readable team name
    description TEXT,                   -- Optional description
    team_xp REAL DEFAULT 0,            -- Aggregated team XP pool
    team_level INTEGER DEFAULT 1,       -- Team level (floor(team_xp / 100) + 1)
    lead_role TEXT,                     -- Role of the team lead
    created_at REAL NOT NULL,           -- Unix timestamp
    updated_at REAL NOT NULL            -- Unix timestamp
);

CREATE TABLE IF NOT EXISTS team_members (
    id TEXT PRIMARY KEY,                -- UUID
    team_id TEXT NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL UNIQUE,       -- Unique agent identifier (e.g., "dev-lead-1")
    role TEXT NOT NULL,                 -- Role name from RoleManager
    parent_member_id TEXT,              -- Self-ref: parent in hierarchy (NULL = team lead)
    member_level INTEGER DEFAULT 1,    -- Individual level
    member_xp REAL DEFAULT 0,         -- Individual XP
    is_lead INTEGER DEFAULT 0,         -- 1 = orchestrator, 0 = leaf
    position TEXT,                      -- Human-readable position label ("Frontend Worker")
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (team_id) REFERENCES teams(id),
    FOREIGN KEY (parent_member_id) REFERENCES team_members(id)
);

CREATE TABLE IF NOT EXISTS skill_trees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,                  -- Which role this skill belongs to
    skill_name TEXT NOT NULL,            -- e.g., "CSS Master", "CI/CD Expert"
    skill_type TEXT NOT NULL DEFAULT 'passive',
                                        -- 'passive' = always-on bonus
                                        -- 'active' = unlocks a new tool/toolset
                                        -- 'prompt' = adds system_prompt_extra snippet
    level_threshold INTEGER NOT NULL,    -- Role level required to unlock
    description TEXT,                   -- What this skill does
    unlocked_content TEXT,              -- JSON: {"toolset": "web"} or {"prompt": "..."} or {"bonus": {"kpi_weights": {...}}}
    unlocked INTEGER DEFAULT 0,         -- 0 = locked, 1 = unlocked
    unlocked_at REAL,                   -- When it was unlocked
    UNIQUE(role, skill_name)
);

CREATE TABLE IF NOT EXISTS team_achievements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id TEXT NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    achievement_id TEXT NOT NULL,        -- Unique within team
    name TEXT NOT NULL,
    description TEXT,
    unlocked_at REAL NOT NULL,
    UNIQUE(team_id, achievement_id)
);

CREATE INDEX IF NOT EXISTS idx_teams_name ON teams(name);
CREATE INDEX IF NOT EXISTS idx_team_members_team ON team_members(team_id);
CREATE INDEX IF NOT EXISTS idx_team_members_parent ON team_members(parent_member_id);
CREATE INDEX IF NOT EXISTS idx_team_members_role ON team_members(role);
CREATE INDEX IF NOT EXISTS idx_skill_trees_role ON skill_trees(role);
CREATE INDEX IF NOT EXISTS idx_team_achievements_team ON team_achievements(team_id);
```

### 3.2 Existing Tables (no changes needed)

| Table | Usage |
|-------|-------|
| `agent_skills_xp` | Individual role XP/levels (existing, reused as-is) |
| `agent_kpi` | Per-session metrics (existing, now also aggregated into team XP) |
| `agent_achievements` | Individual achievements (existing, kept separate from team achievements) |

### 3.3 Data Flow: XP Aggregation

```
Task completes
  └─→ KPITracker.record_session_metrics(session_id, role, metrics)
        └─→ calculate XP delta from metrics * role.kpi_weights
              ├─→ KPITracker.add_xp(role, delta)              # Individual XP
              └─→ TeamManager.add_team_xp(team_id, delta)     # Team XP pool
                    └─→ check_skill_unlocks(role, new_level)   # Skill Tree
```

## 4. New Modules

### 4.1 `agent/team_manager.py`

```python
class TeamManager:
    """CRUD and hierarchy operations for agent teams."""
    
    def __init__(self, db: SessionDB = None): ...
    
    # ── Team CRUD ──
    def create_team(self, name: str, description: str = None, lead_role: str = None) -> Dict: ...
    def get_team(self, team_id: str) -> Optional[Dict]: ...
    def get_team_by_name(self, name: str) -> Optional[Dict]: ...
    def list_teams(self) -> List[Dict]: ...
    def disband_team(self, team_id: str) -> bool: ...
    
    # ── Member management ──
    def add_member(self, team_id: str, agent_id: str, role: str, 
                   parent_member_id: str = None, is_lead: bool = False,
                   position: str = None) -> Dict: ...
    def remove_member(self, team_id: str, agent_id: str) -> bool: ...
    def get_member(self, agent_id: str) -> Optional[Dict]: ...
    def get_team_members(self, team_id: str) -> List[Dict]: ...
    def get_children(self, member_id: str) -> List[Dict]: ...
    
    # ── Hierarchy ──
    def get_lineage(self, agent_id: str) -> List[Dict]:
        """Return the full chain from root lead down to this agent."""
        ...
    def get_subtree(self, member_id: str) -> List[Dict]:
        """Return all descendants of a member (BFS)."""
        ...
    def get_depth(self, agent_id: str) -> int:
        """Return hierarchy depth (0 = team lead)."""
        ...
    
    # ── Team XP ──
    def add_team_xp(self, team_id: str, xp_delta: float) -> Dict:
        """Add XP to team pool. Returns {old_xp, new_xp, old_level, new_level, leveled_up}."""
        ...
    def get_team_level(self, team_id: str) -> Dict:
        """Return {team_id, level, xp, xp_to_next}."""
        ...
    
    # ── Team Achievements ──
    def unlock_team_achievement(self, team_id: str, achievement_id: str,
                                 name: str, description: str = None) -> bool: ...
    def get_team_achievements(self, team_id: str) -> List[Dict]: ...
    
    # ── Rendering ──
    def render_hierarchy(self, team_id: str) -> str:
        """Return a tree-formatted string of the team hierarchy."""
        ...
```

### 4.2 `agent/skill_tree.py`

```python
class SkillTree:
    """Role-based skill trees that unlock on level-up."""
    
    # ── Default skill trees per role ──
    DEFAULT_SKILL_TREES: Dict[str, List[Dict]] = {
        "devops": [
            {"skill_name": "CI/CD Novice",       "level_threshold": 2, "skill_type": "passive",
             "description": "Basic CI pipeline understanding",
             "unlocked_content": "{\"bonus\": {\"kpi_weights\": {\"task_success_rate\": 0.1}}}"},
            {"skill_name": "Shell Scripter",      "level_threshold": 3, "skill_type": "active",
             "description": "Unlocks advanced shell tools",
             "unlocked_content": "{\"toolset\": \"system-engineer\"}"},
            {"skill_name": "Infrastructure Wizard", "level_threshold": 5, "skill_type": "prompt",
             "description": "Deep infrastructure intuition",
             "unlocked_content": "{\"prompt\": \"You have deep expertise in infrastructure-as-code and SRE practices.\"}"},
            {"skill_name": "K8s Master",           "level_threshold": 8, "skill_type": "active",
             "description": "Unlocks Kubernetes expert tools",
             "unlocked_content": "{\"toolset\": \"devops\"}"},
        ],
        "quant-trader": [
            {"skill_name": "Statistics Apprentice", "level_threshold": 2, "skill_type": "passive",
             "description": "Improved statistical reasoning",
             "unlocked_content": "{\"bonus\": {\"kpi_weights\": {\"role_proficiency_score\": 0.2}}}"},
            {"skill_name": "Options Strategist",    "level_threshold": 4, "skill_type": "active",
             "description": "Unlocks options pricing tools",
             "unlocked_content": "{\"toolset\": \"quant-trader\"}"},
            {"skill_name": "Risk Architect",         "level_threshold": 7, "skill_type": "prompt",
             "description": "Advanced risk management instincts",
             "unlocked_content": "{\"prompt\": \"You have elite-level risk management intuition.\"}"},
        ],
        "fullstack-dev": [
            {"skill_name": "CSS Master",            "level_threshold": 2, "skill_type": "passive",
             "description": "Frontend styling expertise",
             "unlocked_content": "{\"bonus\": {\"kpi_weights\": {\"tool_diversity_score\": 0.2}}}"},
            {"skill_name": "React Pro",             "level_threshold": 3, "skill_type": "active",
             "description": "Unlocks advanced frontend tools",
             "unlocked_content": "{\"toolset\": \"fullstack-dev\"}"},
            {"skill_name": "API Architect",         "level_threshold": 5, "skill_type": "prompt",
             "description": "Backend API design mastery",
             "unlocked_content": "{\"prompt\": \"You excel at designing robust, scalable APIs.\"}"},
            {"skill_name": "System Designer",        "level_threshold": 8, "skill_type": "active",
             "description": "Full system design capabilities",
             "unlocked_content": "{\"toolset\": \"system-engineer\"}"},
        ],
        "propfirm-trader": [
            {"skill_name": "Chart Reader",          "level_threshold": 2, "skill_type": "passive",
             "description": "Pattern recognition in charts",
             "unlocked_content": "{\"bonus\": {\"kpi_weights\": {\"task_success_rate\": 0.1}}}"},
            {"skill_name": "Risk Manager",           "level_threshold": 4, "skill_type": "prompt",
             "description": "Enhanced risk management",
             "unlocked_content": "{\"prompt\": \"You prioritize capital preservation above all else.\"}"},
            {"skill_name": "Trade Journaler",        "level_threshold": 6, "skill_type": "active",
             "description": "Unlocks trade journaling tools",
             "unlocked_content": "{\"toolset\": \"propfirm-trader\"}"},
        ],
        "content-creator": [
            {"skill_name": "Wordsmith",              "level_threshold": 2, "skill_type": "passive",
             "description": "Enhanced writing quality",
             "unlocked_content": "{\"bonus\": {\"kpi_weights\": {\"role_proficiency_score\": 0.2}}}"},
            {"skill_name": "Designer Eye",           "level_threshold": 4, "skill_type": "active",
             "description": "Unlocks image generation tools",
             "unlocked_content": "{\"toolset\": \"content-creator\"}"},
            {"skill_name": "SEO Expert",             "level_threshold": 6, "skill_type": "prompt",
             "description": "SEO-optimized content creation",
             "unlocked_content": "{\"prompt\": \"You optimize all content for search engines and audience engagement.\"}"},
        ],
        "system-engineer": [
            {"skill_name": "Kernel Explorer",        "level_threshold": 2, "skill_type": "passive",
             "description": "Deeper OS internals knowledge",
             "unlocked_content": "{\"bonus\": {\"kpi_weights\": {\"error_recovery_rate\": 0.2}}}"},
            {"skill_name": "Network Guru",            "level_threshold": 4, "skill_type": "active",
             "description": "Unlocks network diagnostic tools",
             "unlocked_content": "{\"toolset\": \"system-engineer\"}"},
            {"skill_name": "Security Sentinel",       "level_threshold": 7, "skill_type": "prompt",
             "description": "Security-first mindset",
             "unlocked_content": "{\"prompt\": \"You think like a security researcher — always enumerating attack surfaces.\"}"},
        ],
    }
    
    def __init__(self, db: SessionDB = None): ...
    
    # ── Skill definitions ──
    def seed_skills(self) -> None:
        """Insert DEFAULT_SKILL_TREES into DB (idempotent, skips existing)."""
        ...
    def get_skills_for_role(self, role: str) -> List[Dict]: ...
    def get_unlocked_skills(self, role: str) -> List[Dict]: ...
    def get_locked_skills(self, role: str) -> List[Dict]: ...
    
    # ── Unlocking ──
    def check_and_unlock(self, role: str, current_level: int) -> List[Dict]:
        """Check if any skills should be unlocked at this level. Returns newly unlocked skills."""
        ...
    def unlock_skill(self, role: str, skill_name: str) -> bool: ...
    
    # ── Applying ──
    def apply_unlocked_bonuses(self, role: str, base_kpi_weights: Dict[str, float]) -> Dict[str, float]:
        """Merge all unlocked 'passive' skill bonuses into kpi_weights."""
        ...
    def apply_unlocked_prompts(self, role: str) -> str:
        """Concatenate all unlocked 'prompt' skill bonuses into a system_prompt snippet."""
        ...
    def resolve_unlocked_toolsets(self, role: str) -> List[str]:
        """Return additional toolset names unlocked by 'active' skills."""
        ...
```

### 4.3 Extensions to `agent/gamification.py`

Add to `KPITracker`:

```python
def record_session_and_award_xp(
    self,
    session_id: str,
    role: str,
    metrics_dict: Dict[str, float],
    kpi_weights: Dict[str, float],
    team_id: str = None,
    team_manager: 'TeamManager' = None,
    skill_tree: 'SkillTree' = None,
) -> Dict:
    """Record KPI metrics, calculate XP, add to role + team, check skill unlocks.
    
    Returns: {
        "xp_result": {...},           # from add_xp()
        "team_xp_result": {...},      # from team_manager.add_team_xp() if team_id
        "newly_unlocked_skills": [],  # from skill_tree.check_and_unlock()
        "achievements_unlocked": [],  # auto-unlocked achievements
    }
    """
```

### 4.4 `agent/default_teams/` — YAML Team Templates

Create directory `agent/default_teams/` with pre-built team templates:

```yaml
# agent/default_teams/dev-squad.yaml
name: "Dev Squad"
description: "Full-stack development team"
lead_role: "fullstack-dev"
default_members:
  - agent_id: "dev-lead"
    role: "fullstack-dev"
    is_lead: true
    position: "Dev Lead"
  - agent_id: "frontend"
    role: "fullstack-dev"
    parent: "dev-lead"
    position: "Frontend Developer"
  - agent_id: "backend"
    role: "system-engineer"
    parent: "dev-lead"
    position: "Backend Developer"
  - agent_id: "sre"
    role: "devops"
    parent: "dev-lead"
    position: "Site Reliability Engineer"
```

```yaml
# agent/default_teams/quant-desk.yaml
name: "Quant Desk"
description: "Quantitative trading research and execution team"
lead_role: "quant-trader"
default_members:
  - agent_id: "strategy-lead"
    role: "quant-trader"
    is_lead: true
    position: "Strategy Lead"
  - agent_id: "analyst"
    role: "quant-trader"
    parent: "strategy-lead"
    position: "Research Analyst"
  - agent_id: "risk-mgr"
    role: "propfirm-trader"
    parent: "strategy-lead"
    position: "Risk Manager"
```

```yaml
# agent/default_teams/content-studio.yaml
name: "Content Studio"
description: "Content creation and social media team"
lead_role: "content-creator"
default_members:
  - agent_id: "editor-lead"
    role: "content-creator"
    is_lead: true
    position: "Editorial Lead"
  - agent_id: "writer"
    role: "content-creator"
    parent: "editor-lead"
    position: "Writer"
  - agent_id: "designer"
    role: "content-creator"
    parent: "editor-lead"
    position: "Visual Designer"
```

## 5. CLI Commands

### 5.1 New Command Registry Entry

In `hermes_cli/commands.py`, add to `COMMAND_REGISTRY`:

```python
CommandDef(
    "team", "Manage agent teams and hierarchy", "Configuration",
    args_hint="[create|list|show|disband|add-member|remove-member|skills|leaderboard]",
    subcommands=("create", "list", "show", "disband", "add-member", "remove-member",
                 "skills", "leaderboard", "xp", "achieve"),
),
```

### 5.2 Command Handlers (in `cli.py`)

| Command | Description | Example |
|---------|-------------|---------|
| `/team create <name>` | Create a new team (interactive or with flags) | `/team create dev-squad` |
| `/team list` | List all teams | `/team list` |
| `/team show <name\|id>` | Show team hierarchy tree | `/team show dev-squad` |
| `/team disband <name\|id>` | Delete a team and all members | `/team disband dev-squad` |
| `/team add-member <team> <agent_id> <role>` | Add member to team | `/team add-member dev-squad backend system-engineer --parent=dev-lead` |
| `/team remove-member <team> <agent_id>` | Remove member from team | `/team remove-member dev-squad backend` |
| `/team skills <role>` | Show skill tree for a role | `/team skills quant-trader` |
| `/team leaderboard` | Show team + individual XP ranking | `/team leaderboard` |
| `/team xp <team>` | Show team XP and level | `/team xp dev-squad` |
| `/team achieve <team>` | Show team achievements | `/team achieve dev-squad` |

### 5.3 Rendered Output Example

```
╭─ Dev Squad ──────────────────────────────────╮
│ Team Level: 5 | Team XP: 450/500 (90%)      │
│ Achievements: 2 (First Deploy, 10x Combo)    │
╰──────────────────────────────────────────────╯

  🎖️ dev-lead [fullstack-dev] Lv.7 ━━━━━━━━━━━━━ Leader
  ├── 🗂️ frontend [fullstack-dev] Lv.3 ★CSS Master ✓
  │   └── Skill Tree: [React Pro ✓] [A11y Expert ✗]
  ├── 🗂️ backend [system-engineer] Lv.5 ★Kernel Explorer ✓
  └── 🗂️ sre [devops] Lv.2
      └── Skill Tree: [CI/CD Novice ✗]
```

## 6. Integration with `delegate_task`

### 6.1 New Parameter: `team_context`

Add to `delegate_task` tool schema:

```python
{
    "name": "team_context",
    "type": "object",
    "description": "Team context for spawning a team-aware sub-agent.",
    "properties": {
        "team_id": {"type": "string", "description": "Team ID"},
        "team_name": {"type": "string", "description": "Team name (alternative to team_id)"},
        "role": {"type": "string", "description": "Role to assign (from RoleManager)"},
        "agent_id": {"type": "string", "description": "Agent ID within the team"},
        "parent_agent_id": {"type": "string", "description": "Parent agent in hierarchy"},
    },
}
```

### 6.2 Sub-Agent Prompt Assembly

When `team_context` is provided, the sub-agent's system prompt is assembled as:

```
{existing_system_prompt}

# Team Context
You are **{agent_id}** on team **{team_name}**.

{role_manager.build_role_system_prompt(role)}

{skill_tree.apply_unlocked_prompts(role)}

# Skill Tree Status
{rendered_skill_tree_for_role}
```

And the sub-agent's `enabled_toolsets` is assembled as:

```python
base_toolsets = role.toolsets
unlocked_toolsets = skill_tree.resolve_unlocked_toolsets(role)
final_toolsets = list(set(base_toolsets + unlocked_toolsets))
```

### 6.3 Post-Task XP Award

After a sub-agent completes its task, the orchestrator (or run_conversation loop) calls:

```python
gamification.record_session_and_award_xp(
    session_id=session_id,
    role=role,
    metrics_dict=computed_metrics,      # from session stats
    kpi_weights=role.kpi_weights,        # from RoleProfile
    team_id=team_id,                    # from team_context
    team_manager=team_manager,
    skill_tree=skill_tree,
)
```

## 7. Skill Tree Unlock Mechanics

### 7.1 Skill Types

| Type | Effect | Example |
|------|--------|---------|
| `passive` | Modifies `kpi_weights` (bonus multiplier) | +0.2 to `task_success_rate` weight |
| `active` | Unlocks additional `toolset` access | DevOps Lv.3 → unlocks `system-engineer` tools |
| `prompt` | Adds `system_prompt_extra` snippet | "You think like a security researcher..." |

### 7.2 Unlock Flow

```
Agent completes task
  └─→ KPITracker.add_xp(role, delta)
        └─→ new_level = floor(new_xp / 100) + 1
              └─→ if new_level > old_level:
                    └─→ SkillTree.check_and_unlock(role, new_level)
                          └─→ for each skill where level_threshold <= new_level AND unlocked == 0:
                                ├─→ SET unlocked = 1, unlocked_at = now()
                                └─→ yield {skill_name, skill_type, description}
```

### 7.3 Applying Unlocked Skills

When assembling a sub-agent prompt or computing effective KPI weights:

```python
effective_kpi_weights = skill_tree.apply_unlocked_bonuses(role, base_kpi_weights)
unlocked_prompt_snippets = skill_tree.apply_unlocked_prompts(role)
additional_toolsets = skill_tree.resolve_unlocked_toolsets(role)
```

## 8. Team Achievement System

### 8.1 Auto-Achievements

Achievements unlock automatically based on triggers:

| Achievement ID | Name | Trigger | Description |
|-------|------|---------|-------------|
| `first-deploy` | First Deploy | Team completes first task | Team completed its first deployment |
| `team-level-5` | Level 5 | Team reaches level 5 | Team reached level 5 |
| `team-level-10` | Level 10 | Team reaches level 10 | Team reached level 10 |
| `5-members` | Squad | Team has 5+ members | Team has 5 or more members |
| `all-roles` | Full House | Team has 6 unique roles | All default roles represented |
| `xp-1000` | XP Hoarders | Team XP exceeds 1000 | Team accumulated 1000+ XP |
| `combo-10x` | 10x Combo | 10 consecutive successful tasks | 10 tasks in a row succeeded |

### 8.2 Custom Achievements

Users can define custom achievements in team YAML:

```yaml
name: "Dev Squad"
achievements:
  - id: "ship-it"
    name: "Ship It!"
    description: "Deployed to production"
  - id: "zero-bugs"
    name: "Zero Bugs"
    description: "One week without bugs"
```

## 9. User-Configurable Team Templates

Users can override or add team templates in `~/.hermes/teams/*.yaml`:

```yaml
# ~/.hermes/teams/my-custom-team.yaml
name: "My Custom Team"
description: "My personal agent team"
lead_role: "fullstack-dev"
default_members:
  - agent_id: "lead"
    role: "fullstack-dev"
    is_lead: true
    position: "Team Lead"
  - agent_id: "dev"
    role: "fullstack-dev"
    parent: "lead"
    position: "Developer"
  - agent_id: "ops"
    role: "devops"
    parent: "lead"
    position: "DevOps Engineer"
achievements:
  - id: "custom-first"
    name: "Custom Achievement"
    description: "A custom team achievement"
```

`TeamManager` loads templates from both `agent/default_teams/*.yaml` (built-in) and `~/.hermes/teams/*.yaml` (user overrides), similar to how `RoleManager` works.

## 10. Implementation Order

### Phase 1: Foundation (DB + Core Classes)
1. **hermes_state.py**: Bump `SCHEMA_VERSION` to 12, add `teams`, `team_members`, `skill_trees`, `team_achievements` tables + v12 migration
2. **agent/team_manager.py**: `TeamManager` class (team CRUD, member management, hierarchy, team XP, team achievements)
3. **agent/skill_tree.py**: `SkillTree` class (default skill trees, seed/unlock/apply logic)

### Phase 2: Gamification Integration
4. **agent/gamification.py**: Add `record_session_and_award_xp()` method that chains KPI → XP → team XP → skill unlock checks
5. **agent/skill_tree.py**: Seed default skills on first init

### Phase 3: CLI
6. **hermes_cli/commands.py**: Add `/team` CommandDef with subcommands
7. **cli.py**: Implement `/team` command handlers (create, list, show, disband, add-member, remove-member, skills, leaderboard, xp, achieve)

### Phase 4: delegate_task Integration
8. **tools/delegate_tool.py**: Add `team_context` parameter, inject role prompt + skill tree bonuses + unlocked toolsets into sub-agent prompt
9. **run_agent.py**: Hook `record_session_and_award_xp()` into the agent loop after task completion when role/team context is active

### Phase 5: User Templates + polish
10. **agent/default_teams/*.yaml**: Ship 3 built-in team templates (dev-squad, quant-desk, content-studio)
11. **TeamManager._load_user_teams()**: Scan `~/.hermes/teams/*.yaml` and merge with defaults
12. **Rendering**: `render_hierarchy()` ASCII tree display for `/team show`

### Phase 6: Tests
13. **tests/integration/test_team_hierarchy.py**: Team creation, member CRUD, hierarchy operations
14. **tests/integration/test_skill_tree.py**: Skill unlock, apply bonuses, apply prompts, resolve toolsets
15. **tests/integration/test_team_gamification.py**: Team XP, team achievements, auto-achievement triggers
16. **tests/integration/test_delegate_team_context.py**: Sub-agent spawning with team context

## 11. File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `hermes_state.py` | Modify | Bump schema to v12, add 4 new tables + indexes |
| `agent/team_manager.py` | New | TeamManager class |
| `agent/skill_tree.py` | New | SkillTree class + DEFAULT_SKILL_TREES |
| `agent/gamification.py` | Modify | Add `record_session_and_award_xp()` |
| `hermes_cli/commands.py` | Modify | Add `/team` CommandDef |
| `cli.py` | Modify | Add `/team` command handlers |
| `tui_gateway/server.py` | Modify | Add `/team` RPC methods |
| `tools/delegate_tool.py` | Modify | Add `team_context` parameter |
| `run_agent.py` | Modify | Hook gamification into agent loop |
| `agent/default_teams/dev-squad.yaml` | New | Built-in team template |
| `agent/default_teams/quant-desk.yaml` | New | Built-in team template |
| `agent/default_teams/content-studio.yaml` | New | Built-in team template |
| `tests/integration/test_team_hierarchy.py` | New | Team hierarchy tests |
| `tests/integration/test_skill_tree.py` | New | Skill tree tests |
| `tests/integration/test_team_gamification.py` | New | Team gamification tests |

## 12. Design Decisions & Rationale

### Why separate Team XP vs Individual XP?
- Individual XP tracks role proficiency (devops Lv.5 means that role is experienced)
- Team XP tracks collective achievement (the team as a whole levels up)
- A high-level individual joining a new team shouldn't inflate its team level
- Team level unlocks team achievements, individual level unlocks skill tree nodes

### Why Skill Trees per Role (not per agent)?
- Roles are the stable unit — agents come and go, but roles persist
- A "devops Lv.5" skill means the same thing across all agents with that role
- This matches how the existing `agent_skills_xp` table works (keyed on `skill_name` = role name)

### Why YAML for team templates?
- Consistent with `RoleManager` pattern (roles defined in YAML)
- Easy for users to customize in `~/.hermes/teams/`
- Can be version-controlled
- No need for a database migration to add a new template

### Why `parent_member_id` instead of `depth` field?
- Supports arbitrary tree structures (not just 2 levels)
- Enables `get_subtree()` and `get_lineage()` traversals
- Matches the `delegate_task` hierarchy (orchestrator spawns sub-orchestrators)

### Why auto-achievements + custom achievements?
- Auto-achievements provide immediate gamification without user setup
- Custom achievements via YAML give teams personality and purpose-specific milestones
- Both stored in `team_achievements` table with `achievement_id` uniqueness

## 13. Resolved Design Decisions (from user requirements session 2026-04-30)

### Hierarchy Type: Deep Hierarchy
User chose: **Lead → Sub-Lead → Workers** (multi-level, like org chart)
NOT flat team (1 lead + N leaves) and NOT dynamic swarm.

### Gamification Depth: Full RPG
User chose: **Team Level + Individual Level + Achievements + Skill Trees**
- Team XP Pool: aggregated team level (like Guild Level in RPG)
- Individual XP: per-role XP/Level that unlocks skill tree nodes
- Achievements: both auto-triggered and custom via YAML
- Skill Trees: unlock new capabilities (passive bonuses, active toolsets, prompt snippets) on level-up

### Team Creation: CLI Command
User chose: **`/team create` from CLI command** (not YAML-only, not auto-spawned)
Example syntax:
```
/team create my-team --lead orchestrator --members 'quant-trader,fullstack-dev'
/team create dev-squad --template dev-squad
```
YAML templates in `~/.hermes/teams/` serve as presets that `/team create` can reference via `--template`.

### XP Design: Both Team + Individual
- Individual role XP/Level: tracks skill proficiency (existing `agent_skills_xp`)
- Team XP Pool: sum of individual contributions → Team Level
- Skill Trees keyed on Individual Level (per role)
- Team Achievements keyed on Team Level/stats

## 14. Open Questions (for implementation session)

1. **Should `/team create` be interactive (prompt for members) or require all args?**
   → Recommend: interactive if no args, one-shot with `--template` flag
   → User preference: one-shot CLI command with flags (`--lead`, `--members`, `--template`)
   
2. **Should skill tree unlocks be retroactive when seeding?**
   → Recommend: Yes — if role is already Lv.5, seed all skills with threshold ≤ 5 as unlocked

3. **Should team XP be a flat sum of all member XP, or weighted?**
   → Recommend: Flat sum for simplicity, with a configurable `team_xp_multiplier` per team

4. **Max team size limit?**
   → Recommend: Soft limit of 20 members (configurable), hard limit bounded by `delegation.max_spawn_depth`

5. **Should `/team create` auto-add the Team Lead as first member?**
   → Recommend: Yes — `--lead <role>` creates the team AND adds a lead member with that role

6. **Should individual XP contribution to team XP be one-time (on task completion) or recalculated?**
   → Recommend: One-time on task completion (append-only) — avoids retroactive recalculation

7. **Should Sub-Leads get XP bonus for tasks completed by their subtree workers?**
   → Recommend: Yes — configurable `subtree_xp_pct` (default 10%) flows up the hierarchy