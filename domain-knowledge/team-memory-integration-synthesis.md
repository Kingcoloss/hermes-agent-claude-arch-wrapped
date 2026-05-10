---
scope: Integration synthesis — Agent Team Hierarchy + Second Brain Memory
audience: Implementation agent (next session)
sources:
  - domain-knowledge/agent-team-hierarchy-gamification.md
  - domain-knowledge/second-brain-memory-architecture.md
generated: 2026-04-30
status: planning-complete
---

# Team Hierarchy × Second Brain Memory — Integration Synthesis

## 1. Executive Summary

สองระบบนี้มีจุดตัดกันหลายจุด โดยหลักๆ คือ **Team เป็นหน่วยของความรู้เช่นเดียวกับ Role** และ **การทำ task ของ agent ในทีมคือแหล่งข้อมูลหลักของ consolidation pipeline** บูรณาการแล้ว จะได้ระบบที่:

- Team members มี role-scoped vault (LTM) + team-shared vault (collective memory)
- XP award triggers consolidation — ทุกครั้งที่ `record_session_and_award_xp()` ทำงาน จะ trigger ให้ STM ถูกประเมินและ promote เป็น LTM อัตโนมัติ
- Skill tree unlocks ขยาย vault access — level-up ไม่แค่ปล่อย skill ใหม่ แต่ unlock พื้นที่ความรู้ใหม่ใน vault
- Team hierarchy กำหนด memory access pattern — Lead เห็น vault กว้างกว่า Worker
- Team achievements คือ LTM experience notes ที่ถูก consolidate อัตโนมัติ

## 2. Integration Matrix

| จาก Team/Gamification | จาก Second Brain | จุดบูรณาการ |
|---|---|---|
| Team Lead → Sub-Leads → Workers | Role-scoped vaults per member | Hierarchy เป็น access control layer ของ vault |
| `record_session_and_award_xp()` | Consolidation pipeline (STM → LTM) | XP award = consolidation trigger |
| Skill Trees (passive/active/prompt) | Vault knowledge dimensions (D1-D4) | Skill unlock = vault section unlock |
| Team XP Pool + Team Level | LTM collective experience (D4) | Team level = vault depth/breadth |
| Team Achievements | LTM experience notes | Achievement = auto-consolidated LTM note |
| `delegate_task` team_context | `build_role_memory_context()` boot protocol | ประกอบ system prompt รวมทั้ง team + vault |
| Role KPI weights | Vault confidence scores | KPI performance = confidence calibration |
| YAML team templates | Vault INDEX.md + wiki-links | Template = vault seed structure |
| `render_hierarchy()` | Vault backlinks | Hierarchy tree = memory access graph |

## 3. New Concept: Team-Scoped Vault

### 3.1 Three-Layer Memory Architecture

เดิม Second Brain มี 2 layer (LTM + STM) ต่อหนึ่ง role เมื่อบูรณาการกับ Team Hierarchy จะกลายเป็น 3 layer:

```
┌──────────────────────────────────────────────────────┐
│              Team Vault (Collective LTM)               │
│           ~/.hermes/vaults/teams/<team_id>/           │
│                                                       │
│  ┌────────────┐ ┌──────────────┐ ┌─────────────────┐ │
│  │ Team       │ │ Cross-Role   │ │ Team            │ │
│  │ Decisions  │ │ Patterns     │ │ Experience      │ │
│  │            │ │              │ │                 │ │
│  │ strategic  │ │ patterns     │ │ project         │ │
│  │ choices    │ │ shared       │ │ outcomes        │ │
│  │ rationale  │ │ across roles │ │ lessons         │ │
│  └────────────┘ └──────────────┘ └─────────────────┘ │
│                                                       │
│  Access: all team members (read), Lead (write)       │
│  Structure: wiki-linked markdown, auto-indexed        │
└──────────────────────────────────────────────────────┘
          ▲                ▲                ▲
          │ wiki-links     │ wiki-links     │ consolidation
          │                │                │
┌─────────┴──────┐ ┌──────┴───────┐ ┌──────┴───────────────┐
│  Role Vault    │ │  Role Vault  │ │  Role Vault           │
│  fullstack-dev │ │  devops      │ │  quant-trader         │
│  (Individual   │ │  (Individual │ │  (Individual          │
│   LTM)         │ │   LTM)      │ │   LTM)                │
│                │ │              │ │                       │
│  personality/  │ │  personality/│ │  personality/         │
│  hard-skill/  │ │  hard-skill/ │ │  hard-skill/          │
│  soft-skill/  │ │  soft-skill/ │ │  soft-skill/          │
│  experience/  │ │  experience/ │ │  experience/          │
└────────────────┘ └──────────────┘ └───────────────────────┘
```

### 3.2 Team Vault Schema

```sql
-- เพิ่มใน hermes_state.py schema v12 (ร่วมกับ teams table)
CREATE TABLE IF NOT EXISTS team_vault_notes (
    id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    note_path TEXT NOT NULL,             -- relative path in team vault
    dimension TEXT NOT NULL,             -- 'team-decision' | 'cross-role-pattern' | 'team-experience'
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    backlinks TEXT,                      -- JSON array of [[wiki-link]] targets
    tags TEXT,                           -- JSON array of role tags
    confidence REAL DEFAULT 0.8,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(team_id, note_path)
);

CREATE INDEX IF NOT EXISTS idx_team_vault_team ON team_vault_notes(team_id);
CREATE INDEX IF NOT EXISTS idx_team_vault_dimension ON team_vault_notes(dimension);
```

### 3.3 Directory Structure (Extended)

```
~/.hermes/
├── vaults/
│   ├── fullstack-dev/          # Individual role vault (from Second Brain)
│   ├── devops/
│   ├── quant-trader/
│   ├── ...
│   └── teams/                  # NEW: Team-scoped collective vault
│       ├── dev-squad/
│       │   ├── INDEX.md
│       │   ├── decisions/
│       │   │   └── architecture-choice.md
│       │   ├── cross-role/
│       │   │   └── frontend-backend-api-contract.md
│       │   └── experience/
│       │       └── v1-launch-postmortem.md
│       └── quant-desk/
│           ├── INDEX.md
│           ├── decisions/
│           └── experience/
├── sessions/                   # STM (unchanged)
└── memories/                   # Legacy (migration source)
```

## 4. XP Award = Consolidation Trigger

### 4.1 Unified Event: `task_completed`

เมื่อ agent ทำ task เสร็จ ปัจจุบัน gamification flow คือ:

```
Task completes → record_session_and_award_xp() → {xp_result, team_xp_result, new_skills, achievements}
```

เมื่อบูรณาการ จะขยายเป็น:

```
Task completes
  └─→ record_session_and_award_xp()     # Gamification
  │     ├─→ add_xp(role, delta)         # Individual XP
  │     ├─→ add_team_xp(team_id, delta)  # Team XP
  │     ├─→ check_and_unlock(role, lvl)  # Skill Tree
  │     └─→ unlock_team_achievements()   # Team Achievements
  │
  └─→ consolidation_engine.consolidate_session(session_id)  # NEW: Second Brain
        ├─→ evaluate STM entries (relevance + novelty + team context)
        ├─→ summarize against existing LTM (role vault + team vault)
        ├─→ merge into appropriate vault (role or team)
        ├─→ link wiki-links (cross-vault if team member)
        └─→ tag with role + team access annotations
```

### 4.2 Modified `record_session_and_award_xp()`

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
    consolidation_engine: 'ConsolidationEngine' = None,   # NEW
) -> Dict:
    """Record KPI metrics, calculate XP, add to role + team, check skill unlocks,
    AND trigger memory consolidation.

    Returns: {
        "xp_result": {...},
        "team_xp_result": {...},
        "newly_unlocked_skills": [],
        "achievements_unlocked": [],
        "consolidation_result": {                    # NEW
            "promoted": [...],    # STM entries promoted to LTM
            "skipped": [...],     # Low relevance entries
            "team_notes": [...],  # Notes written to team vault
        },
    }
    """
```

### 4.3 Consolidation Routing: Role Vault vs Team Vault

Consolidation pipeline ต้องตัดสินใจว่า STM entry ควรไปที่ไหน:

| STM Entry Type | Destination | Rationale |
|---|---|---|
| Role-specific technical fact | Role vault (hard-skill) | ความรู้เฉพาะ role เช่น FastAPI pattern → fullstack-dev vault |
| User personality observation | Role vault (personality) | ความชอบของ user → shared across all roles |
| Cross-role collaboration pattern | **Team vault** (cross-role) | Frontend + Backend API contract → team knowledge |
| Project decision/rationale | **Team vault** (decisions) | การตัดสินใจระดับทีม → ทุกคนต้องรู้ |
| Task outcome/lesson | **Team vault** (experience) | Post-mortem → ทีมเรียนรู้ร่วมกัน |
| Error recovery pattern | Role vault (soft-skill) | วิธีจัดการ error → role-specific |
| Sub-tree performance (Sub-Lead) | **Team vault** (decisions) | การบริหาร subtree → team-level |

### 4.4 Team-Aware Evaluation Score

`ConsolidationEngine._evaluate()` ต้องคำนึงถึง team context:

```python
def _evaluate(self, entry: dict, ltm_index: dict, team_context: dict = None) -> float:
    """Score relevance (0-1) for consolidation, considering team membership."""
    base_score = self._base_evaluate(entry, ltm_index)

    if team_context:
        # Boost score if entry is relevant to team's active project
        team_project_keywords = team_context.get("active_project_keywords", [])
        keyword_overlap = self._keyword_overlap(entry, team_project_keywords)
        base_score += keyword_overlap * 0.15

        # Boost if entry contains cross-role collaboration signals
        if self._has_cross_role_signals(entry, team_context.get("member_roles", [])):
            base_score += 0.1

    return min(base_score, 1.0)
```

## 5. Skill Tree Unlock = Vault Section Unlock

### 5.1 Concept: Skill-Gated Knowledge

Skill tree ไม่แค่ปล่อย tool/prompt/bonus แต่ปล่อย **การเข้าถึงความรู้ใน vault** ด้วย:

| Skill Type | Vault Effect | Example |
|---|---|---|
| `passive` | Unlocks vault sections with bonus KPI weights | "CSS Master" Lv.2 → อ่าน `hard-skill/css-advanced.md` ได้ |
| `active` | Unlocks vault sections with new toolset access | "React Pro" Lv.3 → อ่าน `hard-skill/react-patterns.md` + unlock fullstack-dev toolset |
| `prompt` | Unlocks vault sections + injects prompt snippet | "API Architect" Lv.5 → อ่าน `hard-skill/api-design.md` + system prompt bonus |
| **`vault`** (NEW) | Unlocks specific vault note/section only | "Infrastructure Wizard" Lv.5 → อ่าน `experience/k8s-migration.md` ได้ (ก่อนหน้านี้ locked) |

### 5.2 Skill Tree Schema Extension

```sql
-- เพิ่มคอลัมน์ใน skill_trees table
ALTER TABLE skill_trees ADD COLUMN vault_unlocks TEXT;
-- JSON array of vault paths unlocked: ["hard-skill/css-advanced.md", "experience/k8s-migration.md"]
```

ตัวอย่างใน `DEFAULT_SKILL_TREES`:

```python
"fullstack-dev": [
    {
        "skill_name": "CSS Master",
        "level_threshold": 2,
        "skill_type": "passive",
        "description": "Frontend styling expertise",
        "unlocked_content": '{"bonus": {"kpi_weights": {"tool_diversity_score": 0.2}}}',
        "vault_unlocks": '["hard-skill/css-advanced.md"]',  # NEW
    },
    {
        "skill_name": "API Architect",
        "level_threshold": 5,
        "skill_type": "prompt",
        "description": "Backend API design mastery",
        "unlocked_content": '{"prompt": "You excel at designing robust, scalable APIs."}',
        "vault_unlocks": '["hard-skill/api-design-patterns.md", "experience/api-migration-guide.md"]',  # NEW
    },
    ...
],
```

### 5.3 Boot-Time Vault Assembly with Skill Gates

เมื่อ `build_role_memory_context()` ทำงาน จะกรอง vault notes ตาม skill unlock status:

```python
def build_role_memory_context(role: str, session_id: str, skill_tree: SkillTree = None) -> str:
    """Load role's LTM index at session start, respecting skill-gated access."""
    vault = Path(get_hermes_home()) / "vaults" / role
    index_path = vault / "INDEX.md"

    if not index_path.exists():
        return ""

    context = f"[{role} Second Brain Index]\n"

    # Load INDEX.md — แต่ mark locked sections
    index_content = index_path.read_text()

    if skill_tree:
        unlocked_paths = skill_tree.resolve_unlocked_vault_paths(role)
        # Annotate INDEX.md: mark locked vault notes with 🔒
        annotated_index = _annotate_locked_sections(index_content, unlocked_paths)
        context += annotated_index
    else:
        context += index_content

    # Load personality (always available — not skill-gated)
    personality_path = vault / "personality" / "user-personality.md"
    if personality_path.exists():
        context += f"\n[User Personality (high confidence)]\n{personality_path.read_text()[:2000]}\n"

    # Load only unlocked hard-skill notes
    if skill_tree:
        for path in unlocked_paths:
            note_file = vault / path
            if note_file.exists():
                context += f"\n[Unlocked: {path}]\n{note_file.read_text()[:1000]}\n"

    return context
```

## 6. Team Hierarchy = Memory Access Hierarchy

### 6.1 Access Control by Depth

Team hierarchy กำหนดว่า agent เห็นอะไรใน team vault:

| Position | Team Vault Access | Role Vault Access |
|---|---|---|
| Team Lead (depth=0) | Full read/write all team vault notes | Full read own vault + `roles_aware` summaries of all member roles |
| Sub-Lead (depth=1) | Read all + write in subtree scope | Full read own vault + `roles_aware` summaries of subtree roles |
| Worker (depth=2+) | Read all team vault (read-only) | Full read own vault only |

### 6.2 Access Enforcement in ConsolidationEngine

```python
def can_write_team_vault(self, member: dict) -> bool:
    """Only leads and sub-leads can write to team vault."""
    return member.get("is_lead", 0) == 1 or member.get("depth", 99) <= 1

def get_readable_vault_notes(self, member: dict, team: dict) -> list:
    """Return vault notes this member is allowed to read."""
    role_vault = self._get_role_vault_notes(member["role"])
    team_vault = self._get_team_vault_notes(team["id"])

    # Role vault: always read own, plus roles_aware from others
    readable = [n for n in role_vault if n["role"] == member["role"]
                or member["role"] in n.get("roles_aware", [])]

    # Team vault: all members can read, depth-based write
    readable.extend(team_vault)

    return readable
```

### 6.3 Sub-Lead XP Bonus → Memory Bonus

จาก open question #7 ใน team hierarchy doc: "Sub-Leads get XP bonus for tasks completed by their subtree workers"

เมื่อบูรณาการแล้ว Sub-Lead ไม่แค่ได้ XP bonus แต่ได้ **memory bonus** — สามารถเห็น consolidated experience notes จาก subtree ทั้งหมด:

```python
def get_subtree_experience_notes(self, member_id: str, team_manager: TeamManager) -> List[Dict]:
    """Sub-Leads can read all experience notes from their subtree."""
    subtree = team_manager.get_subtree(member_id)
    subtree_roles = [m["role"] for m in subtree]
    notes = []
    for role in set(subtree_roles):
        vault_dir = Path(get_hermes_home()) / "vaults" / role / "experience"
        if vault_dir.exists():
            notes.extend(self._load_vault_notes(vault_dir))
    return notes
```

## 7. Team Achievements = Auto-Consolidated LTM Notes

### 7.1 Achievement → LTM Note Flow

เมื่อ team achievement ถูก unlock อัตโนมัติ จะ trigger การสร้าง LTM note ใน team vault:

```
Achievement unlocked: "10x Combo"
  └─→ TeamManager.unlock_team_achievement(team_id, "combo-10x", "10x Combo", ...)
        └─→ ConsolidationEngine.create_team_experience_note(
              team_id=team_id,
              title="10-Consecutive-Success Streak",
              dimension="team-experience",
              content="Team achieved 10 consecutive successful tasks. Key factors: ...",
              tags=["team-achievement", "combo-10x"],
              auto_generated=True,
            )
```

### 7.2 Achievement-Triggered Notes Schema

```python
TEAM_ACHIEVEMENT_LTM_MAP = {
    "first-deploy": {
        "dimension": "team-experience",
        "template": "First Deployment — The team completed its first deployment. Context and outcome: {context}",
    },
    "combo-10x": {
        "dimension": "team-experience",
        "template": "10x Success Streak — Team achieved 10 consecutive successes. Patterns that contributed: {context}",
    },
    "team-level-5": {
        "dimension": "team-decision",
        "template": "Team Level 5 Milestone — Strategic decisions and member contributions that drove growth: {context}",
    },
    "all-roles": {
        "dimension": "cross-role-pattern",
        "template": "Full House — All roles represented. Cross-role collaboration patterns observed: {context}",
    },
}
```

## 8. Unified Prompt Assembly for `delegate_task`

### 8.1 Current Flow (from Team Hierarchy doc)

```
{existing_system_prompt}
# Team Context
You are **{agent_id}** on team **{team_name}**.
{role_manager.build_role_system_prompt(role)}
{skill_tree.apply_unlocked_prompts(role)}
# Skill Tree Status
{rendered_skill_tree_for_role}
```

### 8.2 Integrated Flow (Team + Memory)

```
{existing_system_prompt}

# Team Context
You are **{agent_id}** on team **{team_name}**.
Your position: {position} (depth={depth})
Team Level: {team_level} | Team XP: {team_xp}/{xp_to_next}

{role_manager.build_role_system_prompt(role)}

# Second Brain — Role Knowledge
{build_role_memory_context(role, session_id, skill_tree)}

# Second Brain — Team Knowledge
{build_team_memory_context(team_id, member_depth)}

{skill_tree.apply_unlocked_prompts(role)}

# Skill Tree Status
{rendered_skill_tree_for_role}

# Vault Access
You can access: {list_accessible_vault_paths(role, skill_tree, team_id, depth)}
```

### 8.3 `build_team_memory_context()`

```python
def build_team_memory_context(team_id: str, member_depth: int = 99) -> str:
    """Load team vault index + high-relevance notes for team context."""
    team_vault = Path(get_hermes_home()) / "vaults" / "teams" / team_id
    index_path = team_vault / "INDEX.md"

    if not index_path.exists():
        return ""

    context = f"[Team {team_id} — Collective Knowledge Index]\n"
    context += index_path.read_text()

    # For leads: load recent team decisions in full
    if member_depth <= 1:
        decisions_dir = team_vault / "decisions"
        if decisions_dir.exists():
            for note in sorted(decisions_dir.glob("*.md"))[-3:]:  # 3 ล่าสุด
                context += f"\n[Decision: {note.stem}]\n{note.read_text()[:1000]}\n"

    return context
```

## 9. Cross-Role Wiki-Links in Team Context

### 9.1 Auto-Linking at Consolidation Time

เมื่อ consolidation pipeline ทำงานใน team context จะค้นหา wiki-link targets ข้าม vault boundary:

```python
def _discover_links(self, note: MemoryNote, index: dict, team_context: dict = None) -> list[str]:
    """Find wiki-link targets, including cross-role and team vault."""
    links = []

    # 1. Same-role links (existing behavior)
    links.extend(self._same_role_links(note, index))

    # 2. Cross-role links via roles_aware
    if team_context:
        member_roles = team_context.get("member_roles", [])
        for other_role in member_roles:
            if other_role != note.role:
                other_vault = Path(get_hermes_home()) / "vaults" / other_role
                cross_links = self._find_semantic_links(note, other_vault)
                links.extend(f"{other_role}:{link}" for link in cross_links)

    # 3. Team vault links
    if team_context and team_context.get("team_id"):
        team_vault = Path(get_hermes_home()) / "vaults" / "teams" / team_context["team_id"]
        team_links = self._find_semantic_links(note, team_vault)
        links.extend(f"team:{link}" for link in team_links)

    return links
```

### 9.2 Team Wiki-Link Syntax Extension

| Syntax | Meaning | Example |
|---|---|---|
| `[[note-name]]` | Same-role vault link | `[[fastapi-patterns]]` |
| `[[role:note-name]]` | Cross-role vault link | `[[quant-trader:black-scholes-patterns]]` |
| `[[team:note-name]]` | **NEW**: Team vault link | `[[team:api-contract-frontend-backend]]` |
| `[[team:note-name#section]]` | Team vault section link | `[[team:architecture-choice#database]]` |

## 10. Data Flow: Complete Integrated Cycle

```
User: /team create dev-squad --template dev-squad
  ├─→ TeamManager.create_team("dev-squad", ...)
  ├─→ Seed team vault directory: ~/.hermes/vaults/teams/dev-squad/
  └─→ Create INDEX.md for team vault

User: delegates task to "frontend" agent with team_context
  ├─→ delegate_task(team_context={team_id, agent_id, role, parent})
  ├─→ Assemble prompt:
  │     ├─→ role system prompt (RoleManager)
  │     ├─→ role vault INDEX.md (Second Brain)
  │     ├─→ team vault INDEX.md (NEW)
  │     ├─→ skill tree bonuses + unlocked prompts
  │     └─→ vault access list (skill-gated)
  └─→ Sub-agent runs task

Task completes
  ├─→ record_session_and_award_xp(role, metrics, team_id, ...)
  │     ├─→ add_xp(role, delta)          → Individual XP
  │     ├─→ add_team_xp(team_id, delta)  → Team XP Pool
  │     ├─→ check_and_unlock(role, lvl)  → Skill Tree (may unlock vault sections)
  │     └─→ unlock_team_achievements()   → May create LTM team experience note
  │
  └─→ consolidation_engine.consolidate_session(session_id, team_context)
        ├─→ Evaluate: score STM entries (team-aware scoring)
        ├─→ Summarize: dedup against role vault + team vault
        ├─→ Merge: route to role vault or team vault based on dimension
        ├─→ Link: discover cross-role + team wiki-links
        └─→ Tag: annotate roles_direct + team_id

Next session boot
  ├─→ build_role_memory_context(role, skill_tree)  → Skill-gated vault load
  └─→ build_team_memory_context(team_id, depth)   → Team collective knowledge
```

## 11. New Modules / Extensions

### 11.1 New Files

| File | Description |
|---|---|
| `agent/team_vault_manager.py` | CRUD for team vault notes, access control based on hierarchy depth |
| `agent/default_teams/` (extended) | YAML templates now include `vault_seed` section for initial team vault content |

### 11.2 Modified Files

| File | Change |
|---|---|
| `agent/gamification.py` | `record_session_and_award_xp()` เพิ่ม `consolidation_engine` param + trigger |
| `agent/consolidation_engine.py` | เพิ่ม team_context param, team vault routing, cross-vault wiki-links |
| `agent/vault_memory_provider.py` | เพิ่ม team vault boot loading, skill-gated access |
| `agent/skill_tree.py` | เพิ่ม `vault_unlocks` field, `resolve_unlocked_vault_paths()` method |
| `agent/team_manager.py` | เพิ่ม team vault directory seeding, achievement → LTM note trigger |
| `hermes_state.py` | เพิ่ม `team_vault_notes` table + `skill_trees.vault_unlocks` column |
| `tools/delegate_tool.py` | เพิ่ม team vault context in prompt assembly |
| `agent/prompt_builder.py` | เพิ่ม `build_team_memory_context()` injection |

### 11.3 New Test Files

| File | Description |
|---|---|
| `tests/integration/test_team_vault.py` | Team vault CRUD, access control by depth |
| `tests/integration/test_skill_vault_gating.py` | Skill unlock → vault access verification |
| `tests/integration/test_team_consolidation_routing.py` | STM → role vault vs team vault routing |
| `tests/integration/test_achievement_ltm_notes.py` | Achievement unlock → LTM note creation |

## 12. Implementation Phases (Integrated)

### Phase 1: Foundation (shared with both docs)
- Schema v12: teams, team_members, skill_trees (with vault_unlocks), team_achievements, team_vault_notes
- TeamManager class
- SkillTree class (with vault_unlocks)
- ConsolidationEngine class (base, without team awareness)

### Phase 2: Team Vault Layer
- TeamVaultManager class
- Team vault directory seeding on `/team create`
- Access control by hierarchy depth
- Team vault INDEX.md auto-generation

### Phase 3: Integration Wiring
- `record_session_and_award_xp()` triggers consolidation
- ConsolidationEngine team-aware routing (role vs team vault)
- Skill tree `resolve_unlocked_vault_paths()` method
- Achievement → LTM team experience note auto-creation

### Phase 4: Prompt Assembly Integration
- `delegate_tool.py` injects team vault context
- `build_role_memory_context()` with skill-gated access
- `build_team_memory_context()` for team collective knowledge
- Cross-vault wiki-link resolution (`[[team:...]]` syntax)

### Phase 5: CLI + UX
- `/team show` renders hierarchy + vault status
- `/team vault <team>` shows team vault INDEX
- `/memory consolidate` supports team context
- `/memory lint` covers team vault

### Phase 6: Tests
- Team vault CRUD + access control
- Skill-gated vault access
- Consolidation routing (role vs team)
- Achievement → LTM note creation
- Cross-vault wiki-link resolution
- End-to-end: task → XP → consolidation → next boot context

## 13. Key Design Decisions

### 13.1 Why team vault as separate layer (not just shared role vault)?

- **Scope**: Team knowledge is different from role knowledge — "we chose FastAPI for this project" is not a fullstack-dev fact, it's a team decision
- **Access control**: Team vault access depends on hierarchy position, not role
- **Lifecycle**: When a team is disbanded, team vault can be archived or merged into role vaults — role vaults persist independently
- **Cross-role**: Team vault naturally captures knowledge at the intersection of roles (API contracts, integration patterns, project decisions)

### 13.2 Why skill-gated vault access?

- **Incentive alignment**: Level-up gives tangible knowledge access, not just KPI bonuses
- **Context budget**: Lower-level agents get smaller context windows — vault gating prevents information overload
- **Progressive disclosure**: New agents start with core knowledge, unlock advanced patterns as they prove competence
- **Research-backed**: Karpathy's LLM Wiki pattern — "compilation is lossy and opinionated" — gating means the LLM curates what's important at each level

### 13.3 Why achievements auto-create LTM notes?

- **Knowledge preservation**: Achievements represent significant events — losing them means the team forgets why they succeeded
- **Cross-session continuity**: Without LTM notes, achievements are just counters in a DB — no narrative, no lessons learned
- **Motivation loop**: Team members can read team experience notes → learn from past successes → repeat patterns

### 13.4 Why consolidation triggers on XP award (not just session end)?

- **Temporal alignment**: XP is awarded when a task completes — that's the moment the knowledge is freshest and most relevant
- **Completeness guarantee**: If XP is awarded, consolidation runs — no orphan STM entries
- **Team context availability**: At XP award time, we know the team_id, role, and metrics — everything needed for intelligent routing
- **Not mutually exclusive**: Session-end consolidation still runs as a catch-all for entries that didn't get promoted during XP-triggered consolidation

## 14. Risk & Mitigation

| Risk | Mitigation |
|---|---|
| Context window overflow with 3 vault layers | INDEX.md as compact catalog, full notes loaded on-demand via `[[wiki-link]]` resolution, not bulk |
| Team vault becoming write-only (no one reads it) | Boot protocol injects team vault INDEX.md; achievements auto-create notes with high confidence |
| Skill-gated vault creating "knowledge silos" | `roles_aware` tags allow summary-level access; team vault bridges role vaults |
| Consolidation double-trigger (XP + session end) | Idempotent: STM entries marked `consolidation_status: promoted` after first pass |
| Team disbanded but vault orphaned | `disband_team()` offers archive option; archived vaults become read-only role vault notes |
| Cross-vault wiki-links creating tangles | Lint pass includes broken cross-vault link detection; team vault deletion removes links |