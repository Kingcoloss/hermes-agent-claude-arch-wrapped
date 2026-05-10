---
scope: Real-World Company Architecture — Hermes as a Full Company Operating System
audience: Implementation agent (next session), CEO (user)
sources:
  - domain-knowledge/agent-team-hierarchy-gamification.md
  - domain-knowledge/second-brain-memory-architecture.md
  - domain-knowledge/team-memory-integration-synthesis-v2.md
  - domain-knowledge/team-memory-integration-synthesis-v2-reviewed.md (gap analysis)
  - codebase analysis (code-review-graph: 37,674 nodes, schema v11)
generated: 2026-04-30
status: planning-complete
---

# Real-World Company Architecture for Hermes Agent

> **Core premise**: The Hermes instance IS a company. You are the CEO. Every process a real company has — from OKR alignment to product delivery — must exist here. This document merges the three domain-knowledge docs with real-world gaps into a single coherent architecture.

## 1. Executive Summary

### 1.1 Current State

| Layer | Status |
|---|---|
| Role system | Built (RoleManager, 6 roles, XP/Level/Achievements) |
| KPI/Gamification | Built (KPITracker, add_xp, unlock_achievement) |
| Memory | Built partially (MemoryExtractor = flat JSONL, no vault) |
| Delegation | Built (delegate_task, 2532 lines, sub-agent spawning) |
| Planning | Built (PlanningEngine, should_plan + plan_and_execute) |
| Session persistence | Built (SessionDB, schema v11, FTS5) |
| Team hierarchy | Not built (docs only) |
| Skill tree | Not built (docs only) |
| Vault / Second Brain | Not built (docs only) |
| Three-tier vault (Company/Team/Role) | Not built (docs only) |
| OKR / Goal cascade | Not built, not in docs |
| Department layer | Not built, not in docs |
| Project layer | Not built, not in docs |
| Delivery pipeline | Not built, not in docs |
| External entities | Not built, not in docs |
| Information classification | Not built, not in docs |
| Identity lifecycle | Not built, not in docs |

### 1.2 Target: Full Company Operating System

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HERMES COMPANY (Your Instance)                    │
│                    CEO: You (human operator)                         │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                 STRATEGIC LAYER (CEO Dashboard)                │  │
│  │  Company OKRs  │  Budget  │  Roadmap  │  Company Health       │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  ┌───────────────┬───────────────────────┬────────────────────┐      │
│  │  Engineering  │  Quant & Trading     │  Content & Growth  │      │
│  │  Department   │  Department          │  Department        │      │
│  ├───────────────┤───────────────────────┤────────────────────┤      │
│  │ Dev Squad     │  Quant Desk           │  Content Studio    │      │
│  │ Infra Squad   │  Prop Desk            │  Marketing Squad   │      │
│  └───────────────┴───────────────────────┴────────────────────┘      │
│                              │                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                 DELIVERY LAYER                                 │  │
│  │  Projects  │  Releases  │  Staging  │  Production  │  Monitor │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                 EXTERNAL INTERFACE                             │  │
│  │  Customers  │  Vendors  │  Partners  │  Competitors           │  │
│  └────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. Organizational Model: Five-Layer Hierarchy

### 2.1 Layers

```
Layer 0: CEO (You)
  │  Sets company OKRs, approves strategic decisions, reviews dashboards
  │
Layer 1: Company
  │  Single source of truth: personality, policies, OKRs, brand
  │
Layer 2: Department / Division
  │  Groups related teams. Has department-level knowledge + OKRs
  │
Layer 3: Team
  │  Persistent unit with team OKRs, team vault, team XP
  │
Layer 4: Role (Individual Agent)
  │  Specialized unit with role OKRs, role vault, individual XP/Level/Skills
```

### 2.2 Why Five Layers (not three)

| v2 (3-tier) | Real-World Company | Why It Matters |
|---|---|---|
| Company → Team → Role | Company → Department → Team → Role | Engineering Dept knowledge (tech stack standards) is neither company-wide nor team-specific |
| No OKRs | OKR cascade at every layer | Without goals, XP is just grinding — no alignment |
| No delivery | Delivery pipeline | Company that never ships isn't a company |
| No external | Customer/Vendor/Partner | Company doesn't exist in a vacuum |

### 2.3 Data Model: Departments

```sql
CREATE TABLE IF NOT EXISTS departments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    head_role TEXT,                    -- Department head role (e.g., "system-engineer" for Engineering)
    dept_okr_id TEXT,                  -- Current quarter OKR set
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- Extend teams table with department_id
ALTER TABLE teams ADD COLUMN department_id TEXT REFERENCES departments(id);
```

Department vault: `~/.hermes/vaults/departments/<dept>/`

## 3. OKR / Goal Cascade System

### 3.1 OKR Hierarchy

```
Company OKR (quarterly, set by CEO)
  │
  ├── Department OKR (derived, 2-4 per quarter)
  │     │
  │     ├── Team OKR (derived, 2-3 per quarter)
  │     │     │
  │     │     └── Individual OKR (derived, 1-2 per quarter)
  │     │           └── Drives KPI weights dynamically
  │     │
  │     └── ...
  │
  └── ...
```

### 3.2 OKR Data Model

```sql
CREATE TABLE IF NOT EXISTS okr_sets (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,               -- 'company' | 'department' | 'team' | 'individual'
    scope_id TEXT NOT NULL,            -- department_id, team_id, or agent_id
    quarter TEXT NOT NULL,             -- '2026-Q2' format
    created_at REAL NOT NULL,
    UNIQUE(scope, scope_id, quarter)
);

CREATE TABLE IF NOT EXISTS objectives (
    id TEXT PRIMARY KEY,
    okr_set_id TEXT NOT NULL REFERENCES okr_sets(id) ON DELETE CASCADE,
    objective_text TEXT NOT NULL,
    weight REAL DEFAULT 1.0,            -- Relative weight within set
    status TEXT DEFAULT 'on_track',     -- 'on_track' | 'at_risk' | 'off_track' | 'done'
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS key_results (
    id TEXT PRIMARY KEY,
    objective_id TEXT NOT NULL REFERENCES objectives(id) ON DELETE CASCADE,
    kr_text TEXT NOT NULL,
    target_value REAL NOT NULL,
    current_value REAL DEFAULT 0,
    unit TEXT,                          -- 'count' | 'percentage' | 'hours' | 'xp' | 'revenue'
    status TEXT DEFAULT 'on_track',
    updated_at REAL NOT NULL
);
```

### 3.3 OKR → KPI Weight Dynamic Alignment

Current problem: KPI weights are role-static (defined in DEFAULT_ROLES). Fix:

```python
def get_effective_kpi_weights(role: str, team_id: str = None, department_id: str = None) -> dict:
    """Base KPI weights modified by current quarter OKR alignment."""
    base = DEFAULT_ROLES[role].kpi_weights.copy()

    # If individual OKRs exist for current quarter, boost aligned KPIs
    current_quarter = get_current_quarter()
    individual_okrs = get_active_okrs("individual", agent_id, current_quarter)
    for obj in individual_okrs:
        for kr in obj.key_results:
            if kr.unit == "kpi":
                base[kr.kpi_name] = base.get(kr.kpi_name, 0) * (1 + OKR_KPI_BOOST)

    return base
```

### 3.4 CEO OKR Commands

| Command | Description |
|---|---|
| `/okr set-company <quarter>` | Interactive: set company OKRs for the quarter |
| `/okr set-team <team> <quarter>` | Set team OKRs (auto-suggests from company OKRs) |
| `/okr set-individual <agent> <quarter>` | Set individual OKRs |
| `/okr review` | Show all OKR progress for current quarter |
| `/okr dashboard` | CEO dashboard: company health + OKR progress + delivery status |

## 4. Five-Tier Vault Architecture

### 4.1 Directory Structure

```
~/.hermes/
├── vaults/
│   ├── company/                          # TIER 1: Company LTM (global)
│   │   ├── INDEX.md
│   │   ├── personality/                 # D1: User personality (CEO preferences)
│   │   ├── policies/                    # D5: Company policies, security, standards
│   │   ├── decisions/                   # D4: Strategic ADRs (with provenance)
│   │   ├── shared-patterns/             # D6: Cross-department reusable patterns
│   │   ├── brand/                       # D-NEW: Brand voice, public identity
│   │   └── experience/                  # D4: Company-level post-mortems
│   │
│   ├── departments/                     # TIER 2: Department LTM
│   │   ├── engineering/
│   │   │   ├── INDEX.md
│   │   │   ├── standards/              # Tech stack, coding conventions
│   │   │   ├── decisions/              # Dept-level ADRs
│   │   │   └── experience/
│   │   ├── quant-trading/
│   │   └── content-growth/
│   │
│   ├── teams/                           # TIER 3: Team LTM (from v2 doc)
│   │   ├── dev-squad/
│   │   │   ├── INDEX.md
│   │   │   ├── decisions/
│   │   │   ├── cross-role/
│   │   │   └── experience/
│   │   └── quant-desk/
│   │
│   ├── projects/                        # TIER 3.5: Project LTM (time-bounded)
│   │   ├── hermes-v2-launch/
│   │   │   ├── INDEX.md
│   │   │   ├── charter.md              # Project scope, sponsor, timeline
│   │   │   ├── decisions/
│   │   │   ├── status/                 # Weekly status summaries
│   │   │   └── retrospective.md        # Created on project close
│   │   └── trading-dashboard/
│   │
│   └── roles/                           # TIER 4: Role LTM (from v2 doc)
│       ├── fullstack-dev/
│       ├── devops/
│       ├── quant-trader/
│       ├── propfirm-trader/
│       ├── content-creator/
│       └── system-engineer/
│
├── entities/                            # TIER 5: External Entities
│   ├── customers/
│   │   ├── INDEX.md
│   │   └── <customer-id>/
│   │       ├── profile.md
│   │       ├── feedback.md
│   │       └── contracts.md
│   ├── vendors/
│   ├── partners/
│   └── competitors/
│
├── sessions/                            # STM (session-scoped, unchanged)
└── memories/                             # Legacy (migration source)
```

### 4.2 Information Classification

Every vault note carries a `classification` field in frontmatter:

```yaml
---
type: ltm
dimension: decisions
classification: internal       # public | internal | confidential | restricted
owner: dev-squad               # Who is responsible for this note
review_by: 2026-07-30         # Next review date
decided_by: ceo                # Who made this decision
consulted: [dev-lead, quant-lead]  # Who was consulted
supersedes: null               # Previous decision this replaces
status: accepted               # proposed | accepted | superseded | deprecated
---
```

| Classification | Who Can Read | Who Can Write | Example |
|---|---|---|---|
| `public` | Everyone + external | Department head | Brand voice, public docs |
| `internal` | All company members | Team lead+ | Coding standards, OKRs, ADRs |
| `confidential` | Need-to-know (tagged roles/teams) | Department head | Customer data, HR, compensation |
| `restricted` | Specific clearance list | CEO only | API keys, trade secrets, legal |

### 4.3 Decision Provenance (ADR Metadata)

All decisions (company, department, team) carry formal provenance:

```yaml
---
type: ltm
dimension: decisions
classification: internal
decided_by: ceo
decided_at: 2026-04-30T10:00:00Z
consulted: [dev-lead, system-engineer]
inputs_considered: [performance-benchmark, team-capacity]
review_by: 2026-10-30
supersedes: null
status: accepted
---
```

Decision lifecycle: `proposed` → `accepted` → (optionally) `superseded` or `deprecated`

### 4.4 Knowledge Stewardship

| Field | Purpose | Auto-flag |
|---|---|---|
| `owner` | Agent or team responsible | Missing owner → lint warning |
| `review_by` | Date for next review | Past due → `/memory lint` flags |
| `review_cadence` | 30d / 90d / 180d / annual | Auto-set based on classification |
| `last_reviewed_at` | Last review timestamp | > review_cadence → stale warning |

## 5. Project Layer (Time-Bounded Initiatives)

### 5.1 Why Projects Matter

Teams are persistent, but real companies execute through **projects** — time-bounded initiatives that may span multiple teams. Without projects:
- There's no concept of "shipping" something
- No timeline, no scope, no retrospective
- Delivery tracking is impossible

### 5.2 Project Data Model

```sql
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    status TEXT DEFAULT 'proposed',       -- proposed | active | on_hold | completed | cancelled
    sponsor_department_id TEXT REFERENCES departments(id),
    priority TEXT DEFAULT 'medium',       -- critical | high | medium | low
    target_date REAL,                     -- Unix timestamp
    started_at REAL,
    completed_at REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS project_teams (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    team_id TEXT NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    role TEXT DEFAULT 'contributor',       -- contributor | lead | reviewer
    PRIMARY KEY (project_id, team_id)
);

CREATE TABLE IF NOT EXISTS project_milestones (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    status TEXT DEFAULT 'pending',         -- pending | in_progress | done | blocked
    target_date REAL,
    completed_at REAL,
    sort_order INTEGER DEFAULT 0
);
```

### 5.3 Project Lifecycle

```
proposed → active → (on_hold → active) → completed
                 └→ cancelled

On completion:
  1. Auto-create retrospective in vault (via ConsolidationEngine)
  2. Archive project vault (read-only)
  3. Promote project decisions to team/company vault
  4. Update team OKR progress
  5. Award project-completion XP to contributors
```

### 5.4 Project Commands

| Command | Description |
|---|---|
| `/project create <name>` | Create project (interactive) |
| `/project list` | List all projects with status |
| `/project show <name>` | Show project details + milestones |
| `/project milestone <name>` | Add/complete milestone |
| `/project complete <name>` | Close project, trigger retrospective + archive |
| `/project assign <name> <team>` | Assign team to project |

## 6. Delivery Pipeline

### 6.1 Why This Matters

A company that never delivers isn't a company. The delivery pipeline connects internal work to external value:

```
Internal Work → Build → Test → Stage → Ship → Monitor → Learn
     │                                              │
     └────────────────── Feedback Loop ◄────────────┘
```

### 6.2 Release Data Model

```sql
CREATE TABLE IF NOT EXISTS releases (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    version TEXT NOT NULL,                -- semver: 1.2.0
    status TEXT DEFAULT 'draft',          -- draft | testing | staged | shipped | rolled_back
    changelog TEXT,
    created_at REAL NOT NULL,
    shipped_at REAL,
    UNIQUE(project_id, version)
);

CREATE TABLE IF NOT EXISTS release_checks (
    id TEXT PRIMARY KEY,
    release_id TEXT NOT NULL REFERENCES releases(id) ON DELETE CASCADE,
    check_type TEXT NOT NULL,             -- test_suite | security_scan | performance | manual_review
    status TEXT DEFAULT 'pending',        -- pending | passed | failed | skipped
    result_text TEXT,
    checked_at REAL
);
```

### 6.3 Release Lifecycle

```
draft → testing → staged → shipped
  │                              │
  └→ rolled_back ◄──────────────┘

Pre-ship gates (configurable per project):
  1. Test suite: all tests pass
  2. Security scan: no P0/P1 findings
  3. Performance: no regression > 10%
  4. Manual review: department head sign-off
```

### 6.4 Delivery Commands

| Command | Description |
|---|---|
| `/release create <project> <version>` | Create release draft |
| `/release check <release>` | Run all checks |
| `/release ship <release>` | Ship (if all checks pass) |
| `/release rollback <release>` | Roll back (triggers incident) |
| `/release changelog <release>` | Generate changelog from commits + project decisions |

### 6.5 Monitoring + Incident Response

```sql
CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    release_id TEXT REFERENCES releases(id),
    severity TEXT NOT NULL,               -- P0 | P1 | P2 | P3
    status TEXT DEFAULT 'open',          -- open | investigating | mitigated | resolved
    description TEXT,
    opened_at REAL NOT NULL,
    resolved_at REAL,
    resolution TEXT
);
```

| Command | Description |
|---|---|
| `/incident open <severity> <description>` | Open incident |
| `/incident update <id>` | Update status |
| `/incident resolve <id>` | Resolve + auto-create post-mortem in vault |

## 7. External Entities

### 7.1 Why External Matters

Real companies don't exist in a vacuum. Hermes as a company needs to track:
- **Customers**: Who uses your product, what they think, contracts
- **Vendors**: Who supplies your infrastructure (API providers, hosting)
- **Partners**: Who collaborates with you
- **Competitors**: What the market looks like (restricted classification)

### 7.2 External Entity Data Model

```sql
CREATE TABLE IF NOT EXISTS external_entities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,           -- customer | vendor | partner | competitor
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    classification TEXT DEFAULT 'internal',  -- public | internal | confidential | restricted
    status TEXT DEFAULT 'active',        -- active | inactive | churned (for customers)
    metadata TEXT,                       -- JSON: type-specific fields
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_interactions (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES external_entities(id) ON DELETE CASCADE,
    interaction_type TEXT NOT NULL,      -- feedback | contract | issue | meeting | delivery
    summary TEXT NOT NULL,
    classification TEXT DEFAULT 'internal',
    created_at REAL NOT NULL
);
```

### 7.3 Entity Commands

| Command | Description |
|---|---|
| `/entity add <type> <name>` | Add external entity |
| `/entity list [type]` | List entities (filterable by type) |
| `/entity interact <name>` | Record interaction/feedback |
| `/entity show <name>` | Show entity profile + interaction history |

## 8. Identity & Membership Lifecycle

### 8.1 Lifecycle Stages

```
onboarding → active → (transfer → active) → alumni
```

| Stage | Vault Access | Write Access | Special |
|---|---|---|---|
| `onboarding` | Company + Department + Team INDEX only | None (learning phase) | Buddy assigned, reading list |
| `active` | Full per-depth rules | Per-depth write rules | Normal operation |
| `transfer` | Old team (read) + New team (onboarding) | New team only | Knowledge carry-over |
| `alumni` | Company LTM (read only) | None | Legacy contributions preserved |

### 8.2 Membership Fields (extend team_members)

```sql
ALTER TABLE team_members ADD COLUMN status TEXT DEFAULT 'active';
-- onboarding | active | transfer | alumni

ALTER TABLE team_members ADD COLUMN buddy_id TEXT REFERENCES team_members(id);
-- Assigned buddy during onboarding

ALTER TABLE team_members ADD COLUMN onboarded_at REAL;
-- When onboarding completed

ALTER TABLE team_members ADD COLUMN department_id TEXT REFERENCES departments(id);
-- Multi-team membership: same agent in multiple teams (different rows)
```

### 8.3 Multi-Team Membership (Matrix Org)

An agent can belong to multiple teams. Each membership is a separate row:

```python
def get_agent_teams(self, agent_id: str) -> list[dict]:
    """Return all team memberships for an agent (matrix org support)."""
    return self.db.query(
        "SELECT tm.*, t.name as team_name FROM team_members tm "
        "JOIN teams t ON tm.team_id = t.id "
        "WHERE tm.agent_id = ? AND tm.status != 'alumni'",
        (agent_id,)
    )
```

### 8.4 Onboarding Protocol

```python
def onboarding_context(self, agent_id: str, team_id: str) -> str:
    """Build onboarding reading list for new agent."""
    context = ""
    context += self._load_company_ltm()  # Company policies (always)
    context += self._load_department_ltm(team.department_id)
    context += self._load_team_ltm(team_id, member_depth=99)  # Full team context
    context += "\n# Onboarding Reading List\n"
    context += "- Read company policies in full\n"
    context += "- Review team INDEX.md\n"
    context += f"- Your buddy: {buddy.agent_id} ({buddy.role})\n"
    return context
```

## 9. Consolidation Pipeline: Five-Tier Routing

### 9.1 Extended Routing Decision Tree

```
STM Entry
  │
  ├─ Is this about CEO personality/preferences?
  │   └─→ Company LTM → personality/
  │
  ├─ Is this a company-wide policy or guardrail?
  │   └─→ Company LTM → policies/
  │
  ├─ Is this a strategic decision affecting the whole company?
  │   └─→ Company LTM → decisions/ (with ADR provenance)
  │
  ├─ Is this a cross-department reusable pattern?
  │   └─→ Company LTM → shared-patterns/
  │
  ├─ Is this about brand, public identity, or external-facing?
  │   └─→ Company LTM → brand/
  │
  ├─ Is this a department-level standard or convention?
  │   └─→ Department LTM → standards/ or decisions/
  │
  ├─ Is this a team-specific decision or pattern?
  │   └─→ Team LTM → decisions/ or cross-role/
  │
  ├─ Is this a project-level decision or status update?
  │   └─→ Project LTM → decisions/ or status/
  │
  ├─ Is this a customer/vendor/partner interaction?
  │   └─→ External Entity → entity_interactions
  │
  ├─ Is this a role-specific technical fact?
  │   └─→ Role LTM → hard-skill/
  │
  ├─ Is this about communication/collaboration style?
  │   └─→ Role LTM → soft-skill/
  │
  └─ Ambiguous?
      └─→ Default to Role LTM (lowest blast radius)
```

### 9.2 Promotion System (Extended)

```
Role LTM → Team LTM → Department LTM → Company LTM

Triggers:
  Role → Team:         2+ roles in same team reference the note
  Team → Department:   2+ teams in same department reference the note
  Department → Company: 2+ departments reference the note, OR CEO manual promote

All promotions require:
  - Classification review (may need to upgrade sensitivity)
  - Owner assignment (if missing)
  - Review date set (if missing)
```

## 10. Unified Prompt Assembly (Full Company Context)

### 10.1 Agent Boot Context

```python
def build_full_company_context(
    role: str,
    session_id: str,
    skill_tree: SkillTree = None,
    team_id: str = None,
    department_id: str = None,
    member_depth: int = 99,
    project_id: str = None,
) -> str:
    """Load all LTM tiers at session start."""

    context = ""

    # TIER 1: Company LTM — always loaded, no skill gates
    context += _load_company_ltm()

    # TIER 2: Department LTM — loaded if agent is in a department
    if department_id:
        context += _load_department_ltm(department_id)

    # TIER 3: Team LTM — loaded if agent is in a team
    if team_id:
        context += _load_team_ltm(team_id, member_depth, skill_tree)

    # TIER 3.5: Project LTM — loaded if agent is on an active project
    if project_id:
        context += _load_project_ltm(project_id)

    # TIER 4: Role LTM — loaded for the agent's role, skill-gated
    context += _load_role_ltm(role, skill_tree)

    # Current OKRs (affects KPI weights and priorities)
    context += _load_current_okrs(role, team_id, department_id)

    # Vault access manifest (what this agent can see)
    accessible = _list_accessible_vault_paths(
        role, skill_tree, team_id, department_id, member_depth
    )
    context += f"\n# Vault Access\n{accessible}\n"

    return context
```

### 10.2 System Prompt for delegate_task (Extended)

```
{existing_system_prompt}

# ═══ Company Knowledge (Global) ═══
{build_company_ltm_context()}

# ═══ Department Context ═══
You are in the **{department_name}** department.
{build_department_ltm_context(department_id)}

# ═══ Team Context ═══
You are **{agent_id}** on team **{team_name}**.
Your position: {position} (depth={depth})
Team Level: {team_level} | Team XP: {team_xp}/{xp_to_next}

{build_team_ltm_context(team_id, member_depth)}

# ═══ Project Context ═══
Active project: **{project_name}** (status: {project_status})
Milestones: {project_milestones}
{build_project_ltm_context(project_id)}

# ═══ Role Knowledge (Skill-Gated) ═══
{build_role_ltm_context(role, skill_tree)}
{role_manager.build_role_system_prompt(role)}
{skill_tree.apply_unlocked_prompts(role)}

# ═══ Current OKRs ═══
{render_current_okrs(role, team_id, department_id)}

# ═══ Skill Tree Status ═══
{rendered_skill_tree_for_role}

# ═══ Vault Access ═══
Company: full read access
Department: full read, {write_dept} write
Team: {"read" if depth >= 0 else "none"} | Write: {"yes" if depth <= 1 else "no"}
Role: full read (skill-gated sections unlocked as shown above)
Classification ceiling: {max_classification}
```

## 11. CEO Dashboard

### 11.1 `/company dashboard` — The CEO's Command Center

```
╭─ Hermes Company — CEO Dashboard ──────────────────────────────────╮
│ Company Level: 12 | Total XP: 1,150                              │
│ Quarter: 2026-Q2 | OKR Progress: 67%                             │
│ Active Projects: 3 | Releases This Quarter: 2                    │
│ Open Incidents: 1 (P2) | Delivery Health: GREEN                  │
╰───────────────────────────────────────────────────────────────────╯

  OKRs (Q2):
  ├── O1: Launch Hermes V2          [████████░░] 80%
  │   ├── KR1: 95% test coverage    [█████████░] 90%
  │   ├── KR2: Ship by June 30     [██████░░░░] 60%
  │   └── KR3: 50+ vault notes     [███████░░░] 70%
  ├── O2: Quant Trading Alpha       [████░░░░░░] 40%
  └── O3: Customer Growth           [██░░░░░░░░] 20%

  Departments:
  ├── Engineering (3 teams, 8 members, Lv.7)     ████████░░
  ├── Quant & Trading (2 teams, 5 members, Lv.4) ████░░░░░░
  └── Content & Growth (1 team, 3 members, Lv.2) ██░░░░░░░░

  Active Projects:
  ├── hermes-v2-launch (Engineering)     [███████░░] 70%
  ├── trading-dashboard (Quant)           [████░░░░░] 40%
  └── brand-website (Content)            [██░░░░░░░] 20%

  Delivery Pipeline:
  ├── v1.8.2 (shipped 2026-04-28)       ✓ all checks passed
  └── v1.9.0-rc1 (staged)               ✓ tests, ⟳ security scan

  External:
  ├── Customers: 2 active
  ├── Vendors: 3 (Anthropic, OpenAI, Modal)
  └── Incidents: 1 open (P2: slow API response)
```

### 11.2 CEO Commands Summary

| Command | Description |
|---|---|
| `/company dashboard` | Full CEO dashboard |
| `/company info` | Company stats (XP, teams, vault size) |
| `/company vault` | Company vault INDEX |
| `/company promote <note>` | Promote note to company level |
| `/company achievements` | Company-level achievements |
| `/okr set-company <quarter>` | Set company OKRs |
| `/okr review` | OKR progress review |
| `/project create <name>` | Create new project |
| `/release ship <release>` | Ship a release |
| `/incident open <severity>` | Open incident |
| `/entity add <type> <name>` | Add external entity |
| `/department create <name>` | Create department |
| `/department assign <team>` | Assign team to department |

## 12. Implementation Phases

### Phase 1: Foundation (Schema + Core Classes)
**Goal**: Data model + core classes exist, tests pass

1. **hermes_state.py**: Bump SCHEMA_VERSION to 12
   - Add tables: `departments`, `teams` (with `department_id`), `team_members` (with `status`, `buddy_id`, `department_id`), `skill_trees` (with `vault_unlocks`), `team_achievements`, `company_vault_notes`, `company_achievements`
   - Add tables: `okr_sets`, `objectives`, `key_results`
   - Add tables: `projects`, `project_teams`, `project_milestones`
   - Add tables: `releases`, `release_checks`
   - Add tables: `incidents`
   - Add tables: `external_entities`, `entity_interactions`
   - **Verify**: `scripts/run_tests.sh` passes with new schema

2. **agent/team_manager.py**: TeamManager class (from team-hierarchy doc)
   - Team CRUD, member management, hierarchy, team XP, team achievements
   - Extended: `department_id`, `status` lifecycle, multi-team membership

3. **agent/skill_tree.py**: SkillTree class (from team-hierarchy doc)
   - Default skill trees, seed/unlock/apply logic
   - Extended: `vault_unlocks` across all 5 tiers (from v2 doc)

4. **agent/department_manager.py**: DepartmentManager class (NEW)
   - Department CRUD, team assignment, department OKRs

5. **agent/okr_manager.py**: OKRManager class (NEW)
   - OKR CRUD at all levels, progress tracking, KPI weight alignment

### Phase 2: Memory + Vault
**Goal**: Second Brain vault system works, consolidation pipeline runs

6. **agent/vault_memory_provider.py**: VaultMemoryProvider (from second-brain doc)
   - Extends MemoryProvider, role-scoped vault, INDEX.md auto-generation

7. **agent/consolidation_engine.py**: ConsolidationEngine (from second-brain + v2 doc)
   - 5-stage pipeline: Evaluate → Summarize → Merge → Link → Tag
   - Extended: Five-tier routing (company → department → team → project → role)
   - Extended: Classification-aware (writes correct classification level)
   - Extended: ADR provenance for decision notes

8. **agent/vault_linter.py**: VaultLinter (from second-brain doc)
   - Orphan detection, stale detection, broken links, classification gaps
   - Extended: Owner audit, review_cadence checks, ADR completeness

9. **agent/memory_migrator.py**: MemoryMigrator (from second-brain doc)
   - Claude Code memory → vault, legacy JSONL → vault

### Phase 3: Project + Delivery
**Goal**: Projects exist, releases can be shipped

10. **agent/project_manager.py**: ProjectManager class (NEW)
    - Project CRUD, milestone tracking, team assignment, lifecycle

11. **agent/release_manager.py**: ReleaseManager class (NEW)
    - Release CRUD, check gates, ship/rollback, changelog generation

12. **agent/incident_manager.py**: IncidentManager class (NEW)
    - Incident CRUD, severity tracking, auto-post-mortem on resolve

### Phase 4: External Entities
**Goal**: Company can track and interact with external parties

13. **agent/entity_manager.py**: EntityManager class (NEW)
    - External entity CRUD, interaction logging, classification-aware access

### Phase 5: Gamification Integration
**Goal**: XP triggers three-tier consolidation, OKRs drive KPI weights

14. **agent/gamification.py**: Extend `record_session_and_award_xp()` (from team-hierarchy doc)
    - Chain: KPI → XP → team XP → skill unlock → three-tier consolidation
    - Extended: Department XP aggregation, project completion XP

15. **agent/prompt_builder.py**: Integrate full company context into boot
    - `build_full_company_context()` with all 5 tiers + OKRs + project context

### Phase 6: CLI + UX
**Goal**: CEO and agents can use all features from CLI

16. **hermes_cli/commands.py**: Add CommandDefs for all new commands
    - `/company`, `/department`, `/okr`, `/project`, `/release`, `/incident`, `/entity`
    - Extended: `/team` commands (from team-hierarchy doc)
    - Extended: `/memory consolidate`, `/memory lint`, `/memory promote`

17. **cli.py**: Implement all command handlers
18. **tui_gateway/server.py**: Add RPC methods for all new features
19. **web/src/pages/**: Add CEO dashboard page, project pages, OKR pages

### Phase 7: Delegate Integration
**Goal**: Sub-agents receive full company context

20. **tools/delegate_tool.py**: Add `team_context`, `department_context`, `project_context`
    - Extended system prompt with all tiers + OKRs + project context

### Phase 8: Tests
**Goal**: Full test coverage for all new features

21. **tests/integration/test_team_hierarchy.py**: Team CRUD, hierarchy, XP
22. **tests/integration/test_skill_tree.py**: Skill unlock, vault_unlocks
23. **tests/integration/test_okr_cascade.py**: OKR at all levels, KPI weight alignment
24. **tests/integration/test_project_lifecycle.py**: Project CRUD, milestones, completion
25. **tests/integration/test_release_pipeline.py**: Release checks, ship, rollback
26. **tests/integration/test_incident_response.py**: Incident open → resolve → post-mortem
27. **tests/integration/test_entity_management.py**: External entity CRUD, interactions
28. **tests/integration/test_consolidation_five_tier.py**: Five-tier routing, classification
29. **tests/integration/test_vault_lint.py**: Lint checks, owner audit, review cadence
30. **tests/integration/test_identity_lifecycle.py**: Onboarding, transfer, alumni

## 13. File Changes Summary

| File | Change Type | Phase | Description |
|---|---|---|---|
| `hermes_state.py` | Modify | 1 | Schema v12, 13 new tables |
| `agent/team_manager.py` | New | 1 | TeamManager |
| `agent/skill_tree.py` | New | 1 | SkillTree + vault_unlocks |
| `agent/department_manager.py` | New | 1 | DepartmentManager |
| `agent/okr_manager.py` | New | 1 | OKRManager |
| `agent/vault_memory_provider.py` | New | 2 | VaultMemoryProvider |
| `agent/consolidation_engine.py` | New | 2 | Five-tier ConsolidationEngine |
| `agent/vault_linter.py` | New | 2 | VaultLinter |
| `agent/memory_migrator.py` | New | 2 | MemoryMigrator |
| `agent/project_manager.py` | New | 3 | ProjectManager |
| `agent/release_manager.py` | New | 3 | ReleaseManager |
| `agent/incident_manager.py` | New | 3 | IncidentManager |
| `agent/entity_manager.py` | New | 4 | EntityManager |
| `agent/gamification.py` | Modify | 5 | Extended XP + consolidation trigger |
| `agent/prompt_builder.py` | Modify | 5 | Full company context in boot |
| `hermes_cli/commands.py` | Modify | 6 | All new CommandDefs |
| `cli.py` | Modify | 6 | All new command handlers |
| `tui_gateway/server.py` | Modify | 6 | All new RPC methods |
| `tools/delegate_tool.py` | Modify | 7 | Full company context injection |
| `agent/default_teams/` | New | 1 | 3 built-in team templates |
| `agent/default_departments/` | New | 1 | 3 built-in department templates |
| 10 test files | New | 8 | Integration tests |

## 14. Key Design Decisions

### 14.1 Why Five Tiers (not three)?

v2 doc proposed Company → Team → Role. We add Department and Project because:
- **Department**: Engineering standards are neither company-wide nor team-specific. Without departments, every engineering standard must be either company-level (too broad) or team-level (too narrow, duplicated across teams).
- **Project**: Teams are persistent, but work happens in time-bounded projects. Without projects, there's no concept of delivery, milestones, or retrospectives.

### 14.2 Why OKRs Drive KPI Weights (not role-static)?

Role-static KPI weights mean a quant-trader always optimizes the same metrics regardless of whether the company goal is "launch V2" or "maximize alpha." OKRs align every agent's incentives with the company's current priorities.

### 14.3 Why Information Classification?

"Company LTM อ่านได้ทุกคน" is wrong for a real company. Customer data, financials, and trade secrets need restricted access. Classification is a non-optional safety mechanism.

### 14.4 Why External Entities?

A company without customers isn't a company. Tracking external interactions enables:
- Customer feedback → product improvements (consolidated into project/team vault)
- Vendor issues → incident response
- Competitor intelligence → strategic decisions

### 14.5 Why Delivery Pipeline?

Internal work without delivery is just activity, not productivity. The release pipeline with checks + incidents creates a complete ship → monitor → learn loop.

### 14.6 Why Identity Lifecycle?

Onboarding ensures new agents don't start cold. Transfer preserves knowledge continuity. Alumni ensures former contributions aren't lost. Without these, every new agent is a blank slate.

## 15. Migration from Current State

### 15.1 Schema Migration (v11 → v12)

```python
def migrate_v11_to_v12(db: SessionDB) -> None:
    """Add all new tables. Existing v11 tables unchanged."""
    # Create all new tables (IF NOT EXISTS handles idempotency)
    # ALTER TABLE teams ADD COLUMN department_id (if not exists)
    # ALTER TABLE team_members ADD COLUMN status, buddy_id, department_id
    # Seed default departments
    # Seed default skill trees
    # Rebuild vault INDEX.md files
```

### 15.2 Vault Migration (Flat → Five-Tier)

```python
def migrate_flat_to_five_tier(hermes_home: Path) -> None:
    """Migrate v2 three-tier vault to five-tier with departments + projects."""
    vaults = hermes_home / "vaults"

    # 1. Create new directories
    (vaults / "company").mkdir(exist_ok=True)
    (vaults / "departments").mkdir(exist_ok=True)
    (vaults / "teams").mkdir(exist_ok=True)
    (vaults / "projects").mkdir(exist_ok=True)
    (vaults / "roles").mkdir(exist_ok=True)

    # 2. Move personality from roles to company (v2 already does this)
    # 3. Move role vaults to roles/
    # 4. Create department vaults from team membership
    # 5. Rebuild all INDEX.md files
```

### 15.3 Backward Compatibility

- All existing RoleManager, KPITracker, MemoryExtractor code works unchanged
- New features are additive — old sessions still work
- New `/company`, `/okr`, `/project` commands are opt-in
- Departments default to "General" if not configured (no breaking change)

## 16. Risk & Mitigation

| Risk | Mitigation |
|---|---|
| Context window overflow with 5 vault tiers | Company LTM loaded as INDEX.md only (personality + policies full). Other tiers: INDEX.md only, full notes on-demand |
| 13 new tables is a big migration | Schema v12 is additive only — no breaking changes to existing tables. IF NOT EXISTS handles idempotency |
| Too many CLI commands overwhelms user | CEO commands are in `/company dashboard`. Other commands auto-discover context (no need to specify department if only in one) |
| OKR system too bureaucratic for small setups | OKRs are optional. Without OKRs, KPI weights fall back to role-static defaults |
| Delivery pipeline overkill for internal tools | Release checks are configurable per project. Simple projects can skip gates |
| Consolidation routing too complex with 5 tiers | LLM classification with tier-aware prompt. Fallback to role-level if confidence < 0.7 |
| External entity tracking seems premature | Entities are optional. Start with customers only, add vendors/partners/competitors when needed |