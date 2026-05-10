---
scope: Integration synthesis v2 — Company-centric memory hierarchy (Company > Team > Role)
audience: Implementation agent (next session)
sources:
  - domain-knowledge/agent-team-hierarchy-gamification.md
  - domain-knowledge/second-brain-memory-architecture.md
  - domain-knowledge/team-memory-integration-synthesis.md (v1)
generated: 2026-04-30
status: planning-complete
---

# Company-Centric Memory Hierarchy — Integration Synthesis v2

> **Key shift from v1**: The computer running Hermes is **one company**. All LTM belongs to the company. Teams are organizational units within the company, not independent knowledge silos. This changes the vault architecture from "team vault on top" to a **three-tier hierarchy: Company → Team → Role**, where knowledge flows downward and upward through the org chart.

## 1. Executive Summary

Hermes instance = Company. ทุกคนใน company แชร์ LTM รากฐานเดียวกัน (user personality, company policies, project history) Teams เป็นหน่วยย่อยที่มี team-specific knowledge (decisions, patterns, experience) และ Roles เป็นหน่วยปลายทางที่มี role-specific hard-skill knowledge

บูรณาการแล้ว จะได้ระบบที่:

- **Company LTM** เป็น single source of truth — user personality, company decisions, shared patterns อยู่ที่นี่
- **Team LTM** เป็น scoped subset — team-specific decisions, cross-role patterns, team experience
- **Role LTM** เป็น specialized layer — role-specific hard-skill, soft-skill knowledge
- XP award triggers consolidation ที่รู้จัก route STM entries ไปยัง Company/Team/Role อย่างถูกต้อง
- Skill tree unlocks ขยาย vault access ทุกระดับ (ไม่ใช่แค่ role vault)
- Team hierarchy กำหนด memory access pattern — Lead เห็นกว้างกว่า Worker
- Company-level achievements คือ LTM notes ที่ถูก consolidate อัตโนมัติ

## 2. The Organizational Model

### 2.1 Company = Hermes Instance

```
┌─────────────────────────────────────────────────────────────────┐
│                        COMPANY (Hermes Instance)                 │
│                    ~/.hermes/vaults/company/                     │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │              Company LTM (Single Source of Truth)          │   │
│  │                                                           │   │
│  │  personality/     User personality, preferences, style    │   │
│  │  policies/       Company policies, standards, guardrails  │   │
│  │  decisions/      Strategic decisions, architecture ADRs  │   │
│  │  experience/      Project post-mortems, lessons learned    │   │
│  │  shared-patterns/ Cross-team patterns, reusable solutions │   │
│  │                                                           │   │
│  │  All roles and teams inherit this knowledge               │   │
│  └───────────────────────────────────────────────────────────┘   │
│                              │                                    │
│              ┌───────────────┼───────────────┐                   │
│              ▼               ▼               ▼                   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Team: Dev     │ │  Team: Quant    │ │ Team: Content   │   │
│  │   Squad         │ │  Desk           │ │ Studio           │   │
│  │                 │ │                 │ │                 │   │
│  │ Team LTM:       │ │ Team LTM:       │ │ Team LTM:       │   │
│  │ - API contracts │ │ - Risk models  │ │ - Brand voice   │   │
│  │ - Sprint goals  │ │ - Backtest logs │ │ - Style guides  │   │
│  │ - Dev decisions │ │ - Quant decisions│ │ - Content plan  │   │
│  │                 │ │                 │ │                 │   │
│  │ ┌─────┐ ┌────┐ │ │ ┌─────┐ ┌────┐ │ │ ┌─────┐        │   │
│  │ │ Role│ │Role│ │ │ │Role│ │Role│ │ │ │Role│        │   │
│  │ │ FS- │ │Dev│ │ │ │Quant│ │Prop│ │ │ │Cont│        │   │
│  │ │ Dev │ │Ops│ │ │ │Trader│ │Firm│ │ │ │Creator│      │   │
│  │ └─────┘ └────┘ │ │ └─────┘ └────┘ │ │ └─────┘        │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│                                                                  │
│  Role LTM (per role, inherited by all teams):                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │fullstack-│ │  devops   │ │quant-    │ │content-  │         │
│  │  dev     │ │          │ │ trader   │ │creator   │         │
│  │ hard-skill│ │ hard-skill│ │ hard-skill│ │ hard-skill│      │
│  │ soft-skill│ │ soft-skill│ │ soft-skill│ │ soft-skill│      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Knowledge Flow Direction

```
                    Company LTM
                    (Global Truth)
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
        Team LTM      Team LTM     Team LTM
      (dev-squad)  (quant-desk)  (content-studio)
            │            │            │
        ┌───┴───┐    ┌──┴──┐     ┌──┴──┐
        ▼       ▼    ▼     ▼     ▼
     Role LTM  Role LTM  Role LTM
     (shared across teams — same role in different teams
      shares the same hard-skill knowledge base)

Consolidation routes UP:
  STM entry → evaluate → determine scope (company/team/role) → merge into appropriate LTM

Boot loads DOWN:
  Company LTM (always) → Team LTM (if in team) → Role LTM (always) → Skill-gated sections
```

## 3. Three-Tier Vault Architecture

### 3.1 Directory Structure

```
~/.hermes/
├── vaults/
│   ├── company/                          # TIER 1: Company LTM (global)
│   │   ├── INDEX.md                      # Master index
│   │   ├── personality/
│   │   │   └── user-personality.md       # D1: User personality (shared across ALL)
│   │   ├── policies/
│   │   │   ├── coding-standards.md       # Company coding standards
│   │   │   ├── security-policy.md         # Security guardrails
│   │   │   └── deployment-process.md      # How deploys work at this company
│   │   ├── decisions/
│   │   │   ├── why-fastapi.md             # ADR: why FastAPI over Flask
│   │   │   ├── monorepo-choice.md          # ADR: monorepo structure
│   │   │   └── database-choice.md          # ADR: SQLite for state
│   │   ├── experience/
│   │   │   ├── hermes-v1-launch.md         # Project post-mortem
│   │   │   └── security-remediation.md     # Major incident lessons
│   │   └── shared-patterns/
│   │       ├── error-handling.md           # Cross-team error patterns
│   │       └── testing-isolation.md         # _isolate_hermes_home pattern
│   │
│   ├── teams/                            # TIER 2: Team LTM (scoped)
│   │   ├── dev-squad/
│   │   │   ├── INDEX.md
│   │   │   ├── decisions/
│   │   │   │   └── frontend-backend-contract.md
│   │   │   ├── cross-role/
│   │   │   │   └── api-integration-patterns.md
│   │   │   └── experience/
│   │   │       └── v1-launch-postmortem.md
│   │   └── quant-desk/
│   │       ├── INDEX.md
│   │       ├── decisions/
│   │       └── experience/
│   │
│   └── roles/                            # TIER 3: Role LTM (specialized)
│       ├── fullstack-dev/
│       │   ├── INDEX.md
│       │   ├── hard-skill/
│       │   │   ├── fastapi-patterns.md
│       │   │   ├── react-dashboard.md
│       │   │   └── test-isolation.md
│       │   └── soft-skill/
│       │       ├── error-communication.md
│       │       └── plan-then-execute.md
│       ├── devops/
│       ├── quant-trader/
│       ├── propfirm-trader/
│       ├── content-creator/
│       └── system-engineer/
│
├── sessions/                             # STM (session-scoped, unchanged)
└── memories/                             # Legacy (migration source)
```

**สิ่งที่เปลี่ยนจาก v1:**
- `vaults/` มี 3 sub-directories: `company/`, `teams/`, `roles/`
- `personality/` ย้ายจาก role vault ไป company vault (เพราะ user personality เป็นของทั้ง company)
- `decisions/` แบ่งเป็น company-level (strategic) และ team-level (tactical)
- `shared-patterns/` เป็น tier ใหม่ — cross-team reusable patterns

### 3.2 Data Model: Three-Tier Vault Notes

```sql
-- Company vault notes (global, always loaded)
CREATE TABLE IF NOT EXISTS company_vault_notes (
    id TEXT PRIMARY KEY,
    note_path TEXT NOT NULL UNIQUE,       -- e.g., 'decisions/why-fastapi.md'
    dimension TEXT NOT NULL,              -- 'personality' | 'policies' | 'decisions' | 'experience' | 'shared-patterns'
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    backlinks TEXT,                       -- JSON array of [[wiki-link]] targets
    tags TEXT,                            -- JSON array of tags
    confidence REAL DEFAULT 0.8,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_company_vault_dimension ON company_vault_notes(dimension);

-- Team vault notes (scoped to team)
CREATE TABLE IF NOT EXISTS team_vault_notes (
    id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    note_path TEXT NOT NULL,
    dimension TEXT NOT NULL,             -- 'team-decision' | 'cross-role-pattern' | 'team-experience'
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    backlinks TEXT,
    tags TEXT,
    confidence REAL DEFAULT 0.8,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(team_id, note_path)
);

CREATE INDEX IF NOT EXISTS idx_team_vault_team ON team_vault_notes(team_id);
CREATE INDEX IF NOT EXISTS idx_team_vault_dimension ON team_vault_notes(dimension);

-- Role vault notes (per role, shared across teams)
-- Uses filesystem (existing Second Brain architecture) with frontmatter metadata
-- No separate SQL table — role vault is file-based for Obsidian compatibility
```

### 3.3 LTM Dimensions: Who Owns What?

| Dimension | Owner | Scope | Example |
|---|---|---|---|
| D1: User Personality | **Company** | Global — every role in every team | "Prefers terse responses, Thai/English, plan-then-execute" |
| D2: Hard-Skill Knowledge | **Role** | Per role, shared across teams | "FastAPI async patterns" → fullstack-dev role vault |
| D3: Soft-Skill Knowledge | **Role** | Per role, shared across teams | "Error communication style" → devops role vault |
| D4: Project Experience | **Split** | Company for strategic, Team for tactical | "Why we chose FastAPI" → company; "Sprint 3 API contract" → team |
| D5: Company Policies | **Company** | Global — enforced guardrails | "All APIs must have /health endpoint" |
| D6: Shared Patterns | **Company** | Global — cross-team reusable | "_isolate_hermes_home fixture pattern" |
| D7: Team Decisions | **Team** | Scoped to team | "Frontend uses React, Backend uses FastAPI" |
| D8: Cross-Role Patterns | **Team** | Scoped to team | "How frontend and backend agree on API contracts" |
| D9: Team Experience | **Team** | Scoped to team | "V1 launch post-mortem — what went wrong" |

**เปลี่ยนจาก v1**: D1 (personality) และ D5/D6 (policies, shared patterns) ย้ายขึ้น company เพราะเป็นความรู้ที่ทุกคนในทุกทีมต้องรู้ ไม่ใช่ของ role ใด role หนึ่ง

## 4. Company-Level Memory Loading (Boot Protocol)

### 4.1 Three-Tier Boot

เมื่อ session เริ่ม จะโหลด LTM ทั้ง 3 tier ตามลำดับความสำคัญ:

```python
def build_full_memory_context(
    role: str,
    session_id: str,
    skill_tree: SkillTree = None,
    team_id: str = None,
    member_depth: int = 99,
) -> str:
    """Load all LTM tiers at session start: Company → Team → Role (skill-gated)."""

    context = ""

    # TIER 1: Company LTM — ALWAYS loaded for every agent in the company
    context += _load_company_ltm()

    # TIER 2: Team LTM — loaded if agent is in a team
    if team_id:
        context += _load_team_ltm(team_id, member_depth)

    # TIER 3: Role LTM — loaded for the agent's role, skill-gated
    context += _load_role_ltm(role, skill_tree)

    return context
```

### 4.2 Company LTM Loading

```python
def _load_company_ltm() -> str:
    """Company vault is always loaded — single source of truth."""
    company_vault = Path(get_hermes_home()) / "vaults" / "company"
    index_path = company_vault / "INDEX.md"

    if not index_path.exists():
        return ""

    context = "[Company Knowledge Index]\n"
    context += index_path.read_text()

    # Always load personality in full (highest confidence, global scope)
    personality_path = company_vault / "personality" / "user-personality.md"
    if personality_path.exists():
        context += f"\n[User Personality (Company-wide)]\n{personality_path.read_text()[:2000]}\n"

    # Always load policies in full (company guardrails)
    policies_dir = company_vault / "policies"
    if policies_dir.exists():
        for policy in policies_dir.glob("*.md"):
            context += f"\n[Policy: {policy.stem}]\n{policy.read_text()[:500]}\n"

    return context
```

### 4.3 Company INDEX.md Format

```markdown
# Hermes Agent — Company Knowledge Index

## Personality (D1)
- [user-personality](personality/user-personality.md) — Prefers terse responses, Thai/English, plan-then-execute

## Policies (D5)
- [coding-standards](policies/coding-standards.md) — Python 3.11+, uv, conventional commits
- [security-policy](policies/security-policy.md) — OWASP top 10, shlex.quote, no secrets in logs
- [deployment-process](policies/deployment-process.md) — Test before push, _isolate_hermes_home

## Strategic Decisions (D4)
- [why-fastapi](decisions/why-fastapi.md) — Async REST, concurrent sessions, /api/roles p99 ~15ms
- [monorepo-choice](decisions/monorepo-choice.md) — Single repo, Python + TypeScript

## Shared Patterns (D6)
- [error-handling](shared-patterns/error-handling.md) — _cprint for ANSI, space-padding for spinners
- [testing-isolation](shared-patterns/testing-isolation.md) — _isolate_hermes_home autouse fixture

## Experience (D4)
- [hermes-v1-launch](experience/hermes-v1-launch.md) — 119 integration tests, all passing
- [security-remediation](experience/security-remediation.md) — P0-P2 fixes, guardrail engine, 322+ tests
```

## 5. Consolidation Pipeline: Three-Tier Routing

### 5.1 Routing Decision Tree

เมื่อ STM entry ถูก evaluate แล้ว ต้องตัดสินใจว่าจะ merge เข้า tier ไหน:

```
STM Entry
  │
  ├─ Is this about the user's personality/preferences?
  │   └─→ Company LTM → personality/
  │
  ├─ Is this a company-wide policy, standard, or guardrail?
  │   └─→ Company LTM → policies/
  │
  ├─ Is this a strategic decision that affects the whole project?
  │   └─→ Company LTM → decisions/
  │
  ├─ Is this a reusable pattern that applies across teams?
  │   └─→ Company LTM → shared-patterns/
  │
  ├─ Is this a team-specific decision or pattern?
  │   └─→ Team LTM → decisions/ or cross-role/
  │
  ├─ Is this a team-level experience or post-mortem?
  │   └─→ Team LTM → experience/
  │
  ├─ Is this a role-specific technical fact?
  │   └─→ Role LTM → hard-skill/
  │
  ├─ Is this about communication/collaboration style?
  │   └─→ Role LTM → soft-skill/
  │
  └─ Is this ambiguous?
      └─→ Default to Role LTM (lowest blast radius, can be promoted later)
```

### 5.2 Company-Level Consolidation Engine

```python
class ConsolidationEngine:
    """Three-tier consolidation: Company → Team → Role."""

    def consolidate_session(
        self,
        session_id: str,
        role: str,
        team_context: dict = None,
    ) -> ConsolidationResult:
        stm_entries = self._load_stm(session_id)
        company_index = self._load_ltm_index("company")
        role_index = self._load_ltm_index(role)
        team_index = self._load_team_ltm_index(team_context) if team_context else None

        promoted = []
        skipped = []

        for entry in stm_entries:
            # Stage 1: Evaluate
            score = self._evaluate(entry, company_index, role_index, team_index)
            if score < CONSOLIDATION_THRESHOLD:
                skipped.append(entry)
                continue

            # Stage 2: Classify + Route
            target_tier, dimension = self._classify_target(entry, team_context)

            # Stage 3: Summarize against existing LTM in the target tier
            if target_tier == "company":
                existing = self._gather_related_ltm(entry, company_index)
            elif target_tier == "team":
                existing = self._gather_related_ltm(entry, team_index)
            else:  # role
                existing = self._gather_related_ltm(entry, role_index)

            summary = self._summarize(entry, existing)

            # Stage 4: Merge into target tier
            if target_tier == "company":
                merge_result = self._merge_to_company_vault(summary, dimension)
            elif target_tier == "team":
                merge_result = self._merge_to_team_vault(summary, dimension, team_context)
            else:
                merge_result = self._merge_to_role_vault(summary, dimension, role)

            # Stage 5: Link (cross-tier wiki-links)
            links = self._discover_links(merge_result, company_index, role_index, team_index)
            self._apply_links(merge_result, links)

            # Stage 6: Tag
            role_tags = self._tag_roles(merge_result, dimension, team_context)
            self._apply_role_tags(merge_result, role_tags)

            promoted.append(merge_result)

        # Rebuild all affected indexes
        self._rebuild_index("company")
        self._rebuild_index(role)
        if team_context:
            self._rebuild_team_index(team_context["team_id"])

        return ConsolidationResult(promoted=promoted, skipped=skipped)

    def _classify_target(self, entry: dict, team_context: dict = None) -> tuple:
        """Classify STM entry → (target_tier, dimension).

        Returns one of:
          ("company", "personality")  — user preferences
          ("company", "policies")      — company standards
          ("company", "decisions")     — strategic decisions
          ("company", "shared-patterns") — cross-team patterns
          ("team", "team-decision")    — team-specific decisions
          ("team", "cross-role-pattern") — team cross-role patterns
          ("team", "team-experience")  — team experience
          ("role", "hard-skill")       — role technical knowledge
          ("role", "soft-skill")       — role soft skills
          ("role", "experience")       — role project experience
        """
        # Use LLM classification with tier-aware prompt
        prompt = f"""Classify this knowledge into the most appropriate tier and dimension.
        Consider: Does this apply to the entire company, a specific team, or a specific role?

        Company-level: user personality, company policies, strategic decisions, reusable patterns
        Team-level: team decisions, cross-role collaboration patterns, team experience
        Role-level: role-specific technical knowledge, soft skills, project experience

        Knowledge: {entry['data']}
        Current role: {entry.get('role', 'unknown')}
        Current team: {team_context.get('team_name', 'none') if team_context else 'none'}

        Respond with JSON: {{"tier": "company|team|role", "dimension": "...", "confidence": 0.0-1.0}}"""

        result = self.auxiliary.classify(prompt)
        return result["tier"], result["dimension"]
```

### 5.3 Promotion: Role → Team → Company

Knowledge สามารถ "เลื่อนขึ้น" ได้เมื่อมันกลายเป็นที่รู้จักมากขึ้น:

```
Role LTM entry: "FastAPI async pattern works well"
  └─→ Multiple roles reference it → promote to Team LTM (cross-role pattern)
        └─→ Multiple teams reference it → promote to Company LTM (shared pattern)

Company LTM entry: "User prefers terse responses"
  └─→ Cannot be demoted — it's global truth
```

Promotion triggers:
- **Role → Team**: 2+ roles in the same team link to the same role-vault note
- **Team → Company**: 2+ teams link to the same team-vault note
- **Manual**: `/memory promote <note>` command

```python
def check_promotion_candidates(self) -> list:
    """Find notes that should be promoted to a higher tier."""
    candidates = []

    # Role → Team: notes with backlinks from 2+ roles in the same team
    for role_vault in (Path(get_hermes_home()) / "vaults" / "roles").iterdir():
        if not role_vault.is_dir():
            continue
        for note_file in role_vault.rglob("*.md"):
            note = MemoryNote.from_file(note_file)
            if len(note.backlinks) >= 2:
                backlink_roles = {bl["role"] for bl in note.backlinks}
                if len(backlink_roles) >= 2:
                    candidates.append({
                        "note": note,
                        "from_tier": "role",
                        "to_tier": "team",
                        "reason": f"Referenced by {len(backlink_roles)} roles",
                    })

    # Team → Company: notes with backlinks from 2+ teams
    for team_vault in (Path(get_hermes_home()) / "vaults" / "teams").iterdir():
        if not team_vault.is_dir():
            continue
        for note_file in team_vault.rglob("*.md"):
            note = MemoryNote.from_file(note_file)
            if len(note.backlinks) >= 2:
                backlink_teams = {bl["team"] for bl in note.backlinks}
                if len(backlink_teams) >= 2:
                    candidates.append({
                        "note": note,
                        "from_tier": "team",
                        "to_tier": "company",
                        "reason": f"Referenced by {len(backlink_teams)} teams",
                    })

    return candidates
```

## 6. XP Award = Three-Tier Consolidation Trigger

### 6.1 Modified Data Flow

```
Task completes
  └─→ record_session_and_award_xp(role, metrics, team_id, ...)
  │     ├─→ add_xp(role, delta)              # Individual XP → Role Level
  │     ├─→ add_team_xp(team_id, delta)       # Team XP → Team Level
  │     ├─→ check_and_unlock(role, level)     # Skill Tree → may unlock vault sections
  │     └─→ unlock_team_achievements()         # Team Achievements
  │
  └─→ consolidation_engine.consolidate_session(session_id, role, team_context)
        ├─→ Evaluate: score STM entries (company/team/role aware)
        ├─→ Classify: determine target tier (company/team/role)
        ├─→ Summarize: dedup against target tier's existing LTM
        ├─→ Merge: route to correct tier
        ├─→ Link: discover cross-tier wiki-links
        └─→ Tag: annotate with scope + role access
```

### 6.2 Modified `record_session_and_award_xp()`

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
    consolidation_engine: 'ConsolidationEngine' = None,
) -> Dict:
    """Record KPI metrics, calculate XP, add to role + team, check skill unlocks,
    AND trigger three-tier memory consolidation.

    Returns: {
        "xp_result": {...},
        "team_xp_result": {...},
        "newly_unlocked_skills": [],
        "achievements_unlocked": [],
        "consolidation_result": {
            "promoted": [...],        # STM entries promoted to LTM
            "company_notes": [...],   # Notes written to company vault
            "team_notes": [...],      # Notes written to team vault
            "role_notes": [...],      # Notes written to role vault
            "skipped": [...],         # Low relevance entries
        },
    }
    """
```

## 7. Company-Level Achievements and Knowledge

### 7.1 Company Achievements (in addition to Team Achievements)

Company-level achievements represent milestones for the entire Hermes instance:

| Achievement ID | Name | Trigger | Description |
|---|---|---|---|
| `company-first-session` | First Contact | First session ever | The company's first interaction with Hermes |
| `company-all-roles` | Full Roster | All 6 default roles have been used | All default roles are active |
| `company-3-teams` | Organization | 3+ teams created | The company has grown to 3+ teams |
| `company-1000-xp` | Knowledge Hoarders | Total company XP exceeds 1000 | Company accumulated 1000+ XP across all teams |
| `company-vault-50` | Library | 50+ LTM notes across all tiers | The company knowledge base has 50+ notes |
| `company-promotion-5` | Knowledge Flows | 5+ notes promoted from role → team or team → company | Knowledge naturally flows upward |

### 7.2 Company Achievement → Company LTM Note

เช่นเดียวกับ team achievements ที่สร้าง team experience notes แต่ company achievements สร้าง **company-level LTM notes**:

```python
COMPANY_ACHIEVEMENT_LTM_MAP = {
    "company-first-session": {
        "dimension": "experience",
        "template": "First Contact — The company began its journey with Hermes. Initial setup and first task completed: {context}",
    },
    "company-all-roles": {
        "dimension": "shared-patterns",
        "template": "Full Roster — All 6 default roles are active. Cross-role collaboration patterns emerging: {context}",
    },
    "company-3-teams": {
        "dimension": "decisions",
        "template": "Organization — The company has formed 3+ teams. Organizational structure and team boundaries: {context}",
    },
}
```

### 7.3 Company XP (Aggregated)

Company XP คือผลรวมของ team XP ทั้งหมด + individual XP ทั้งหมด:

```python
def get_company_xp(self) -> Dict:
    """Aggregate all XP across all teams and roles."""
    teams = self.team_manager.list_teams()
    total_team_xp = sum(t["team_xp"] for t in teams)
    total_individual_xp = self.kpi_tracker.get_total_xp()
    return {
        "company_xp": total_team_xp + total_individual_xp,
        "company_level": floor((total_team_xp + total_individual_xp) / 100) + 1,
        "total_teams": len(teams),
        "total_members": sum(t.get("member_count", 0) for t in teams),
        "total_notes": self._count_all_vault_notes(),
    }
```

## 8. Wiki-Link System: Three-Tier Cross-References

### 8.1 Link Syntax Extension

| Syntax | Meaning | Target Tier |
|---|---|---|
| `[[note-name]]` | Same-tier, same-scope link | Current vault |
| `[[role:note-name]]` | Cross-role link (role vault) | Role tier |
| `[[team:note-name]]` | Team vault link | Team tier |
| `[[company:note-name]]` | **NEW**: Company vault link | Company tier |
| `[[team:note-name#section]]` | Team vault section link | Team tier |
| `[[company:note-name\|display]]` | **NEW**: Company link with alias | Company tier |

### 8.2 Cross-Tier Link Resolution

```python
def resolve_wiki_link(link: str, current_scope: dict) -> Optional[MemoryNote]:
    """Resolve a [[wiki-link]] across three tiers."""
    current_tier = current_scope["tier"]      # "company" | "team" | "role"
    current_role = current_scope["role"]
    current_team = current_scope.get("team_id")

    if ":" in link:
        prefix, note_name = link.split(":", 1)

        if prefix == "company":
            target_vault = Path(get_hermes_home()) / "vaults" / "company"
        elif prefix == "team":
            if not current_team:
                return None  # Not in a team context
            target_vault = Path(get_hermes_home()) / "vaults" / "teams" / current_team
        elif prefix in DEFAULT_ROLES:
            target_vault = Path(get_hermes_home()) / "vaults" / "roles" / prefix
        else:
            return None  # Unknown prefix
    else:
        # Same-tier, same-scope
        if current_tier == "company":
            target_vault = Path(get_hermes_home()) / "vaults" / "company"
        elif current_tier == "team":
            target_vault = Path(get_hermes_home()) / "vaults" / "teams" / current_team
        else:
            target_vault = Path(get_hermes_home()) / "vaults" / "roles" / current_role

    candidates = list(target_vault.rglob(f"{note_name}.md"))
    if not candidates:
        return None
    return MemoryNote.from_file(candidates[0])
```

### 8.3 Access Rules for Cross-Tier Links

| From ↓ To → | Company LTM | Team LTM | Role LTM |
|---|---|---|---|
| **Company LTM** | Full access | Full access | `roles_aware` summaries only |
| **Team LTM** | Full access | Same team full, other teams `roles_aware` | Own role full, other roles `roles_aware` |
| **Role LTM** | Full access | Own team full, other teams none | Own role full, other roles `roles_aware` |

**หลักการ**: Company LTM เป็น single source of truth ที่ทุกคนอ่านได้ Team LTM อ่านได้เฉพาะในทีมตัวเอง Role LTM อ่านได้เฉพาะ role ตัวเอง + `roles_aware` summaries

## 9. Skill Tree Unlock = Three-Tier Vault Access

### 9.1 Skill-Gated Access Across Tiers

เดิม v1 มี skill-gated access เฉพาะ role vault ตอนนี้ขยายเป็นทุก tier:

| Skill Type | Company Vault Effect | Team Vault Effect | Role Vault Effect |
|---|---|---|---|
| `passive` | Unlock company policy notes | Unlock team decision notes | Unlock role hard-skill notes |
| `active` | Unlock company shared-patterns | Unlock team cross-role patterns | Unlock role toolset + hard-skill notes |
| `prompt` | Inject company policy prompt | Inject team context prompt | Inject role knowledge prompt |
| `vault` | Unlock specific company vault note | Unlock specific team vault note | Unlock specific role vault note |

### 9.2 Skill Tree Schema (Extended)

```sql
CREATE TABLE IF NOT EXISTS skill_trees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    skill_name TEXT NOT NULL,
    skill_type TEXT NOT NULL DEFAULT 'passive',
    level_threshold INTEGER NOT NULL,
    description TEXT,
    unlocked_content TEXT,              -- JSON: bonus/toolset/prompt
    vault_unlocks TEXT,                 -- JSON: {"role": [...], "team": [...], "company": [...]}
    unlocked INTEGER DEFAULT 0,
    unlocked_at REAL,
    UNIQUE(role, skill_name)
);
```

ตัวอย่าง:

```python
"fullstack-dev": [
    {
        "skill_name": "CSS Master",
        "level_threshold": 2,
        "skill_type": "passive",
        "description": "Frontend styling expertise",
        "unlocked_content": '{"bonus": {"kpi_weights": {"tool_diversity_score": 0.2}}}',
        "vault_unlocks": '{"role": ["hard-skill/css-advanced.md"], "team": [], "company": []}',
    },
    {
        "skill_name": "API Architect",
        "level_threshold": 5,
        "skill_type": "prompt",
        "description": "Backend API design mastery",
        "unlocked_content": '{"prompt": "You excel at designing robust, scalable APIs."}',
        "vault_unlocks": '{"role": ["hard-skill/api-design-patterns.md"], "team": ["decisions/api-standards.md"], "company": ["shared-patterns/api-contract-patterns.md"]}',
    },
    {
        "skill_name": "System Designer",
        "level_threshold": 8,
        "skill_type": "active",
        "description": "Full system design capabilities",
        "unlocked_content": '{"toolset": "system-engineer"}',
        "vault_unlocks": '{"role": ["experience/system-architecture.md"], "team": ["cross-role/architecture-review-patterns.md"], "company": ["decisions/system-architecture-adr.md"]}',
    },
],
```

### 9.3 Three-Tier Boot with Skill Gates

```python
def build_full_memory_context(
    role: str,
    session_id: str,
    skill_tree: SkillTree = None,
    team_id: str = None,
    member_depth: int = 99,
) -> str:
    """Load all LTM tiers at session start: Company → Team → Role (skill-gated)."""
    context = ""

    # === TIER 1: Company LTM — ALWAYS fully accessible (no skill gates) ===
    # Company policies, user personality, and shared patterns are global knowledge
    context += _load_company_ltm()

    # === TIER 2: Team LTM — skill-gated within team ===
    if team_id:
        context += _load_team_ltm(team_id, member_depth, skill_tree)

    # === TIER 3: Role LTM — skill-gated ===
    context += _load_role_ltm(role, skill_tree)

    # === Vault access manifest ===
    accessible = _list_accessible_vault_paths(role, skill_tree, team_id, member_depth)
    context += f"\n# Vault Access\n{accessible}\n"

    return context
```

**หมายเหตุ**: Company LTM ไม่มี skill gate — เป็น common knowledge ที่ทุก role ในทุก team เข้าถึงได้เต็มที่ เพราะมันคือ "วิธีทำงานของบริษัท" ที่ทุกคนต้องรู้

## 10. Team Hierarchy = Memory Access Hierarchy (v2)

### 10.1 Access Control Matrix

| Position | Company LTM | Team LTM | Role LTM | Can Write Company | Can Write Team |
|---|---|---|---|---|---|
| Team Lead (depth=0) | Full read | Full read/write | Full read own + `roles_aware` others | Yes (policies, decisions) | Yes |
| Sub-Lead (depth=1) | Full read | Read all + write subtree | Full read own + `roles_aware` subtree | No (suggest only) | Yes (subtree scope) |
| Worker (depth=2+) | Full read | Read only | Full read own only | No | No |

### 10.2 Write Suggestion System (for non-Leads)

Workers และ Sub-Leads ไม่สามารถเขียน Company LTM ได้โดยตรง แต่สามารถ **suggest** ได้:

```python
def suggest_company_note(self, member: dict, note_data: dict) -> str:
    """Non-leads can suggest notes for Company LTM. Leads review and approve."""
    suggestion_id = str(uuid.uuid4())
    suggestion = {
        "id": suggestion_id,
        "suggested_by": member["agent_id"],
        "suggested_by_role": member["role"],
        "suggested_by_depth": member["depth"],
        "note_data": note_data,
        "status": "pending",  # pending | approved | rejected
        "created_at": time.time(),
    }
    self._save_suggestion(suggestion)
    return suggestion_id

def review_company_suggestion(self, lead_member: dict, suggestion_id: str, action: str) -> bool:
    """Lead reviews and approves/rejects a Company LTM suggestion."""
    if not self.can_write_company_vault(lead_member):
        return False
    suggestion = self._load_suggestion(suggestion_id)
    if action == "approve":
        self._merge_to_company_vault(suggestion["note_data"], suggestion["note_data"]["dimension"])
        suggestion["status"] = "approved"
    else:
        suggestion["status"] = "rejected"
    self._save_suggestion(suggestion)
    return True
```

## 11. Unified Prompt Assembly for `delegate_task` (v2)

### 11.1 Integrated Prompt (Three-Tier)

```
{existing_system_prompt}

# ═══ Company Knowledge (Global) ═══
{build_company_ltm_context()}

# ═══ Team Context ═══
You are **{agent_id}** on team **{team_name}**.
Your position: {position} (depth={depth})
Team Level: {team_level} | Team XP: {team_xp}/{xp_to_next}

{build_team_ltm_context(team_id, member_depth)}

# ═══ Role Knowledge (Skill-Gated) ═══
{build_role_ltm_context(role, skill_tree)}

{role_manager.build_role_system_prompt(role)}
{skill_tree.apply_unlocked_prompts(role)}

# ═══ Skill Tree Status ═══
{rendered_skill_tree_for_role}

# ═══ Vault Access ═══
Company: full read access
Team: {"read" if depth >= 0 else "none"} | Write: {"yes" if depth <= 1 else "no"}
Role: full read (skill-gated sections unlocked as shown above)
```

## 12. Company-Level Commands

### 12.1 New CLI Commands

| Command | Description | Example |
|---|---|---|
| `/company info` | Show company stats, XP, teams, vault size | `/company info` |
| `/company vault` | Show company vault INDEX | `/company vault` |
| `/company promote <note>` | Suggest promoting a role/team note to company level | `/company promote fastapi-patterns` |
| `/company achievements` | Show company-level achievements | `/company achievements` |
| `/team show <name>` | Show team hierarchy + team vault status | `/team show dev-squad` |
| `/team vault <name>` | Show team vault INDEX | `/team vault dev-squad` |
| `/memory consolidate` | Run consolidation pipeline (all tiers) | `/memory consolidate` |
| `/memory lint` | Run lint pass on all vault tiers | `/memory lint` |
| `/memory promote` | List promotion candidates (role→team, team→company) | `/memory promote` |

### 12.2 `/company info` Output Example

```
╭─ Hermes Company ─────────────────────────────────╮
│ Company Level: 12 | Total XP: 1,150              │
│ Teams: 3 | Members: 9 | Vault Notes: 67           │
│ Achievements: 4 (First Contact, Full Roster,     │
│   Organization, Knowledge Flows)                  │
╰───────────────────────────────────────────────────╯

  📊 Teams:
  ├── Dev Squad (Lv.5, 4 members, 23 notes)
  ├── Quant Desk (Lv.3, 3 members, 12 notes)
  └── Content Studio (Lv.2, 2 members, 8 notes)

  📚 Company Vault: 24 notes
  ├── personality: 1 note (user-personality)
  ├── policies: 3 notes
  ├── decisions: 8 notes
  ├── shared-patterns: 6 notes
  └── experience: 6 notes
```

## 13. Implementation Phases (v2 — Company-Centric)

### Phase 1: Foundation (shared with Team Hierarchy doc)
- Schema v12: teams, team_members, skill_trees (with vault_unlocks), team_achievements, team_vault_notes, company_vault_notes, company_achievements
- TeamManager class
- SkillTree class (with vault_unlocks across 3 tiers)
- ConsolidationEngine class (base, with three-tier routing)

### Phase 2: Company Vault Layer
- CompanyVaultManager class (CRUD, write approval, suggestion system)
- Company vault directory seeding on first Hermes run
- Company INDEX.md auto-generation
- Company-level achievements
- `/company` CLI commands

### Phase 3: Three-Tier Integration Wiring
- `record_session_and_award_xp()` triggers three-tier consolidation
- ConsolidationEngine three-tier routing (company/team/role)
- Skill tree `resolve_unlocked_vault_paths()` across 3 tiers
- Achievement → LTM note creation at both team and company levels
- Promotion system (role → team → company)

### Phase 4: Prompt Assembly Integration
- `delegate_tool.py` injects three-tier context (company + team + role)
- `build_company_ltm_context()` — always loaded
- `build_team_ltm_context()` — team context with skill-gated access
- `build_role_ltm_context()` — role context with skill-gated access
- Cross-tier wiki-link resolution (`[[company:...]]`, `[[team:...]]`, `[[role:...]]`)

### Phase 5: CLI + UX
- `/company info`, `/company vault`, `/company promote`, `/company achievements`
- `/team show` renders hierarchy + vault status for each tier
- `/memory consolidate` supports three-tier routing
- `/memory lint` covers all three tiers
- `/memory promote` lists promotion candidates

### Phase 6: Tests
- Three-tier vault CRUD + access control
- Skill-gated vault access across 3 tiers
- Consolidation routing (company/team/role)
- Company achievement → company LTM note creation
- Cross-tier wiki-link resolution
- Promotion system (role → team → company)
- End-to-end: task → XP → three-tier consolidation → next boot context

## 14. Key Design Decisions (v2 Changes)

### 14.1 Why Company as top tier (not Team)?

**v1 มี Team เป็น top tier → v2 เปลี่ยนเป็น Company เป็น top tier เพราะ:**

- **Single source of truth**: User personality, company policies, and strategic decisions ไม่ใช่ของทีมใดทีมหนึ่ง — มันเป็นของทั้ง company
- **Cross-team knowledge**: Shared patterns (testing, error handling, deployment) ควรอยู่ที่ระดับ company ไม่ใช่ duplicate ในแต่ละ team
- **Organizational hierarchy**: ในโลกจริง company > team > individual ถูกต้องกว่า team > individual
- **Team lifecycle**: ทีมอาจถูก disbanded แต่ company knowledge ยังอยู่ — ถ้า team vault เป็น top tier เมื่อทีมถูกยุบ knowledge ก็จะสูญหาย

### 14.2 Why Company LTM has no skill gates?

- Company LTM คือ "วิธีทำงานของบริษัท" — ทุกคนต้องรู้ไม่ว่าจะระดับไหน
- Policies, coding standards, user preferences เป็น global constraints ที่ทุก agent ต้องปฏิบัติตาม
- การซ่อน company policies จาก low-level agent เป็นอันตรายต่อความปลอดภัยและคุณภาพ

### 14.3 Why promotion system (role → team → company)?

- Knowledge ที่เริ่มจาก role-specific อาจกลายเป็น shared pattern เมื่อใช้ข้ามทีม
- Manual promotion เท่านั้นไม่พอ — ต้องมี automatic detection เมื่อ note ถูก reference ข้าม boundary
- Promotion เป็น one-way (ขึ้นไปบน) เพราะ:
  - Company knowledge เป็น global truth
  - Team knowledge เป็น team truth
  - Role knowledge เป็น specialist truth
  - ลดขึ้นไม่ได้เพราะจะทำให้ higher-tier notes มี broken links

### 14.4 Why non-leads can suggest but not write to Company LTM?

- ป้องกัน noise — ทุกคนมีไอเดีย แต่ไม่ใช่ทุกไอเดียเป็น company-level knowledge
- Lead review ช่วยกรองและ validate ก่อนเขียน
- คล้ายกับ real-world process: พนักงานเสนอ policy, manager อนุมัติ

## 15. Migration from v1

### 15.1 What Changes from v1

| v1 Concept | v2 Equivalent | Migration |
|---|---|---|
| `vaults/<role>/` (all in one) | `vaults/roles/<role>/` | Move to subdirectory |
| `vaults/<role>/personality/` | `vaults/company/personality/` | Move personality up to company |
| `vaults/teams/<team_id>/` | `vaults/teams/<team_id>/` | Unchanged |
| `~/.hermes/vaults/` flat structure | `~/.hermes/vaults/{company,teams,roles}/` | Reorganize |
| `[[role:note]]` cross-role link | Same syntax, same behavior | No change |
| `[[team:note]]` team link | Same syntax, same behavior | No change |
| — | `[[company:note]]` company link | **NEW** |
| Skill `vault_unlocks` role-only | Skill `vault_unlocks` across 3 tiers | Schema change |
| Team vault = top tier | Company vault = top tier | Promotion system |

### 15.2 Migration Script

```python
def migrate_v1_to_v2(hermes_home: Path) -> None:
    """Migrate v1 vault structure to v2 three-tier structure."""
    old_vaults = hermes_home / "vaults"

    # 1. Create new directory structure
    (old_vaults / "company").mkdir(exist_ok=True)
    (old_vaults / "teams").mkdir(exist_ok=True)
    (old_vaults / "roles").mkdir(exist_ok=True)

    # 2. Move personality directories from each role to company
    for role_dir in old_vaults.iterdir():
        if role_dir.is_dir() and role_dir.name not in ("company", "teams", "roles"):
            personality_dir = role_dir / "personality"
            if personality_dir.exists():
                # Merge all role personalities into company personality
                for note in personality_dir.glob("*.md"):
                    dest = old_vaults / "company" / "personality" / note.name
                    if not dest.exists():
                        shutil.copy2(note, dest)

    # 3. Move role directories under roles/
    for role_dir in old_vaults.iterdir():
        if role_dir.is_dir() and role_dir.name not in ("company", "teams", "roles"):
            # Remove personality (already moved to company)
            if (role_dir / "personality").exists():
                shutil.rmtree(role_dir / "personality")
            # Move role dir to roles/
            role_dir.rename(old_vaults / "roles" / role_dir.name)

    # 4. Rebuild all INDEX.md files
    rebuild_all_indexes(hermes_home)
```

## 16. Risk & Mitigation (v2 Additions)

| Risk | Mitigation |
|---|---|
| Context window overflow with 3 vault tiers | Company LTM loaded as compact INDEX.md only (personality + policies are exceptions — always full). Team and Role INDEX.md only, with full notes loaded on-demand |
| Company vault becoming stale (no one updates it) | Company achievements auto-create notes; promotion system surfaces role/team knowledge that should be company-level |
| Write suggestion system too bureaucratic | Auto-promotion for notes with 2+ cross-boundary references; suggestions only needed for manual pushes |
| Three-tier routing adds consolidation complexity | LLM classification prompt is tier-aware; fallback to role-level if classification confidence < 0.7 |
| Personality fragmentation (duplicated across roles in v1) | v2 centralizes personality in company vault; role vaults reference `[[company:user-personality]]` |
| Broken links during promotion (note moves tier) | Promotion preserves original note with a redirect stub; backlinks are updated across all tiers |