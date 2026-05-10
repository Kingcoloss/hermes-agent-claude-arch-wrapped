---
scope: Hermes Company OS — Phased Implementation Plan (M1 + M1.5 first)
audience: Implementation agent (next session), CEO (user)
upstream:
  - domain-knowledge/real-world-company-architecture.md (vision, full 8-phase plan)
  - domain-knowledge/agent-team-hierarchy-gamification.md
  - domain-knowledge/second-brain-memory-architecture.md
  - domain-knowledge/team-memory-integration-synthesis-v2.md
  - domain-knowledge/team-memory-integration-synthesis-v2-reviewed.md
generated: 2026-04-30
status: plan-approved-pending-implementation
verification:
  - codebase analysis via code-review-graph (37,674 nodes, 317,110 edges)
  - schema confirmed at v11 in hermes_state.py:36
  - RoleManager (314 LOC), KPITracker (348 LOC), MemoryExtractor confirmed built
  - team/department/okr/project/release/incident/entity modules confirmed absent
guidelines: karpathy-guidelines (minimum surface, surface tradeoffs, verifiable goals)
---

# Hermes Company OS — Phased Implementation Plan

> Companion to `real-world-company-architecture.md`. The architecture doc is the **vision** (full enterprise OS). This doc is the **executable plan**: what to build first, in what order, with verifiable milestones. Re-bundles the original 8 phases into 5 milestones, each shipping CEO-visible value.

## 1. Why Re-bundle from 8 Phases to 5 Milestones

The original plan groups work by technical layer (schema, memory, delivery, entities, gamification, CLI, delegate, tests). Each phase ships internal plumbing but no user-visible workflow until much later. Karpathy guideline: define verifiable success criteria — every milestone here ships a workflow CEO can actually run.

Each milestone answers a single CEO question:

| Milestone | CEO question answered | Agent gains |
|---|---|---|
| **M1** Identity & Workflow | "ผมมี employees ที่ ship product ได้" | persistent ID, per-agent XP/KPI, personal journal |
| **M1.5** Peer Consultation | "พนักงานปรึกษากันได้ + Lead จัดการทีมแทนผมได้" | consult/panel/team broadcast + chain audit |
| **M2** Skills & Growth | "พนักงานเก่งขึ้นเอง + แชร์ความรู้กัน" | skill tree unlocks, team vault, multi-turn discussion |
| **M3** Goals & Onboarding | "ตั้ง OKR แล้วทุกคน align" | individual OKRs, new-hire ramp-up |
| **M4** Outside World | "ลูกค้า feedback กลับเข้าระบบ" | incidents, customer entities, post-mortem |
| **M5** Enterprise Scale | "บริษัทใหญ่พอแยก dept + governance" | departments, classification, identity transfer/alumni |

## 2. CEO Decisions Locked In

| # | Decision | Choice |
|---|---|---|
| 1 | Identity model | Persistent `agent_id`, multi-team membership (matrix org) |
| 2 | Release artifact | Git tag |
| 3 | Initial slice | Vertical: `create project → assign team → ship` |
| 4 | Git tag format | `<project>/v<version>` (e.g. `hermes-v2/v0.1.0`) |
| 5 | Git tag location | Per-project: `projects.repo_path` field |
| 6 | Agent seeding | Both — auto-seed 6 default agents on migration + `/agent create` for custom |

## 3. Milestone M1 — Identity & Workflow

### 3.1 Goal (verifiable)

CEO opens a fresh session, runs 11 commands end-to-end, git emits real tag `hermes-v2/v0.1.0`, integration test green:

```
/agent create alice --role fullstack-dev
/agent create bob   --role devops
/team create web-team
/team add web-team alice
/team add web-team bob
/project create hermes-v2 --repo .
/project assign hermes-v2 web-team
/release create hermes-v2 0.1.0
/release check <id>          # runs scripts/run_tests.sh
/release ship  <id>           # → git -C <repo_path> tag hermes-v2/v0.1.0
/project show  hermes-v2      # status=active, releases=[0.1.0 shipped]
```

After session close, `~/.hermes/vaults/agents/alice/journal.md` shows entry for the work.

### 3.2 Schema v11 → v12

7 new tables + 4 ALTER (additive, IF NOT EXISTS, no breaking changes to v11 tables):

```sql
CREATE TABLE agents (
  id TEXT PRIMARY KEY,                 -- 'alice', 'dev-1' (CEO-defined)
  role TEXT NOT NULL,
  status TEXT DEFAULT 'active',        -- active | inactive
  created_at REAL NOT NULL
);

CREATE TABLE teams (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  lead_agent_id TEXT REFERENCES agents(id),  -- added in v13 ALTER, listed here for reference
  created_at REAL NOT NULL
);

CREATE TABLE team_members (
  team_id TEXT NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
  agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  joined_at REAL NOT NULL,
  status TEXT DEFAULT 'active',
  PRIMARY KEY (team_id, agent_id)      -- agent อยู่หลายทีมได้ (matrix org)
);

CREATE TABLE projects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  status TEXT DEFAULT 'proposed',      -- proposed | active | completed | cancelled
  repo_path TEXT,                      -- working dir for git tag
  target_date REAL,
  created_at REAL NOT NULL,
  completed_at REAL
);

CREATE TABLE project_teams (
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  team_id TEXT NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
  PRIMARY KEY (project_id, team_id)
);

CREATE TABLE releases (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  version TEXT NOT NULL,                -- '0.1.0'
  status TEXT DEFAULT 'draft',          -- draft | testing | shipped | failed
  git_tag TEXT,                         -- 'hermes-v2/v0.1.0' set on ship
  created_at REAL NOT NULL,
  shipped_at REAL,
  UNIQUE(project_id, version)
);

CREATE TABLE release_checks (
  id TEXT PRIMARY KEY,
  release_id TEXT NOT NULL REFERENCES releases(id) ON DELETE CASCADE,
  check_type TEXT NOT NULL,             -- 'tests' (M1 has only this gate)
  status TEXT DEFAULT 'pending',
  result_text TEXT,
  checked_at REAL
);

-- Backward-compat ALTERs (NULL agent_id = legacy session/row, falls back to role)
ALTER TABLE sessions             ADD COLUMN agent_id TEXT REFERENCES agents(id);
ALTER TABLE agent_kpi            ADD COLUMN agent_id TEXT REFERENCES agents(id);
ALTER TABLE agent_skills_xp      ADD COLUMN agent_id TEXT REFERENCES agents(id);
ALTER TABLE agent_achievements   ADD COLUMN agent_id TEXT REFERENCES agents(id);
```

### 3.3 Auto-seed on migration

When `current_version < 12`, after CREATE TABLE blocks:

```python
# Seed 6 default agents (1 per built-in role) — CEO can rename/disable later
seed_agents = [
    ("dev-1",     "fullstack-dev"),
    ("devops-1",  "devops"),
    ("quant-1",   "quant-trader"),
    ("prop-1",    "propfirm-trader"),
    ("content-1", "content-creator"),
    ("eng-1",     "system-engineer"),
]
```

### 3.4 Per-agent journal (the "alive" feature)

Path: `~/.hermes/vaults/agents/<agent_id>/`

```
INDEX.md      # auto-rebuilt: list of journal entries
journal.md    # append-only, format:
              #
              # ## 2026-04-30 14:32 [hermes-v2] fullstack-dev
              # session: abc123 | KPI: tasks=3 ok=3 | XP +15 (level 2→2)
              # actions: created agents/teams, shipped 0.1.0
```

No LLM consolidation in M1. Journal is plain append; later phases add summarization.

### 3.5 New & modified files

| File | Δ LOC | Type |
|---|---|---|
| `hermes_state.py` | +150 | Modify (schema v12, migration block, seed) |
| `agent/agent_manager.py` | +180 | New — CRUD, multi-team enrollment, seed |
| `agent/team_manager.py` | +200 | New — CRUD, members, set_lead (stub for M1.5) |
| `agent/project_manager.py` | +180 | New — CRUD, assign team, lifecycle, repo_path |
| `agent/release_manager.py` | +250 | New — draft, check (subprocess `scripts/run_tests.sh`), ship (subprocess `git tag`) |
| `agent/agent_vault.py` | +120 | New — file vault + journal append + INDEX rebuild |
| `agent/gamification.py` | +50 | Modify — write per-agent rows when agent_id present |
| `run_agent.py` | +10 | Modify — session-end hook → `agent_vault.append_journal` |
| `hermes_cli/commands.py` | +40 | Modify — 13 CommandDefs |
| `cli.py` | +250 | Modify — handlers (pattern: same as `/role`) |
| `tests/integration/test_company_workflow.py` | +350 | New — full 11-command E2E + journal verification |
| **M1 total** | **~1780** | |

### 3.6 Slash commands added in M1

```
/agent create <id> [--role <role>]   /agent list                 /agent show <id>
/team   create <name>                /team   list                /team   show <name>
/team   add <team> <agent>           /project create <name> [--repo <path>]
/project list                        /project show <name>        /project assign <name> <team>
/release create <project> <ver>      /release check <id>         /release ship <id>
```

### 3.7 Constraints & invariants

1. **agent_id binding at session boundary only** — `sessions.agent_id` frozen for the session. No mid-session rebinding (matches CLAUDE.md "role switches MUST only happen at session boundaries").
2. **`/release ship` is destructive** — must go through Bash confirmation; creates irreversible git tag.
3. **Test runner = `scripts/run_tests.sh`** — never invoke `pytest` directly (CI parity).
4. **`_isolate_hermes_home` autouse fixture** — tests must not touch real `~/.hermes`.
5. **No system-prompt injection added in M1** — defer context fan-in to M3 to preserve prompt cache integrity.

### 3.8 Verification checklist

- [ ] `scripts/run_tests.sh` passes full suite after migration
- [ ] `scripts/run_tests.sh tests/integration/test_company_workflow.py` green
- [ ] Migration on a v11 DB succeeds without data loss
- [ ] 6 default agents seeded; CEO can `/agent create` custom
- [ ] alice in 2 teams works (multi-team membership)
- [ ] git tag `hermes-v2/v0.1.0` actually written in `repo_path`
- [ ] `~/.hermes/vaults/agents/alice/journal.md` populated after session close

## 4. Milestone M1.5 — Peer Consultation & Hierarchy

### 4.1 Goal

Workflows the CEO can run after M1.5:

```
/team set-lead web-team alice
/consult alice "ทีมประเมิน hermes-v2 เสร็จเมื่อไหร่"
   → alice (lead) calls consult_team("web-team")
   → bob, charlie answer in parallel
   → alice aggregates with own judgment
   → returns to CEO

/panel dev-1,quant-1,content-1 "should we add crypto trading?"
/team broadcast web-team "ใครว่างรับ feature X"
/consult-log alice --tree                  # full chain audit
```

### 4.2 Schema v12 → v13

```sql
CREATE TABLE consultations (
  id TEXT PRIMARY KEY,
  caller_session_id TEXT NOT NULL,
  caller_agent_id TEXT REFERENCES agents(id),       -- NULL = CEO
  target_agent_id TEXT NOT NULL REFERENCES agents(id),
  question TEXT NOT NULL,
  context_summary TEXT,                              -- LLM-summarized; never raw history
  response TEXT,
  status TEXT DEFAULT 'pending',                     -- pending | done | failed
  parent_consultation_id TEXT REFERENCES consultations(id),  -- chain root
  depth INTEGER DEFAULT 0,
  cost_tokens INTEGER,
  created_at REAL NOT NULL,
  completed_at REAL
);

ALTER TABLE teams ADD COLUMN lead_agent_id TEXT REFERENCES agents(id);
```

### 4.3 Semantic distinction from `delegate_task`

| | delegate_task (existing 2532 LOC) | consult_agent (new) |
|---|---|---|
| Semantic | "do this task" | "give me your perspective" |
| Output | work product | opinion + caveats |
| Caller's role | manager | peer or report |
| Persists | task result | conversation log w/ chain |
| Reuse code? | No — semantics differ + file already heavy | New file |

### 4.4 New tools

```python
consult_agent(target_agent_id: str, question: str, context_summary: str) -> str
consult_panel(target_agent_ids: list[str], question: str, context_summary: str) -> dict
consult_team (team_id: str, question: str,
              include_lead: bool = True, aggregate: bool = False) -> dict
```

`consult_team` re-uses `claude_subagent_batch` infra for parallel fan-out — no new fan-out code.

### 4.5 Hierarchy & recursion guards

```
CEO (depth 0)
  └─→ alice/lead (depth 1, parent=null)
        ├─→ bob     (depth 2, parent=alice's consult)   ✓ allowed
        └─→ charlie (depth 2)                           ✓ allowed
              └─→ would-be sub-consult                  ✗ BLOCKED (depth 3 > limit=2)
```

| Guard | Default | Configurable |
|---|---|---|
| Max depth | 2 | yes |
| Cycle detection | per-chain agent_id uniqueness | n/a |
| Per-session cost cap | 5 consults | yes |
| Token tracking | `consultations.cost_tokens` | display in `/consult-log` |

### 4.6 Cross-team consult

`alice` (lead web-team) → `bob` (lead mobile-team) is just a regular `consult_agent` — no special handling. `/consult-log` exposes the chain so CEO can audit.

### 4.7 Auto-journal both ends

After consultation completes, both caller and target get journal entries:

```
# alice's journal:
consulted bob (senior) about "code review of feature X" — outcome: refactor approach A

# bob's journal:
consulted by alice (junior) about "code review of feature X" — gave guidance on approach A
```

For `consult_team`, lead's journal records aggregation:

```
# alice's journal:
led team consult on "estimate hermes-v2", aggregated 2 responses, reported to CEO
```

### 4.8 New & modified files

| File | Δ LOC | Type |
|---|---|---|
| `hermes_state.py` | +35 | Modify (schema v13, 1 table + 1 ALTER) |
| `agent/consultation_manager.py` | +240 | New — CRUD, depth/cycle guard, cost track, chain tree query |
| `tools/consult_tool.py` | +320 | New — `consult_agent`, `consult_panel`, `consult_team` |
| `agent/team_manager.py` | +50 | Modify — `set_lead`, `get_members_excluding_lead` |
| `toolsets.py` | +10 | Modify — add `consult` toolset, attach to all roles |
| `hermes_cli/commands.py` | +25 | Modify — 6 new CommandDefs |
| `cli.py` | +180 | Modify — handlers |
| `tests/integration/test_agent_consultation.py` | +370 | New — E2E, hierarchy, recursion, cost cap, panel, broadcast |
| **M1.5 total** | **~1230** | |

### 4.9 Slash commands added in M1.5

```
/consult     <agent> <q>                 1-on-1 (any direction)
/panel       <a,b,c> <q>                 ad-hoc multi-agent (explicit list)
/team set-lead <team> <agent>            assign team lead
/team broadcast <team> <q>               fan-out to team (consult_team include_lead=true)
/team tree   <team>                      show lead + members
/consult-log [agent] [--tree]            audit chain, optionally as tree
```

### 4.10 M1.5 invariants

1. **One-shot only in M1.5** — no multi-turn discussion (defer to M2: thread state).
2. **`context_summary` mandatory** — caller passes LLM-summarized context, never raw history → keeps target's context window bounded.
3. **Open permissions in M1.5** — anyone can call `consult_team` for any team. Restricted access (e.g. lead-only broadcast) is a policy concern; ships with classification in M5.
4. **Cost cap per session covers chains** — `consult_team` with 5 members = 5 consults against quota.

### 4.11 Verification checklist

- [ ] CEO consults alice (lead) → alice consults team → CEO sees aggregated answer
- [ ] Depth-3 attempt blocked
- [ ] Cycle attempt (alice → bob → alice) blocked
- [ ] Session cost cap enforced (6th consult rejected when cap=5)
- [ ] Both caller and target get journal entries
- [ ] `/consult-log alice --tree` renders correct conversation tree
- [ ] Cross-team peer consult (alice → bob, both leads) works without going through CEO

## 5. Order of Work

```
1. M1 schema (v12) + migration  → run scripts/run_tests.sh, all v11 tests pass
2. M1 managers (agent/team/project/release/agent_vault) + journal hook in run_agent.py
3. M1 commands + handlers + integration test → green
4. **CEO trial run with M1** — actual session, ship Hermes-v2 dogfood release
5. (Only if M1 trial reveals no blockers) M1.5 schema (v13) + consultation_manager + tools
6. M1.5 commands + integration test → green
7. CEO trial run with M1.5
```

Step 4 is mandatory. M1.5 must not start before M1 has been used in a real session — guards against feature pile-up.

## 6. Future Milestones (Preview)

### M2 — Skills & Growth (~2000 LOC)

- `skill_trees`, `agent_skill_unlocks` tables (per-agent)
- XP threshold crossed → unlock skill node → inject text into system prompt at next session boot
- Team vault `~/.hermes/vaults/teams/<id>/`
- Simple promotion: agent journal entry tagged `#share` → copy to team vault
- Multi-turn discussion (extends `consult_agent` with thread state)

### M3 — Goals & Onboarding (~1500 LOC)

- OKR cascade: company → team → individual
- OKR weights modulate KPI prioritization (computed at session boundary only — preserves cache)
- Onboarding protocol: new agent's first session reads team vault INDEX
- Full system-prompt context fan-in (deferred from M1 to preserve prompt caching during simpler phases)

### M4 — Outside World (~1500 LOC)

- `incidents` + auto post-mortem on resolve
- `external_entities`, `entity_interactions` (customers, vendors, partners, competitors)
- Customer feedback flows to project vault

### M5 — Enterprise Scale (~1800 LOC)

- `departments` + team `department_id` membership
- 2-level classification (internal | restricted)
- Identity transfer & alumni
- Cross-team promotion (Team → Department → Company vault)
- ADR provenance, review_cadence linter

## 7. What This Plan Defers from `real-world-company-architecture.md`

| Original concept | Reason for defer | Lands in |
|---|---|---|
| 4-level classification (public/internal/confidential/restricted) | Premature for solo CEO; 2-level enough until external sharing | M5 (2-level) |
| 5-tier vault (Company/Dept/Team/Project/Role) | Dept tier needs depts to exist; project tier needs project notes | M2 (3-tier), M5 (5-tier) |
| Onboarding/transfer/alumni | Needs vault content to read first | M3 (onboarding), M5 (transfer/alumni) |
| Skill tree vault_unlocks | Needs vault first | M2 (skill tree without vault), later integration |
| OKR-driven dynamic KPI weights | Cache-breaking if mid-session; needs careful boundary handling | M3 (boundary-only) |
| Promotion triggers (5-tier routing) | Needs LLM consolidation engine | M3 (LLM consolidation), M5 (full routing) |
| ADR provenance + review_cadence linter | Governance for scale, not solo CEO | M5 |
| Incidents + post-mortem | Needs external users to have incidents about | M4 |
| External entities | Needs real users first | M4 |

## 8. Risk Register

| Risk | Mitigation |
|---|---|
| Schema migration v11→v12 breaks existing sessions | Additive only (CREATE IF NOT EXISTS, ALTER ADD with NULL default). Existing sessions read with `agent_id=NULL` and fall back to `sessions.role`. |
| Per-agent KPI rows duplicate per-role KPI rows | Both populated when `agent_id` present; queries that aggregate by role still work. Journal/leaderboard queries pick the dimension they need. |
| `consult_team` runaway cost | Per-session cap (default 5) + cost_tokens column for visibility |
| `consult_*` infinite recursion | Depth limit + cycle detection + chain audit |
| Prompt cache invalidation | M1 + M1.5 do not modify system prompt structure. Context fan-in deferred to M3 with explicit boundary handling. |
| Feature pile-up before validation | Mandatory CEO trial run between M1 and M1.5 |

## 9. Open Items (None Blocking M1 Start)

All blocking decisions resolved (§ 2). Items below surface during implementation:

- Auto-seed agent naming convention: `dev-1` vs `engineer-1` — minor, can rename
- Test gate beyond `scripts/run_tests.sh` (lint, security scan) — defer to M4 with incidents
- Aggregate response in `consult_team` (LLM-merged single answer) — defer to M2
- `/consult-log --tree` rendering format — choose during implementation
