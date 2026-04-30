---
scope: Role-Scoped Second Brain Memory Architecture for Hermes Agent
audience: Implementation agent (next session)
verified_via: code-review-graph (37,337 nodes, develop@60345c9a) + web research (Karpathy LLM Wiki, AgeMem, MAGMA, GAM, FluxMem)
generated: 2026-04-29
status: research-complete
---

# Second Brain Memory Architecture for Role-Scoped Agents

## 1. Executive Summary

Design an Obsidian-wiki-style "Second Brain" memory system where each Hermes role (devops, quant-trader, fullstack-dev, etc.) maintains its own scoped knowledge vault with wiki-link cross-references. The system implements a **dual-tier memory model** (Long-Term + Short-Term) with an automated **Evaluate → Summarize → Merge → Link → Tag** consolidation pipeline that promotes ephemeral project context into durable role knowledge.

Key insight from research: **compilation is lossy and opinionated** — the LLM makes editorial decisions at consolidation time, not retrieval time. This is the fundamental difference from flat RAG. Knowledge compounds across sessions rather than being re-derived each query.

## 2. Current State (Hermes Codebase)

### 2.1 Existing Memory Infrastructure

| Component | File | Capability | Gap |
|-----------|------|------------|-----|
| `MemoryManager` | `agent/memory_manager.py:192` | Orchestrates providers (builtin + 1 external), prefetch/recall, tool routing | No role scoping — all roles share same memory pool |
| `MemoryProvider` | `agent/memory_provider.py:43` | Abstract base: `initialize()`, `prefetch()`, `sync_turn()`, tool schemas | No concept of memory tiers (LTM/STM) or consolidation |
| `MemoryExtractor` | `agent/memory_extractor.py:32` | Post-turn fact extraction via auxiliary_client, SHA-256 dedup, JSONL storage | Flat fact list — no structure, no linking, no role tagging |
| `ClaudeMemoryManager` | `agent/claude_memory_manager.py:129` | Read/write/delete MEMORY.md index + frontmatter files, bidirectional Hermes sync | Tied to Claude Code's flat `user/feedback/project/reference` types — no wiki-links, no consolidation |
| `RoleManager` | `agent/role_manager.py:242` | 6 default roles, YAML overrides, toolset presets | No per-role memory scope — roles define tools, not knowledge |
| `SessionDB` | `hermes_state.py` | SQLite FTS5, session titles, KPI tables | Session data only — no persistent knowledge graph |

### 2.2 What Needs to Change

The current system treats memory as a **flat append-only log of facts**. There is:
- No distinction between durable knowledge (LTM) and ephemeral context (STM)
- No consolidation pipeline (facts accumulate but never graduate)
- No inter-note linking (no `[[wiki-links]]`)
- No role-scoped access (all roles see all memories)
- No structured retrieval beyond keyword/FTS match

## 3. Target Architecture

### 3.1 Dual-Tier Memory Model

```
┌─────────────────────────────────────────────────────────┐
│                   Long-Term Memory (LTM)                 │
│                  ~/.hermes/vaults/<role>/                 │
│                                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐ │
│  │ User         │ │ Hard-Skill   │ │ Soft-Skill      │ │
│  │ Personality  │ │ Knowledge    │ │ Knowledge       │ │
│  │              │ │              │ │                 │ │
│  │ preferences  │ │ lang/framework│ │ communication   │ │
│  │ habits       │ │ patterns     │ │ problem-solving │ │
│  │ work-style   │ │ best-practices│ │ collaboration   │ │
│  └──────┬───────┘ └──────┬───────┘ └────────┬────────┘ │
│         │                │                   │          │
│  ┌──────┴────────────────┴───────────────────┘        │
│  │  Experience from Project Development                │
│  │  (consolidated from STM)                             │
│  │  decisions + rationale + outcomes                    │
│  └─────────────────────────────────────────────────────┘
│                                                          │
│  Structure: Obsidian-wiki markdown with [[wiki-links]]   │
│  Index: vaults/<role>/INDEX.md (auto-maintained)        │
│  Graph: backlinks, tags, role-access annotations        │
└─────────────────────────────────────────────────────────┘
                         ▲
                         │ Evaluate → Summarize → Merge
                         │ (consolidation pipeline)
                         │
┌─────────────────────────────────────────────────────────┐
│                Short-Term Memory (STM)                    │
│            ~/.hermes/sessions/<session_id>/              │
│                                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐ │
│  │ Project      │ │ Project      │ │ Project Plan    │ │
│  │ Progress     │ │ Description  │ │                 │ │
│  │              │ │              │ │ milestones      │ │
│  │ tasks-done   │ │ goals       │ │ deadlines       │ │
│  │ blockers     │ │ constraints │ │ next-steps      │ │
│  │ next-actions │ │ stakeholders │ │ dependencies    │ │
│  └──────────────┘ └──────────────┘ └─────────────────┘ │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐│
│  │ Intraday Action Log                                  ││
│  │ timestamped entries: what was done, what failed,    ││
│  │ what was decided, what needs follow-up               ││
│  └─────────────────────────────────────────────────────┘│
│                                                          │
│  Structure: JSONL + markdown scratchpad                  │
│  Lifecycle: session-scoped, TTL-based expiry             │
└─────────────────────────────────────────────────────────┘
```

### 3.2 LTM Dimensions (Detail)

#### D1: User Personality

Persistent model of the human operator — migratable across agent platforms.

```yaml
# vaults/fullstack-dev/personality/user-personality.md
---
type: ltm
dimension: user-personality
roles: [fullstack-dev, devops, quant-trader]  # shared across these roles
confidence: 0.85
last_updated: 2026-04-29
sources: [claude-code-migration, codex-migration, observation]
---

## Communication Style
- Prefers terse responses with code examples over prose explanations
- Thai primary language, English for code/technical
- Approves plans before execution; dislikes unsolicited refactors

## Technical Preferences
- Python (uv) + TypeScript (npm); avoids Ruby
- Values prompt caching integrity and backward compatibility
- Modular additions over monolithic rewrites

## Working Patterns
- Deep focus sessions (2-4 hours); batch reviews after
- Parallel delegation via sub-agents when tasks are independent
- Checks graph before grep (non-negotiable)
```

**Migration sources** (priority order):
1. **Claude Code** → Parse `~/.claude/memory/` files, extract frontmatter + body
2. **Codex** → Parse Codex session artifacts / project context files
3. **Gemini** → Parse Gemini Code Assist project context
4. **Observation** → Inferred from interaction patterns (lowest confidence, tagged `source: observation`)

Migration format:
```python
def migrate_claude_memory(hermes_home: str, role: str) -> list[MemoryNote]:
    """Read Claude Code memory dir, classify into LTM dimensions, write to role vault."""
    claude_dir = Path.home() / ".claude" / "memory"
    # Each .md file → parse frontmatter (name, description, type)
    # type=user → user-personality dimension
    # type=feedback → soft-skill dimension
    # type=project → project-experience dimension
    # type=reference → hard-skill dimension
    # Write to vaults/<role>/ with [[wiki-links]] injected
```

#### D2: Hard-Skill Knowledge

Role-specific technical knowledge — frameworks, patterns, tooling, best practices.

```yaml
# vaults/quant-trader/hard-skill/black-scholes-patterns.md
---
type: ltm
dimension: hard-skill
roles_direct: [quant-trader, propfirm-trader]
roles_aware: [fullstack-dev]  # fullstack may build quant UI
confidence: 0.9
last_updated: 2026-04-29
---

## Black-Scholes Implementation Pattern
Standard BS pricing with Greeks calculation...

### Related
- [[option-greeks-quick-ref]]  # cross-link to another hard-skill note
- [[quant-error-recovery]]      # soft-skill: how to handle pricing failures
- [[rust-execution-sandbox]]    # hard-skill: execution environment

### Tags
#quant #options #pricing #black-scholes
```

#### D3: Soft-Skill Knowledge

Transferable skills — communication, problem-solving, collaboration patterns.

```yaml
# vaults/fullstack-dev/soft-skill/error-communication.md
---
type: ltm
dimension: soft-skill
roles_direct: [fullstack-dev, devops, system-engineer]
roles_aware: [quant-trader, content-creator]  # all roles benefit
confidence: 0.75
last_updated: 2026-04-29
---

## Error Communication Pattern
When reporting failures to user...

### Related
- [[user-personality]]  # link to D1 — how THIS user prefers error reports
- [[incident-postmortem-template]]  # hard-skill link
```

#### D4: Experience from Project Development

Consolidated outcomes from completed projects — decisions, rationale, results.

```yaml
# vaults/fullstack-dev/experience/web-ui-roles-dashboard.md
---
type: ltm
dimension: project-experience
roles_direct: [fullstack-dev]
roles_aware: [content-creator]  # CC may document UI features
project: hermes-agent
confidence: 0.95
last_updated: 2026-04-29
consolidated_from: sessions/2026-04-22-a3f1
---

## Roles Dashboard — Key Decisions
1. Used FastAPI for REST (not Flask) — async support for concurrent sessions
2. KPI counter in run_agent.py (not per-request calc) — performance at scale
3. Auto-refresh 30s (not WebSocket) — simpler, adequate for dashboard scope

### Outcome
- 119 integration tests, all passing
- /api/roles endpoint: ~15ms p99

### Related
- [[rest-api-pattern]]            # hard-skill: FastAPI patterns
- [[test-isolation-pattern]]      # hard-skill: _isolate_hermes_home
- [[user-personality]]           # D1: user approved single-bundled PR
```

### 3.3 STM Dimensions (Detail)

```jsonl
// sessions/2026-04-29-b7c2/stm.jsonl
{"ts": "2026-04-29T10:30:00Z", "type": "project_progress", "data": {"task": "resolve merge conflicts", "status": "in_progress", "blockers": []}}
{"ts": "2026-04-29T10:35:00Z", "type": "project_description", "data": {"goal": "merge main into develop", "constraints": ["preserve RolesPage", "adopt main layout refactor"]}}
{"ts": "2026-04-29T10:40:00Z", "type": "project_plan", "data": {"milestone": "merge complete", "next": "run test suite", "deadline": null}}
{"ts": "2026-04-29T10:45:00Z", "type": "intraday_action", "data": {"action": "resolved App.tsx import conflict", "result": "merged both imports", "followup": "check TypeScript compilation"}}
```

### 3.4 Directory Structure

```
~/.hermes/
├── vaults/                          # LTM — Obsidian-compatible wiki vaults
│   ├── fullstack-dev/
│   │   ├── INDEX.md                  # auto-maintained content catalog
│   │   ├── personality/
│   │   │   └── user-personality.md
│   │   ├── hard-skill/
│   │   │   ├── fastapi-patterns.md
│   │   │   ├── react-dashboard.md
│   │   │   └── test-isolation.md
│   │   ├── soft-skill/
│   │   │   ├── error-communication.md
│   │   │   └── plan-then-execute.md
│   │   └── experience/
│   │       ├── web-ui-roles-dashboard.md
│   │       └── security-remediation.md
│   ├── quant-trader/
│   │   ├── INDEX.md
│   │   ├── personality/
│   │   │   └── user-personality.md → ../shared/user-personality.md
│   │   ├── hard-skill/
│   │   │   ├── black-scholes-patterns.md
│   │   │   └── rust-execution-sandbox.md
│   │   └── ...
│   ├── devops/
│   ├── propfirm-trader/
│   ├── content-creator/
│   └── system-engineer/
├── sessions/                         # STM — per-session ephemeral context
│   └── <session_id>/
│       ├── stm.jsonl                  # structured short-term entries
│       ├── scratchpad.md              # working notes, unstructured
│       └── consolidation-log.jsonl    # audit trail of what was promoted
└── memories/                          # legacy (MemoryExtractor) — migration source
```

## 4. Consolidation Pipeline

The core innovation: **STM → Evaluate → Summarize → Merge → Link → Tag → LTM**

### 4.1 Pipeline Stages

```
Session End (or TTL trigger)
        │
        ▼
┌─────────────────┐
│ 1. EVALUATE     │  Is this STM entry worth keeping?
│                 │  - Relevance score (0-1) per role
│                 │  - Novelty check vs existing LTM
│                 │  - Actionability: can a future session use this?
└────────┬────────┘
         │ if score > threshold
         ▼
┌─────────────────┐
│ 2. SUMMARIZE    │  Extract essence, deduplicate with LTM
│                 │  - LLM call: "Given existing LTM [[X]], [[Y]],
│                 │    summarize this session fact removing redundancy"
│                 │  - Result: concise, non-redundant knowledge unit
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. MERGE        │  Write to appropriate LTM dimension
│                 │  - Classify: personality | hard-skill | soft-skill | experience
│                 │  - If note exists: UPDATE (not append) — maintain single source
│                 │  - If new: CREATE with frontmatter + wiki-links
│                 │  - Update INDEX.md
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. LINK         │  Create bidirectional [[wiki-links]]
│                 │  - Semantic: "this concept relates to that concept"
│                 │  - Causal: "this decision led to that outcome"
│                 │  - Temporal: "this experience followed that project"
│                 │  - Cross-dimension: D2 ↔ D3 ↔ D4 ↔ D1
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. TAG          │  Annotate role access
│                 │  - roles_direct: [list] — primary consumers
│                 │  - roles_aware: [list] — need to know for communication
│                 │  - Write access tags into frontmatter
└─────────────────┘
```

### 4.2 Consolidation Triggers

| Trigger | When | Scope |
|--------|------|-------|
| **Session end** | User closes session or idle timeout | All STM entries from that session |
| **TTL expiry** | STM entry older than N hours (default: 24h) | Expired entries batch-processed |
| **Semantic shift** | Detected topic change mid-session | Partial consolidation of completed topic |
| **Manual** | User runs `/memory consolidate` | Explicit control |

### 4.3 Consolidation Algorithm (Pseudocode)

```python
class ConsolidationEngine:
    """Promotes STM entries to LTM with deduplication and linking."""

    def __init__(self, role: str, vault_dir: Path, auxiliary_client):
        self.role = role
        self.vault_dir = vault_dir
        self.auxiliary = auxiliary_client  # for LLM-based summarization

    def consolidate_session(self, session_id: str) -> ConsolidationResult:
        stm_entries = self._load_stm(session_id)
        ltm_index = self._load_ltm_index()

        promoted = []
        skipped = []

        for entry in stm_entries:
            # Stage 1: Evaluate
            score = self._evaluate(entry, ltm_index)
            if score < CONSOLIDATION_THRESHOLD:
                skipped.append(entry)
                continue

            # Stage 2: Summarize (deduplicate against existing LTM)
            summary = self._summarize(entry, ltm_index)

            # Stage 3: Merge (classify + write)
            dimension = self._classify_dimension(summary)
            merge_result = self._merge_to_ltm(summary, dimension, ltm_index)

            # Stage 4: Link (wiki-links)
            links = self._discover_links(merge_result, ltm_index)
            self._apply_links(merge_result, links)

            # Stage 5: Tag (role access)
            role_tags = self._tag_roles(merge_result, dimension)
            self._apply_role_tags(merge_result, role_tags)

            promoted.append(merge_result)

        # Update INDEX.md
        self._rebuild_index()

        return ConsolidationResult(promoted=promoted, skipped=skipped)

    def _evaluate(self, entry: dict, ltm_index: dict) -> float:
        """Score relevance (0-1) for consolidation."""
        # Novelty: cosine similarity with existing LTM notes
        # Actionability: does this affect future decisions?
        # Role relevance: does this role use this type of knowledge?
        ...

    def _summarize(self, entry: dict, ltm_index: dict) -> str:
        """LLM-powered summarization with deduplication."""
        existing_context = self._gather_related_ltm(entry, ltm_index)
        prompt = (
            f"Given existing knowledge:\n{existing_context}\n\n"
            f"Summarize this new session fact, removing anything already covered "
            f"above. Extract only the novel essence:\n{entry['data']}"
        )
        return self.auxiliary.summarize(prompt)

    def _discover_links(self, note: MemoryNote, index: dict) -> list[str]:
        """Find wiki-link targets via semantic + keyword matching."""
        # 1. Keyword: extract nouns/technologies from note → match titles
        # 2. Semantic: embed note → find nearest LTM notes (cosine > 0.7)
        # 3. Causal: look for "because", "led to", "resulted in" → link cause→effect
        ...
```

### 4.4 Dual-Stream Consolidation (from MAGMA/GAM research)

Inspired by MAGMA's Fast Path / Slow Path:

| Path | Trigger | Latency | Action |
|------|---------|---------|--------|
| **Fast Path** | Mid-session, semantic shift detected | <2s | Write raw STM entry immediately |
| **Slow Path** | Session end or TTL expiry | Background, async | Full consolidation pipeline (evaluate → summarize → merge → link → tag) |

This ensures no knowledge is lost mid-session (Fast Path captures everything) while expensive LLM-based consolidation runs asynchronously (Slow Path produces quality LTM).

## 5. Role-Scoped Memory Loading

### 5.1 Boot Protocol

Every session startup loads the role's memory index **before** any user interaction:

```python
def build_role_memory_context(role: str, session_id: str) -> str:
    """Load role's LTM index at session start — the 'second brain boot'."""

    vault = Path(get_hermes_home()) / "vaults" / role
    index_path = vault / "INDEX.md"

    if not index_path.exists():
        return ""

    # Read INDEX.md — compact catalog of all LTM notes
    index_content = index_path.read_text()

    # Build context block for system prompt injection
    context = f"[{role} Second Brain Index]\n{index_content}\n"

    # Optionally pre-load high-confidence personality note
    personality_path = vault / "personality" / "user-personality.md"
    if personality_path.exists():
        personality = personality_path.read_text()
        # Strip body beyond threshold to manage context budget
        context += f"\n[User Personality (high confidence)]\n{personility[:2000]}\n"

    return context
```

### 5.2 INDEX.md Format

```markdown
# fullstack-dev — Second Brain Index

## Personality (D1)
- [user-personality](personality/user-personality.md) — Prefers terse responses, Thai/English, plan-then-execute

## Hard-Skill Knowledge (D2)
- [fastapi-patterns](hard-skill/fastapi-patterns.md) — Async REST patterns, dependency injection
- [react-dashboard](hard-skill/react-dashboard.md) — RolesPage dashboard pattern, auto-refresh
- [test-isolation](hard-skill/test-isolation.md) — _isolate_hermes_home fixture pattern
- [[quant-trader:black-scholes-patterns]] — Cross-role link (aware, not direct)

## Soft-Skill Knowledge (D3)
- [error-communication](soft-skill/error-communication.md) — Error reporting style preferences
- [plan-then-execute](soft-skill/plan-then-execute.md) — Approval-before-action workflow

## Project Experience (D4)
- [web-ui-roles-dashboard](experience/web-ui-roles-dashboard.md) — FastAPI choice, KPI counter, 119 tests
- [security-remediation](experience/security-remediation.md) — P0-P2 fixes, guardrail engine, 322+ tests
```

### 5.3 Cross-Role Access Rules

| Tag | Meaning | Load Behavior |
|-----|---------|---------------|
| `roles_direct: [A, B]` | A and B own this knowledge | Full load into context |
| `roles_aware: [C]` | C needs to know for communication | Summary only (1-line from INDEX.md) |
| No tag | Global/shared (e.g., user personality) | All roles load |

Cross-role links use `[[other-role:note-name]]` syntax. At boot time, `roles_aware` notes are loaded as one-line summaries; `roles_direct` notes are loaded in full.

## 6. Wiki-Link System

### 6.1 Link Types

| Syntax | Type | Example |
|--------|------|---------|
| `[[note-name]]` | Same-role link | `[[fastapi-patterns]]` |
| `[[role:note-name]]` | Cross-role link | `[[quant-trader:black-scholes-patterns]]` |
| `[[note-name#section]]` | Section link | `[[fastapi-patterns#dependency-injection]]` |
| `[[note-name\|display text]]` | Aliased link | `[[rust-execution-sandbox\|Rust Sandbox]]` |

### 6.2 Link Resolution at Retrieval Time

```python
def resolve_wiki_link(link: str, current_role: str, vault_dir: Path) -> Optional[MemoryNote]:
    """Resolve a [[wiki-link]] to its MemoryNote content."""
    if ":" in link:
        # Cross-role link: [[quant-trader:black-scholes-patterns]]
        role, note_name = link.split(":", 1)
        target_vault = vault_dir.parent / role
    else:
        role = current_role
        note_name = link
        target_vault = vault_dir

    # Search by filename (without .md extension)
    candidates = list(target_vault.rglob(f"{note_name}.md"))
    if not candidates:
        return None
    return MemoryNote.from_file(candidates[0])
```

### 6.3 Backlink Maintenance

When note A links to note B, B's frontmatter gains a `backlinks` entry:

```yaml
---
backlinks:
  - source: fastapi-patterns
    role: fullstack-dev
    context: "Uses async pattern from..."
---
```

Backlinks are updated during the Slow Path consolidation pass.

## 7. Integration with Existing Hermes Infrastructure

### 7.1 New Provider: `VaultMemoryProvider`

Extends `MemoryProvider` to implement the Second Brain:

```python
class VaultMemoryProvider(MemoryProvider):
    """Obsidian-wiki Second Brain memory provider with role scoping."""

    @property
    def name(self) -> str:
        return "vault"

    def is_available(self) -> bool:
        return True  # Always available — uses local filesystem

    def initialize(self, session_id: str, **kwargs) -> None:
        role = kwargs.get("agent_identity", "fullstack-dev")
        self.role = role
        self.vault_dir = Path(kwargs["hermes_home"]) / "vaults" / role
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.consolidation_engine = ConsolidationEngine(role, self.vault_dir, ...)

    def system_prompt_block(self) -> str:
        """Inject role's INDEX.md into system prompt at boot."""
        return build_role_memory_context(self.role, self.session_id)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Retrieve relevant LTM notes for current query."""
        return self._search_vault(query)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Write STM entry + trigger Fast Path for semantic shifts."""
        self._write_stm_entry(user_content, assistant_content)
        # Fast Path: if semantic shift detected, trigger partial consolidation
        if self._detect_semantic_shift(user_content, assistant_content):
            self.consolidation_engine.consolidate_partial(session_id)

    def on_session_end(self, session_id: str) -> None:
        """Trigger full Slow Path consolidation at session end."""
        self.consolidation_engine.consolidate_session(session_id)
```

### 7.2 Migration from Existing MemoryExtractor

```python
def migrate_legacy_memories(hermes_home: Path, role: str) -> None:
    """One-time migration: MemoryExtractor JSONL → role vault LTM notes."""
    legacy_dir = hermes_home / "memories"
    if not legacy_dir.exists():
        return

    vault_dir = hermes_home / "vaults" / role
    vault_dir.mkdir(parents=True, exist_ok=True)

    for jsonl_file in legacy_dir.glob("*.jsonl"):
        facts = [json.loads(line) for line in jsonl_file.read_text().splitlines()]
        # Batch classify and consolidate into vault structure
        consolidated = batch_consolidate(facts, role)
        for note in consolidated:
            write_vault_note(vault_dir, note)
```

### 7.3 MemoryExtractor Enhancement

The existing `MemoryExtractor` is retained as the **STM writer** (Fast Path). Its output format is extended:

```python
# Current: flat JSONL with fact_hash
entry = {"session_id": ..., "timestamp": ..., "fact": ..., "fact_hash": ...}

# Enhanced: add role + dimension classification
entry = {
    "session_id": ...,
    "timestamp": ...,
    "fact": ...,
    "fact_hash": ...,
    "role": current_role,                    # NEW: which role generated this
    "stm_dimension": "intraday_action",       # NEW: progress|description|plan|action
    "consolidation_status": "pending",        # NEW: pending|promoted|skipped|expired
}
```

## 8. Consolidation Quality Gates

### 8.1 Lint (from Karpathy's LLM Wiki pattern)

Periodic health check of the vault:

| Check | Detects | Fix |
|-------|---------|-----|
| Orphan detection | Notes with no inbound/outbound links | Auto-link via semantic similarity |
| Stale detection | Notes with `last_updated` > 90 days | Flag for re-verification |
| Contradiction detection | Two notes with conflicting claims | Present both for user resolution |
| Low-confidence flag | `confidence < 0.5` | Schedule re-extraction from sources |
| Index drift | INDEX.md doesn't match actual files | Rebuild INDEX.md |
| Broken links | `[[wiki-link]]` targets missing file | Remove link or create stub |
| Duplicate detection | Near-identical content in two notes | Merge into single canonical note |

### 8.2 Verification Protocol

After consolidation, run:
1. `query_graph pattern=callers_of target=VaultMemoryProvider` — verify integration points
2. `semantic_search_nodes query=consolidation` — find related code
3. Lint pass on vault — ensure no orphans or broken links
4. Cross-role access test — load each role's INDEX.md, verify all `[[role:note]]` links resolve

## 9. Research Foundations

### 9.1 Key Papers

| Paper | Year | Key Contribution | Applied Here |
|-------|------|----------------|--------------|
| [Memory for Autonomous LLM Agents](https://arxiv.org/abs/2603.07670v1) | 2026 | Write-manage-read loop, consolidation is underserved | Consolidation pipeline design (Section 4) |
| [Agentic Memory (AgeMem)](https://arxiv.org/abs/2601.01885v1) | 2026 | RL-trained unified LTM+STM management, proactive summarization | Learned consolidation triggers (future) |
| [MAGMA](https://arxiv.org/abs/2601.03236) | 2026 | Multi-graph memory (semantic, temporal, causal, entity), dual-stream | Fast/Slow Path consolidation (Section 4.4) |
| [GAM](https://arxiv.org/abs/2604.12285) | 2026 | Event Progression Graph + Topic Associative Network, semantic shift triggers | Session-end consolidation trigger |
| [FluxMem](https://arxiv.org/abs/2602.14038v1) | 2026 | Context-adaptive memory structure selection, probabilistic gating | Role-scoped structure selection (future) |

### 9.2 Key Community Projects

| Project | Key Pattern | Applied Here |
|---------|-------------|--------------|
| [Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) | Raw → Wiki → Schema 3-layer, compilation over retrieval, lint | Vault structure (Section 3.4), Lint (Section 8.1) |
| [obsidian-memory-for-ai](https://github.com/Jrcruciani/obsidian-memory-for-ai) | Sources → Wiki → Schema, 4 operations | Migration pipeline (Section 3.2 D1) |
| [ClawVault](https://github.com/Versatly/clawvault) | 8 primitives, wake/checkpoint/sleep lifecycle | Session lifecycle (Section 4.2), consolidation at sleep |
| [claude-obsidian](https://claude-blog.md/blog/claude-obsidian-second-brain) | Hot cache (`wiki/hot.md`), compounding wiki | INDEX.md as boot context (Section 5.1) |
| [LLM Wiki Skill](https://alirezarezvani.github.io/claude-skills/skills/engineering/llm-wiki/) | `/wiki-init`, `/wiki-ingest`, `/wiki-lint`, sub-agents | Slash commands, ConsolidationEngine |

### 9.3 Cross-Cutting Research Insights

1. **Graph-structured memory outperforms flat vector stores** — MAGMA achieves 95%+ token reduction while improving accuracy. Our `[[wiki-links]]` create an explicit graph.
2. **Semantic similarity ≠ causal relevance** — for "why" queries, explicit causal links (`this decision → that outcome`) beat embedding proximity. Our `[[wiki-links]]` carry semantic intent.
3. **Consolidation is the critical unsolved problem** — survey identifies it as "particularly underserved." Our 5-stage pipeline (Evaluate → Summarize → Merge → Link → Tag) addresses this.
4. **Learned policies discover non-obvious tactics** — AgeMem's RL agent learned "proactive summarization before context fills." Future: train consolidation triggers with RL from user feedback.
5. **At personal scale (~50-500 notes), grep + good organization beats RAG pipelines** — explicit links beat similarity scores. Our vault stays in this range per role.

## 10. Implementation Phases

### Phase 1: Vault Structure + STM Writer
- Create `VaultMemoryProvider` class extending `MemoryProvider`
- Implement vault directory structure with INDEX.md auto-generation
- Enhance `MemoryExtractor` to add `role`, `stm_dimension`, `consolidation_status` fields
- Register vault provider in `MemoryManager`
- Files: `agent/vault_memory_provider.py` (new), `agent/memory_extractor.py` (modify), `agent/memory_manager.py` (register)

### Phase 2: Consolidation Engine (Core)
- Implement `ConsolidationEngine` with 5-stage pipeline
- LLM-based summarization via `auxiliary_client`
- Dimension classifier (personality / hard-skill / soft-skill / experience)
- Wiki-link discovery (keyword + semantic)
- Role tag assignment (`roles_direct` / `roles_aware`)
- Files: `agent/consolidation_engine.py` (new)

### Phase 3: Memory Migration
- Claude Code memory → vault migration parser
- Codex artifact → vault migration parser
- Gemini context → vault migration parser
- Legacy `memories/` JSONL → vault batch consolidation
- Files: `agent/memory_migrator.py` (new)

### Phase 4: Boot Protocol + Role Integration
- `build_role_memory_context()` in `prompt_builder.py`
- Load INDEX.md + high-confidence personality at session start
- Cross-role link resolution
- `/memory consolidate` slash command
- `/memory lint` slash command
- Files: `agent/prompt_builder.py` (modify), `hermes_cli/commands.py` (add)

### Phase 5: Lint + Quality Gates
- Orphan detection, stale detection, contradiction detection
- Broken link repair
- INDEX.md rebuild
- Automated lint pass on session end (configurable)
- Files: `agent/vault_linter.py` (new)

### Phase 6: Testing + Validation
- Unit tests for ConsolidationEngine (5 stages)
- Integration tests for full pipeline (STM → LTM)
- Migration tests (Claude/Codex/Gemini → vault)
- Cross-role access tests
- Lint validation tests
- Files: `tests/agent/test_consolidation_engine.py`, `tests/agent/test_vault_memory_provider.py`, `tests/agent/test_memory_migrator.py`, `tests/agent/test_vault_linter.py`