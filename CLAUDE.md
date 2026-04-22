# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hermes Agent is a self-improving AI agent built by Nous Research. It is a multi-interface Python application with TypeScript UIs. The core is an OpenAI-compatible tool-calling agent loop with a self-registering tool system, SQLite session persistence, and a messaging gateway supporting Telegram, Discord, Slack, WhatsApp, Signal, and more.

- **Python**: 3.11+ (managed by `uv`)
- **Node**: 18+ (20+ for some packages)
- **Primary package manager**: `uv` for Python, `npm` for JS

## Development Environment Setup

```bash
# One-shot setup (creates venv, installs deps, symlinks CLI)
./setup-hermes.sh

# Manual setup (equivalent)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"

# Optional: browser tools
npm install

# Optional: RL training submodule
git submodule update --init tinker-atropos
uv pip install -e "./tinker-atropos"
```

**Always activate the venv before running Python code**:
```bash
source venv/bin/activate
```

**Hermes config lives at** `~/.hermes/config.yaml` (settings) and `~/.hermes/.env` (API keys).

## Build, Test, and Lint Commands

### Python

```bash
# Run tests — ALWAYS use the canonical wrapper (matches CI behavior)
scripts/run_tests.sh                              # full suite, CI-parity
scripts/run_tests.sh tests/agent/                 # one directory
scripts/run_tests.sh tests/agent/test_foo.py::test_x  # single test
scripts/run_tests.sh -v --tb=long                 # pass-through pytest flags

# Only if you cannot use the wrapper (e.g. IDE integration)
source venv/bin/activate
python -m pytest tests/ -q -n 4 --ignore=tests/integration --ignore=tests/e2e -m "not integration"

# Integration tests (require API keys)
python -m pytest tests/integration/ -m integration -v

# E2E tests
python -m pytest tests/e2e/ -v

# Install in editable mode with all extras
uv pip install -e ".[all,dev]"
```

### TUI (Ink/React terminal UI — `ui-tui/`)

```bash
cd ui-tui
npm install          # first time
npm run dev          # watch mode
npm start            # production
npm run build        # full build
npm run type-check   # tsc --noEmit
npm run lint         # eslint
npm run fmt          # prettier
npm test             # vitest
```

### Web UI (`web/`)

```bash
cd web
npm install
npm run dev          # vite dev server
npm run build        # production build
npm run lint         # eslint
npm run preview      # preview production build
```

## High-Level Architecture

### Core Files

| File | Role |
|------|------|
| `run_agent.py` | `AIAgent` class — core conversation loop, tool dispatch, session persistence |
| `cli.py` | `HermesCLI` class — interactive TUI (prompt_toolkit), banner, skin engine, slash commands |
| `model_tools.py` | Tool orchestration: discovers tools, handles `function_call` dispatch, builds tool definitions for LLM API |
| `toolsets.py` | Tool groupings and presets (`hermes-cli`, `hermes-telegram`, etc.) |
| `hermes_state.py` | `SessionDB` — SQLite with FTS5 full-text search, session titles, conversation storage. Schema v7 includes gamification tables (`agent_kpi`, `agent_skills_xp`, `agent_achievements`) |
| `batch_runner.py` | Parallel batch trajectory generation |
| `trajectory_compressor.py` | Compress conversation trajectories for training data |
| `agent/role_manager.py` | `RoleManager` — role-based agent personas with toolset presets, KPI weights, and system prompt injection |
| `agent/gamification.py` | `KPITracker` — KPI tracking, XP/Level system, and achievements per role |
| `agent/planning_engine.py` | `PlanningEngine` — multi-agent task planning with `should_plan()` heuristic and `plan_and_execute()` delegation + verification |
| `agent/memory_extractor.py` | `MemoryExtractor` — post-turn automatic fact extraction via auxiliary_client, deduplicated by SHA-256, stored as JSONL in `~/.hermes/memories/` |
| `hermes_cli/web_server.py` | FastAPI web server for the Web UI dashboard. REST endpoints: `/api/roles`, `/api/kpi`, `/api/xp`, `/api/achievements`, `/api/leaderboard` |
| `web/src/pages/RolesPage.tsx` | Web UI dashboard for role management, KPI metrics, XP/level, achievements, leaderboard |
| `ui-tui/src/app/slash/commands/roles.ts` | TUI `/roles` slash command — fetches role/KPI/XP/achievement/leaderboard data via RPC and renders a `Panel` dashboard |

### Key Directories

| Directory | Contents |
|-----------|----------|
| `agent/` | Agent internals: `prompt_builder.py` (system prompt assembly), `context_compressor.py` (auto-summarization near token limits), `auxiliary_client.py` (vision/summarization LLM clients), `display.py` (KawaiiSpinner, tool progress), `model_metadata.py` (context lengths, token estimation), `trajectory.py` |
| `hermes_cli/` | CLI command implementations: `main.py` (entry point), `config.py` (settings, migration), `commands.py` (central slash command registry `CommandDef`), `callbacks.py` (clarify, sudo, approval), `skin_engine.py` (data-driven theming), `skills_hub.py` (Skills Hub CLI) |
| `tools/` | Tool implementations (self-registering via `registry.py`). Each file co-locates schema, handler, and `registry.register()` call. Key tools: `terminal_tool.py`, `file_operations.py`/`file_tools.py`, `web_tools.py`, `browser_tool.py`, `delegate_tool.py`, `code_execution_tool.py`, `cronjob_tools.py`, `mcp_tool.py`, `ide_bridge_tool.py` (VS Code/JetBrains JSON-RPC), `voice_input_tool.py` (Whisper STT) |
| `tools/environments/` | Terminal execution backends: `local.py`, `docker.py`, `ssh.py`, `modal.py`, `daytona.py`, `singularity.py` |
| `gateway/` | Messaging gateway: `run.py` (GatewayRunner, message routing, cron), `session.py` (SessionStore), `platforms/` (Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant, QQ, DingTalk, Feishu adapters) |
| `ui-tui/` | Ink (React) terminal UI — `hermes --tui`. TypeScript frontend that communicates with Python via JSON-RPC over stdio |
| `tui_gateway/` | Python JSON-RPC backend for the TUI: `server.py` (RPC handlers), `entry.py` (stdio entrypoint), `slash_worker.py` (persistent subprocess for slash commands) |
| `web/` | Vite + React web UI (`hermes web`) |
| `cron/` | Built-in cron scheduler |
| `acp_adapter/` | ACP server for VS Code / Zed / JetBrains integration |
| `skills/` | Bundled skills (shipped with every install) |
| `optional-skills/` | Official optional skills (discoverable via hub, not activated by default) |
| `tests/` | Pytest suite (~3000 tests). `conftest.py` contains `_isolate_hermes_home` autouse fixture that redirects `HERMES_HOME` to a temp dir |

### Critical Design Patterns

**Self-registering tools**: Every tool file calls `registry.register()` at import time. `model_tools.py` triggers discovery by importing all tool modules. Tools return JSON strings.

**Toolsets**: Tools are grouped into named toolsets (`web`, `terminal`, `file`, `browser`, etc.) that can be enabled/disabled per platform. A tool belongs to exactly one toolset.

**Session persistence**: All conversations stored in SQLite (`hermes_state.py`) with FTS5 search and unique session titles. JSON logs go to `~/.hermes/sessions/`.

**Ephemeral injection**: System prompts and prefill messages are injected at API call time, never persisted to the database or logs.

**Prompt caching integrity**: Do NOT implement changes that alter past context mid-conversation, change toolsets mid-conversation, or reload memories/rebuild system prompts mid-conversation. Cache-breaking forces dramatically higher costs. The only time context is altered is during context compression.

**Role-based management**: Roles define toolset presets, default models, skins, and KPI weightings. User-defined roles in `~/.hermes/roles/*.yaml` override built-in defaults. Role switches MUST only happen at session boundaries (new session), never mid-conversation.

**Slash command registry**: All slash commands are defined centrally in `hermes_cli/commands.py` as `CommandDef` objects. This single list automatically drives CLI dispatch, gateway dispatch, Telegram BotCommand menus, Slack subcommand routing, autocomplete, and help text.

**Profiles (multi-instance)**: Hermes supports multiple fully isolated instances via `HERMES_HOME`. `_apply_profile_override()` in `hermes_cli/main.py` sets `HERMES_HOME` before any module imports.

## Important Constraints and Pitfalls

### Paths: Never hardcode `~/.hermes`
- Use `get_hermes_home()` from `hermes_constants` for all code paths.
- Use `display_hermes_home()` from `hermes_constants` for user-facing messages.
- Hardcoding `~/.hermes` or `Path.home() / ".hermes"` breaks profiles.

### Terminal UI constraints
- Do NOT use `simple_term_menu` for interactive menus — rendering bugs in tmux/iTerm2 cause ghosting on scroll. Use `curses` (stdlib) instead. See `hermes_cli/tools_config.py` for the pattern.
- Do NOT use `\033[K` (ANSI erase-to-EOL) in spinner/display code — leaks as literal `?[K` text under prompt_toolkit's `patch_stdout`. Use space-padding: `f"\r{line}{' ' * pad}"`.

### Cross-platform
- `termios` and `fcntl` are Unix-only. Always catch both `ImportError` and `NotImplementedError`.
- `os.setsid()`, `os.killpg()`, and signal handling differ on Windows. Use `platform.system() != "Windows"` checks.
- Use `pathlib.Path` instead of string path concatenation.
- If you change `scripts/install.sh`, check if the equivalent change is needed in `scripts/install.ps1`.

### Testing discipline
- Tests must not write to `~/.hermes/`. The `_isolate_hermes_home` autouse fixture in `tests/conftest.py` redirects `HERMES_HOME` to a temp dir.
- When testing profile features, also mock `Path.home()` so `_get_profiles_root()` resolves within the temp dir.
- Always run the full suite before pushing: `scripts/run_tests.sh`.

### Security-sensitive patterns
- Always use `shlex.quote()` when interpolating user input into shell commands.
- Resolve symlinks with `os.path.realpath()` before path-based access control checks.
- Don't log secrets (API keys, tokens, passwords).

### Adding a new tool
1. Create `tools/your_tool.py` with a `registry.register()` call.
2. Add the import to `model_tools.py` in the `_modules` list.
3. If it's a new toolset, add it to `toolsets.py` and relevant platform presets.
4. If the tool serves specific roles, add it to the role toolsets in `toolsets.py` (e.g. `quant-trader`, `fullstack-dev`).

### Adding a new role
1. Add the role definition to `agent/role_manager.py` `DEFAULT_ROLES` dict, or create a YAML file in `~/.hermes/roles/<name>.yaml`.
2. Add the role's toolset to `toolsets.py` (or reuse an existing toolset).
3. Run `/role list` to verify — no restart needed for user-defined roles.

### Adding a new slash command
1. Add a `CommandDef` to `COMMAND_REGISTRY` in `hermes_cli/commands.py`.
2. Add handler in `HermesCLI.process_command()` in `cli.py`.
3. If gateway-available, add handler in `gateway/run.py`.
4. For commands with dynamic completions (e.g. `/role switch <name>`), add a `_xxx_completions()` static method in `commands.py` and hook it into `SlashCommandCompleter.get_completions()`.

### Adding config options
1. Add to `DEFAULT_CONFIG` in `hermes_cli/config.py`.
2. Bump `_config_version` to trigger migration for existing users.

### Gamification / KPI tracking
- KPI metrics are tracked automatically per session when a role is active (`run_agent.py` `_kpi_counters`).
- The `KPITracker` in `agent/gamification.py` persists to `agent_kpi`, `agent_skills_xp`, and `agent_achievements` tables (schema v7).
- XP formula: `level = floor(xp / XP_PER_LEVEL) + 1` (default XP_PER_LEVEL = 100).
- When testing gamification features, ensure `_isolate_hermes_home` covers the new tables.

### Role / KPI Frontend Dashboards
- **CLI**: Add slash command handlers in `HermesCLI.process_command()` (`cli.py`). Use `_cprint()` for ANSI-safe output.
- **TUI Gateway**: Add `@method` decorated RPC handlers in `tui_gateway/server.py`. Return `_ok(rid, data)` / `_err(rid, code, msg)`. Import modules inside the handler function.
- **Web Server**: Add FastAPI endpoints in `hermes_cli/web_server.py`. Import `SessionDB` inside the handler. Wrap in `try / finally: db.close()`.
- **TUI Frontend**: Add slash commands in `ui-tui/src/app/slash/commands/`. Use `ctx.gateway.rpc<T>(method, params)` + `ctx.guarded<T>()` + `.catch(ctx.guardedErr)`. Render with `ctx.transcript.panel(title, sections)`.
- **Web Frontend**: Add pages in `web/src/pages/`. Use `AnalyticsPage.tsx` as the canonical dashboard pattern. Define API methods in `web/src/lib/api.ts`. Wire routes in `web/src/App.tsx`. Add translations in `web/src/i18n/en.ts` + `types.ts`.

## Entry Points and Running

```bash
# CLI (classic prompt_toolkit)
hermes

# CLI with a specific role
hermes --role quant-trader

# TUI (Ink/React)
hermes --tui

# Gateway (messaging platforms)
hermes gateway

# Web UI
hermes web

# Setup wizard
hermes setup

# Direct Python (when developing in repo)
python run_agent.py
python cli.py
python -m hermes_cli.main

# Role management slash commands (inside a session)
/role list
/role switch fullstack-dev
/kpi
/leaderboard

# TUI dashboard (Ink/React terminal)
/roles

# Web UI dashboard
hermes web  # then click "Roles" in the sidebar
```

## Commit Style

Conventional Commits: `type(scope): description`

Types: `fix`, `feat`, `docs`, `test`, `refactor`, `chore`
Scopes: `cli`, `gateway`, `tools`, `skills`, `agent`, `install`, `whatsapp`, `security`, `tui`, `web`

Example: `fix(cli): prevent crash in save_config_value when model is a string`
