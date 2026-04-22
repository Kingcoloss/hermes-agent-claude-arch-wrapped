<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
</p>

**The self-improving AI agent built by [Nous Research](https://nousresearch.com).** It's the only agent with a built-in learning loop — it creates skills from experience, improves them during use, nudges itself to persist knowledge, searches its own past conversations, and builds a deepening model of who you are across sessions. Run it on a $5 VPS, a GPU cluster, or serverless infrastructure that costs nearly nothing when idle. It's not tied to your laptop — talk to it from Telegram while it works on a cloud VM.

Use any model you want — [Nous Portal](https://portal.nousresearch.com), [OpenRouter](https://openrouter.ai) (200+ models), [NVIDIA NIM](https://build.nvidia.com) (Nemotron), [Xiaomi MiMo](https://platform.xiaomimimo.com), [z.ai/GLM](https://z.ai), [Kimi/Moonshot](https://platform.moonshot.ai), [MiniMax](https://www.minimax.io), [Hugging Face](https://huggingface.co), OpenAI, or your own endpoint. Switch with `hermes model` — no code changes, no lock-in.

<table>
<tr><td><b>A real terminal interface</b></td><td>Full TUI with multiline editing, slash-command autocomplete, conversation history, interrupt-and-redirect, and streaming tool output.</td></tr>
<tr><td><b>Lives where you do</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal, and CLI — all from a single gateway process. Voice memo transcription, cross-platform conversation continuity.</td></tr>
<tr><td><b>A closed learning loop</b></td><td>Agent-curated memory with periodic nudges. Autonomous skill creation after complex tasks. Skills self-improve during use. FTS5 session search with LLM summarization for cross-session recall. <a href="https://github.com/plastic-labs/honcho">Honcho</a> dialectic user modeling. Compatible with the <a href="https://agentskills.io">agentskills.io</a> open standard.</td></tr>
<tr><td><b>Scheduled automations</b></td><td>Built-in cron scheduler with delivery to any platform. Daily reports, nightly backups, weekly audits — all in natural language, running unattended.</td></tr>
<tr><td><b>Delegates and parallelizes</b></td><td>Spawn isolated subagents for parallel workstreams. Write Python scripts that call tools via RPC, collapsing multi-step pipelines into zero-context-cost turns.</td></tr>
<tr><td><b>Runs anywhere, not just your laptop</b></td><td>Six terminal backends — local, Docker, SSH, Daytona, Singularity, and Modal. Daytona and Modal offer serverless persistence — your agent's environment hibernates when idle and wakes on demand, costing nearly nothing between sessions. Run it on a $5 VPS or a GPU cluster.</td></tr>
<tr><td><b>Research-ready</b></td><td>Batch trajectory generation, Atropos RL environments, trajectory compression for training the next generation of tool-calling models.</td></tr>
</table>

---

## Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

Works on Linux, macOS, WSL2, and Android via Termux. The installer handles the platform-specific setup for you.

> **Android / Termux:** The tested manual path is documented in the [Termux guide](https://hermes-agent.nousresearch.com/docs/getting-started/termux). On Termux, Hermes installs a curated `.[termux]` extra because the full `.[all]` extra currently pulls Android-incompatible voice dependencies.
>
> **Windows:** Native Windows is not supported. Please install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and run the command above.

After installation:

```bash
source ~/.bashrc    # reload shell (or: source ~/.zshrc)
hermes              # start chatting!
```

---

## Getting Started

```bash
hermes              # Interactive CLI — start a conversation
hermes model        # Choose your LLM provider and model
hermes tools        # Configure which tools are enabled
hermes config set   # Set individual config values
hermes gateway      # Start the messaging gateway (Telegram, Discord, etc.)
hermes setup        # Run the full setup wizard (configures everything at once)
hermes claw migrate # Migrate from OpenClaw (if coming from OpenClaw)
hermes update       # Update to the latest version
hermes doctor       # Diagnose any issues
```

📖 **[Full documentation →](https://hermes-agent.nousresearch.com/docs/)**

## CLI vs Messaging Quick Reference

Hermes has two entry points: start the terminal UI with `hermes`, or run the gateway and talk to it from Telegram, Discord, Slack, WhatsApp, Signal, or Email. Once you're in a conversation, many slash commands are shared across both interfaces.

| Action | CLI | Messaging platforms |
|---------|-----|---------------------|
| Start chatting | `hermes` | Run `hermes gateway setup` + `hermes gateway start`, then send the bot a message |
| Start fresh conversation | `/new` or `/reset` | `/new` or `/reset` |
| Change model | `/model [provider:model]` | `/model [provider:model]` |
| Set a personality | `/personality [name]` | `/personality [name]` |
| Retry or undo the last turn | `/retry`, `/undo` | `/retry`, `/undo` |
| Compress context / check usage | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]` |
| Browse skills | `/skills` or `/<skill-name>` | `/skills` or `/<skill-name>` |
| Interrupt current work | `Ctrl+C` or send a new message | `/stop` or send a new message |
| Platform-specific status | `/platforms` | `/status`, `/sethome` |

For the full command lists, see the [CLI guide](https://hermes-agent.nousresearch.com/docs/user-guide/cli) and the [Messaging Gateway guide](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## Role-Based Agent Management

Hermes supports **role-based agent personas** that adapt the available tools, system prompt, and KPI tracking to specific professional roles. This is useful when you want the agent to behave as a specialized expert — whether you're doing DevOps automation, quantitative trading, content creation, or full-stack development.

### Built-in roles

| Role | Focus |
|------|-------|
| `devops` | Infrastructure automation, CI/CD, container orchestration, cloud operations |
| `quant-trader` | Statistical arbitrage, backtesting, options pricing, portfolio optimization |
| `propfirm-trader` | High-frequency execution, risk management, trade journaling |
| `content-creator` | Writing, media generation, social media, SEO research |
| `fullstack-dev` | Frontend, backend, database, API, testing, deployment |
| `system-engineer` | OS internals, networking, security, performance tuning |

### Quick usage

```bash
# List available roles
hermes --role list

# Start a session with a specific role
hermes --role quant-trader

# Switch role inside a running session
/role switch fullstack-dev
```

When you activate a role, Hermes automatically:
- **Filters tools** to only those relevant for the role
- **Injects role context** into the system prompt (e.g. "You are a quantitative trader. Prioritize precision and statistical rigor...")
- **Tracks KPIs** such as task success rate, tool diversity, and error recovery

### Custom roles

You can define your own roles by creating a YAML file in `~/.hermes/roles/`:

```yaml
# ~/.hermes/roles/my-custom-role.yaml
name: my-custom-role
description: A custom agent persona for my specific workflow
toolsets:
  - web
  - file
  - code_execution
default_model: null
skin: null
kpi_weights:
  task_success_rate: 1.2
  tool_diversity_score: 1.0
system_prompt_extra: |
  You are a specialist in X. Always do Y before Z.
```

After creating the file, run `/role list` to see it immediately — no restart needed.

### Gamification & KPIs

Each role tracks performance metrics stored in SQLite (`~/.hermes/state.db`):

| Metric | Description |
|--------|-------------|
| `task_success_rate` | Percentage of successful tool call outcomes |
| `avg_tokens_per_task` | Average token consumption per completed task |
| `tool_diversity_score` | How many different tools are used (higher = more versatile) |
| `error_recovery_rate` | Ability to recover from failed tool calls |
| `role_proficiency_score` | Composite score weighted per role |

```bash
# View KPI summary for the current role
/kpi

# View KPI for a specific role
/kpi quant-trader

# View leaderboard across roles
/leaderboard
```

XP and levels are tracked per role in `agent_skills_xp`:
- `XP_PER_LEVEL = 100` (configurable)
- Level formula: `floor(xp / 100) + 1`

**Important**: Role switches only happen at **session boundaries** to preserve prompt caching integrity. You cannot change roles mid-conversation — start a new session with `/new` or `/role switch <name>`.

### Role / KPI Dashboard

View role profiles, KPI metrics, XP/level progress, achievements, and leaderboards across all interfaces:

**CLI (classic prompt_toolkit)**
```bash
/role list              # List all roles with tool counts
/role switch devops     # Switch active role
/kpi                    # KPI summary for current role
/kpi quant-trader       # KPI summary for a specific role
/leaderboard            # XP leaderboard across all roles
```

**TUI (Ink/React terminal)**
```bash
/roles                  # Full dashboard panel: roles, KPI, XP, achievements, leaderboard
```

**Web UI**
Navigate to **Roles** in the sidebar. The dashboard shows:
- Current role card with tool count
- Level / XP card with progress to next level
- KPI summary cards (success rate, avg tokens, tool diversity, error recovery, proficiency)
- All roles table with descriptions and KPI weights
- Achievements table with unlock dates
- Leaderboard table ranked by XP

---

## Multi-Agent Planning & Auto Memory

Hermes can automatically detect complex multi-step tasks and spawn subagents to parallelize work. After each turn, it also extracts key facts into persistent memory.

### Multi-Agent Planning

When a task is complex (multiple steps, files, or objectives), Hermes automatically breaks it into a plan and delegates to subagents:

```bash
# This happens automatically when a task is detected as complex
# The agent will:
# 1. Break the task into sub-tasks
# 2. Spawn subagents via delegate_task
# 3. Verify results before returning the final answer
```

Planning is handled by `agent/planning_engine.py`. It uses a heuristic to decide when to plan and verifies subagent results for consistency.

### Auto Memory Extraction

After each assistant response, a lightweight auxiliary model extracts key facts and stores them in `~/.hermes/memories/<session_id>.jsonl`:

- Facts are deduplicated using SHA-256 hashes
- Stored per-session for cross-session recall
- Low latency: uses a fast model for extraction

---

## IDE Bridge & Voice Input

### IDE Remote Control (VS Code / JetBrains)

Hermes can communicate directly with your IDE via a simple JSON-RPC protocol over TCP.

```bash
# Set the IDE port (default: 9876)
export HERMES_IDE_PORT=9876
```

Available tools when connected:
- `ide_read_file(path)` — read a file from the IDE workspace
- `ide_edit_file(path, content, line_start, line_end)` — replace lines
- `ide_navigate(path, line, column)` — open/navigate to a position
- `ide_run_command(command)` — run a command in the IDE terminal

If no IDE is listening, tools gracefully degrade with informative errors.

**Note**: Requires a companion IDE plugin/extension that listens on the configured TCP port and implements the JSON-RPC methods.

### Voice Input

Transcribe audio files to text using OpenAI Whisper or a local STT engine.

```bash
# Uses OpenAI Whisper API if OPENAI_API_KEY is set
# Falls back to local `whisper` CLI if available
```

Tools:
- `voice_transcribe(audio_path, model, language)` — transcribe audio to text
- `voice_list_models()` — list available STT backends

---

## MCP & LSP Server Discovery

Hermes includes enhanced MCP (Model Context Protocol) support with automatic Language Server Protocol (LSP) discovery.

```bash
# Discover available LSP servers on PATH
/mcp discover_lsp

# Manually refresh MCP tools from connected servers
/mcp refresh_tools [server_name]

# Check health of all connected MCP servers
/mcp health_check
```

Discovered LSP servers include common language servers across TypeScript, Rust, Python, Go, C/C++, Ruby, Lua, Scala, Kotlin, Dart, and more. Results are cached for the session.

---

## Rust Execution Sandbox

Hermes can compile and run Rust code in a temporary sandboxed environment via `rustc` and `cargo`.

```bash
# Requires Rust toolchain (rustc + cargo) on PATH
# Install from https://rustup.rs
```

Tools:
- `rust_compile(code, edition, args, stdin)` — compile a single `.rs` file and run the binary
- `rust_cargo_run(files, command, args, timeout)` — run a multi-file Cargo project from a dict of file contents
- `rust_version()` — check installed rustc and cargo versions

All compilation happens in temporary directories with automatic cleanup. Output is trimmed to 50KB. Available to `quant-trader`, `propfirm-trader`, `fullstack-dev`, and `system-engineer` roles.

---

## Deterministic Simulation

Run reproducible Monte Carlo and discrete-event simulations with seeded random number generation.

Tools:
- `sim_run(kind, steps, seed, config)` — run a simulation (`monte_carlo` or `discrete_event`)
- `sim_monte_carlo_option(spot, strike, maturity, risk_free_rate, volatility, num_paths, seed, option_type)` — European option pricing via GBM paths
- `sim_save_state(state)` — serialize simulation state to `~/.hermes/sim_states/`
- `sim_load_state(state_id)` — load a previously saved state

Simulations use `numpy.random.default_rng(seed)` for full reproducibility. Results are trimmed to 1,000 steps for model context safety. Available to `quant-trader`, `propfirm-trader`, `fullstack-dev`, and `system-engineer` roles.

---

## Quantitative Math Tools

A suite of financial mathematics functions for trading and portfolio analysis.

Tools:
- `quant_black_scholes(spot, strike, maturity, risk_free_rate, volatility, option_type)` — option price + Greeks (delta, gamma, theta, vega, rho)
- `quant_var(returns, confidence, method)` — Value at Risk (historical or parametric)
- `quant_sharpe_ratio(returns, risk_free_rate, periods_per_year)` — Sharpe ratio + annualized metrics
- `quant_portfolio_optimize(expected_returns, covariance_matrix, target_return, risk_free_rate)` — mean-variance optimization with efficient frontier
- `quant_correlation_matrix(series)` — correlation matrix from asset price/return series
- `quant_drawdown(prices)` — maximum drawdown analysis with duration and series

Available to `quant-trader`, `propfirm-trader`, `fullstack-dev`, and `system-engineer` roles. Uses `numpy` (and `scipy.optimize` when available) for calculations.

---

## Claude Code Integration

Hermes integrates with [Claude Code](https://claude.ai/code) to spawn sub-agents and manage persistent memory across both systems.

### Claude Code Sub-Agent Tool

Spawn Claude Code sub-agents directly from Hermes for tasks that benefit from Claude Code's specific capabilities.

```python
# Single sub-agent
claude_subagent(
    goal="Refactor the authentication module to use JWT tokens",
    context="File: auth.py, current implementation uses session cookies",
    model="claude-sonnet-4-6",
    allowed_tools=["Bash", "Edit", "Read"],
    permission_mode="bypassPermissions",
    bare=True,
)

# Parallel batch
claude_subagent_batch(
    tasks=[
        {"goal": "Write unit tests for login", "allowed_tools": ["Bash", "Edit"]},
        {"goal": "Write integration tests for OAuth", "allowed_tools": ["Bash", "Edit"]},
    ],
    max_concurrent=2,
)
```

**Key features:**
- Non-interactive execution via `claude --print`
- Custom agent personalities via `--agents` JSON
- Tool restriction for security
- Batch parallel execution (max 5 tasks, 3 concurrent default)
- Progress callbacks relayed to parent agent

**Environment variable:** `CLAUDE_CLI_PATH` — override the `claude` binary location.

### Claude Code Memory Manager

Read, write, and sync Claude Code's persistent memory files (`.claude/memory/`) from within Hermes.

Tools:
- `claude_memory_list` — list all memory files
- `claude_memory_read(name)` — read a memory file
- `claude_memory_write(name, body, mem_type, description)` — write/update a memory file
- `claude_memory_delete(name)` — delete a memory file
- `claude_memory_read_index` — read MEMORY.md index
- `claude_memory_read_claude_md` — read project's CLAUDE.md
- `claude_memory_write_claude_md(content)` — write project's CLAUDE.md
- `claude_memory_sync_to_hermes` — sync Claude memories to Hermes store
- `claude_memory_sync_from_hermes` — sync Hermes memories to Claude store

**Memory types:** `user`, `feedback`, `project`, `reference`

Available to all 6 built-in roles.

---

## Testing & Validation

Hermes ships with a comprehensive integration test suite covering all role-based, gamification, and Claude Code features.

### Running Integration Tests

```bash
# All integration tests
python -m pytest tests/integration/ -m integration -v

# Specific component tests
python -m pytest tests/integration/test_role_gamification_integration.py -v
python -m pytest tests/integration/test_web_server_api.py -v
python -m pytest tests/integration/test_cli_role_commands.py -v
python -m pytest tests/integration/test_tui_gateway_rpc.py -v
python -m pytest tests/integration/test_claude_integration.py -v
```

### Test Coverage Summary

| Component | Tests | What’s Covered |
|-----------|-------|---------------|
| Role Manager + Gamification | 37 | Role loading, KPI tracking, XP/level, achievements, leaderboard |
| Web Server REST API | 16 | `GET /api/roles`, `/api/kpi`, `/api/xp`, `/api/achievements`, `/api/leaderboard` |
| CLI Slash Commands | 26 | `/role list\|switch`, `/kpi`, `/leaderboard` handlers |
| TUI Gateway RPC | 19 | `role.list\|get\|switch`, `kpi.summary`, `xp.status`, `achievements.list`, `leaderboard` |
| Claude Code Integration | 21 | Sub-agent spawning, memory CRUD, sync, tool wrappers |

All tests use the `_isolate_hermes_home` autouse fixture from `conftest.py` to ensure hermetic isolation — no writes to the real `~/.hermes/` during test runs.

---

## Documentation

All documentation lives at **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**:

| Section | What's Covered |
|---------|---------------|
| [Quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | Install → setup → first conversation in 2 minutes |
| [CLI Usage](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | Commands, keybindings, personalities, sessions |
| [Configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | Config file, providers, models, all options |
| [Messaging Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant |
| [Security](https://hermes-agent.nousresearch.com/docs/user-guide/security) | Command approval, DM pairing, container isolation |
| [Tools & Toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | 40+ tools, toolset system, terminal backends |
| [Skills System](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | Procedural memory, Skills Hub, creating skills |
| [Memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | Persistent memory, user profiles, best practices |
| [MCP Integration](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | Connect any MCP server for extended capabilities |
| [Cron Scheduling](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | Scheduled tasks with platform delivery |
| [Context Files](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | Project context that shapes every conversation |
| [Architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | Project structure, agent loop, key classes |
| [Contributing](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) | Development setup, PR process, code style |
| [CLI Reference](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) | All commands and flags |
| [Environment Variables](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | Complete env var reference |

---

## Migrating from OpenClaw

If you're coming from OpenClaw, Hermes can automatically import your settings, memories, skills, and API keys.

**During first-time setup:** The setup wizard (`hermes setup`) automatically detects `~/.openclaw` and offers to migrate before configuration begins.

**Anytime after install:**

```bash
hermes claw migrate              # Interactive migration (full preset)
hermes claw migrate --dry-run    # Preview what would be migrated
hermes claw migrate --preset user-data   # Migrate without secrets
hermes claw migrate --overwrite  # Overwrite existing conflicts
```

What gets imported:
- **SOUL.md** — persona file
- **Memories** — MEMORY.md and USER.md entries
- **Skills** — user-created skills → `~/.hermes/skills/openclaw-imports/`
- **Command allowlist** — approval patterns
- **Messaging settings** — platform configs, allowed users, working directory
- **API keys** — allowlisted secrets (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **TTS assets** — workspace audio files
- **Workspace instructions** — AGENTS.md (with `--workspace-target`)

See `hermes claw migrate --help` for all options, or use the `openclaw-migration` skill for an interactive agent-guided migration with dry-run previews.

---

## Contributing

We welcome contributions! See the [Contributing Guide](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) for development setup, code style, and PR process.

Quick start for contributors — clone and go with `setup-hermes.sh`:

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
./setup-hermes.sh     # installs uv, creates venv, installs .[all], symlinks ~/.local/bin/hermes
./hermes              # auto-detects the venv, no need to `source` first
```

Manual path (equivalent to the above):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
python -m pytest tests/ -q
```

> **RL Training (optional):** To work on the RL/Tinker-Atropos integration:
> ```bash
> git submodule update --init tinker-atropos
> uv pip install -e "./tinker-atropos"
> ```

---

## Community

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 💡 [Discussions](https://github.com/NousResearch/hermes-agent/discussions)
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — Community WeChat bridge: Run Hermes Agent and OpenClaw on the same WeChat account.

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Nous Research](https://nousresearch.com).
