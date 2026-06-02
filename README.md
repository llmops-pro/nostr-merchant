# llmops-agent

**Reference Python agent for the LLMOps.Pro sovereign-AI substrate.** Consumes the five MCP servers ([`nwc-mcp`](https://npmjs.com/package/nwc-mcp), [`nostr-ops-mcp`](https://npmjs.com/package/nostr-ops-mcp), [`marketplace-mcp`](https://npmjs.com/package/marketplace-mcp), [`albyhub-admin-mcp`](https://npmjs.com/package/albyhub-admin-mcp), [`paywall-mcp`](https://npmjs.com/package/paywall-mcp)) and demonstrates the agent-pays-paid-MCP loop end-to-end.

Ollama-first, API-pluggable. Built on [`pydantic-ai`](https://ai.pydantic.dev) — no LangChain. MIT.

> **v0.1 operational.** All six CLI commands wired. Agent layer enforces a budget-and-audit safety pipeline on top of every MCP server's own safety stack. Self-paying research workflow runs end-to-end given a working LLM backend (Ollama / Anthropic / OpenAI). Full design at [`../03-python-reference-agent-design.md`](../03-python-reference-agent-design.md).

---

## Install

```bash
pipx install llmops-agent          # recommended: isolated CLI
# or:  uvx --from llmops-agent llmops-agent --help
# or:  pip install llmops-agent
```

The five MCP servers it drives are launched on demand via `npx -y` — no separate install. You need Python 3.11+, an NWC-compatible wallet (e.g. Alby Hub), and an LLM backend (see [Usage](#usage)).

---

## Status

| Layer | State |
|---|---|
| Design doc | ✅ Done — [`../03-python-reference-agent-design.md`](../03-python-reference-agent-design.md) |
| Package scaffold (`uv init`, `pyproject.toml`, `ruff`/`mypy`/`pytest` config) | ✅ Done |
| `config.py` — env-driven Pydantic Settings | ✅ Done |
| `budget.py` — persistent rolling-window tracker | ✅ Done |
| `audit.py` — NDJSON writer | ✅ Done |
| `mcp_servers.py` — `MCPServerStdio` launch + doctor probe | ✅ Done |
| `agent.py` — Pydantic AI agent + `process_tool_call` middleware | ✅ Done |
| `workflows/research.py` — self-paying research workflow | ✅ Done |
| `cli.py` — typer entry-point with `ask` / `doctor` / `budget` / `audit` / `config-print` / `version` | ✅ Done |
| Live LLM smoke test (Anthropic Haiku) | ✅ Done — end-to-end `ask` returns a real `nwc_get_balance` receipt with all 37 substrate tools loaded |

## Usage

```bash
# from this directory, with uv-installed venv
uv run llmops-agent --help
uv run llmops-agent version
uv run llmops-agent budget                # snapshot, no LLM needed
uv run llmops-agent config-print          # effective config, secrets masked
uv run llmops-agent doctor                # ping each configured MCP server
uv run llmops-agent ask "What's my Lightning wallet balance?"   # the main demo loop
uv run llmops-agent audit --tail 20       # recent audit entries
```

Or run the whole sequence with the bundled script: `./smoke.sh` (full run — the `ask` is read-only, so it can never move sats) or `./smoke.sh --no-ask` (skip the LLM call entirely).

`ask` requires a working LLM backend, set via `LLMOPS_MODEL`:

- **`anthropic:claude-haiku-4-5-20251001`** (or any Anthropic/OpenAI model) — fast, reliable tool-calling. Set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) in `~/.llmops-agent/.env`. Recommended for most setups.
- **`ollama:<model>`** — fully local, no KYC, no phone-home (the sovereignty default). Needs Ollama at `localhost:11434`. Reality check: a 37-tool agent loop wants real hardware — a small model on a CPU-only box is too slow to be practical. Run Ollama on a GPU, or use a hosted model and keep your wallet + keys local (the trust boundary that actually matters).

Provider creds in `~/.llmops-agent/.env` are loaded automatically — no need to `export` them.

---

## Dev loop

```bash
uv sync                 # install deps + dev deps into .venv
uv run ruff check       # lint
uv run mypy             # type check
uv run pytest           # tests
```

Build wheel for distribution:

```bash
uv build
```

## License

MIT — see [`LICENSE`](./LICENSE).

## Contact

Built by **LLMOps.Pro**.

- **NOSTR:** [`npub1hdg932jvwc3jdvkqywgqv0ue4nn60exrf92asy8mtazt3hjg7d2s2yw0nw`](https://njump.me/npub1hdg932jvwc3jdvkqywgqv0ue4nn60exrf92asy8mtazt3hjg7d2s2yw0nw)
- **Lightning Address:** `sovereigncitizens@getalby.com`
- **Shopstr:** [shopstr.store/marketplace/SOVEREIGN_CITIZENS](https://shopstr.store/marketplace/SOVEREIGN_CITIZENS)
