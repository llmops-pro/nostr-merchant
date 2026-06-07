# nostr-merchant

**The AI that runs your NOSTR business.** A local-first Python agent that holds the whole merchant toolkit — it checks its Lightning wallet, publishes notes and listings, runs a Shopstr storefront, answers encrypted DMs, and pays for paywalled MCP tools over NWC — all under sats budget caps it can't exceed. Built on five MCP servers ([`nwc-mcp`](https://npmjs.com/package/nwc-mcp), [`nostr-ops-mcp`](https://npmjs.com/package/nostr-ops-mcp), [`marketplace-mcp`](https://npmjs.com/package/marketplace-mcp), [`albyhub-admin-mcp`](https://npmjs.com/package/albyhub-admin-mcp), [`paywall-mcp`](https://npmjs.com/package/paywall-mcp)) — 37 tools the agent picks from.

Ollama-first, API-pluggable. Built on [`pydantic-ai`](https://ai.pydantic.dev) — no LangChain. MIT.

> **v0.2 — renamed from `llmops-agent`.** All six CLI commands wired. The agent layer enforces a budget-and-audit safety pipeline on top of every MCP server's own safety stack. This release ships the full toolkit, the safety stack, and a working self-paying loop as proof the agent-pays-for-tools path holds end to end (given a working LLM backend — Ollama / Anthropic / OpenAI). Directing it at your own merchant tasks is what it's for today; unattended scheduled storefront-tending is the roadmap.

---

## Install

```bash
pipx install nostr-merchant          # recommended: isolated CLI
# or:  uvx --from nostr-merchant nostr-merchant --help
# or:  pip install nostr-merchant
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
uv run nostr-merchant --help
uv run nostr-merchant version
uv run nostr-merchant budget                # snapshot, no LLM needed
uv run nostr-merchant config-print          # effective config, secrets masked
uv run nostr-merchant doctor                # ping each configured MCP server
uv run nostr-merchant ask "What's my Lightning wallet balance?"   # the main demo loop
uv run nostr-merchant audit --tail 20       # recent audit entries
```

Or run the whole sequence with the bundled script: `./smoke.sh` (full run — the `ask` is read-only, so it can never move sats) or `./smoke.sh --no-ask` (skip the LLM call entirely).

`ask` requires a working LLM backend, set via `NOSTR_MERCHANT_MODEL`:

- **`anthropic:claude-haiku-4-5-20251001`** (or any Anthropic/OpenAI model) — fast, reliable tool-calling. Set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) in `~/.nostr-merchant/.env`. Recommended for most setups.
- **`ollama:<model>`** — fully local, no KYC, no phone-home (the sovereignty default). Needs Ollama at `localhost:11434`. Reality check: a 37-tool agent loop wants real hardware — a small model on a CPU-only box is too slow to be practical. Run Ollama on a GPU, or use a hosted model and keep your wallet + keys local (the trust boundary that actually matters).

Provider creds in `~/.nostr-merchant/.env` are loaded automatically — no need to `export` them.

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
