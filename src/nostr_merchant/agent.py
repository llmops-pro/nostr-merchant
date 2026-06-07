"""Pydantic AI agent assembly.

Builds the `Agent` instance with:

- A configured model (`NOSTR_MERCHANT_MODEL` env, e.g. `ollama:qwen3:8b`).
- The five substrate MCP servers attached as tool surfaces.
- A `process_tool_call` middleware on every server that enforces the
  agent-layer budget caps and writes audit entries — sits ON TOP of each
  MCP server's own safety stack.
- A system prompt that teaches the LLM:
  - How to use tools, with explicit examples.
  - The dual-call paywall pattern (call without `payment_hash`, get an
    invoice + payment_hash, call `nwc__pay_invoice` to settle, retry the
    original tool with `payment_hash`).
  - Budget awareness — refuse calls above caps.
  - Citation discipline — every fact obtained from a paid tool MUST be
    accompanied by a receipt in the final answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent

from .mcp_servers import build_mcp_servers

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic_ai.models import KnownModelName, Model

    from .audit import AuditLog
    from .budget import BudgetTracker
    from .config import AgentConfig


SYSTEM_PROMPT = """\
You are nostr-merchant — a sovereign AI assistant with Lightning-paid tools.

# How to use tools

You have access to multiple MCP servers — each exposes tools with a natural
prefix that identifies the server (e.g. `nwc_get_balance` for the Lightning
wallet, `nostr_publish_text_note` for NOSTR identity, `paywall_*` for paid
upstream tools). Pick the right tool, pass the right args, then interpret
the JSON result.

# The Lightning paywall pattern

Some tools cost sats. When you call a paywalled tool WITHOUT a `payment_hash`
argument, the response will look like:

    {
      "error": "payment_required",
      "invoice": "lnbc...",
      "payment_hash": "abc123...",
      "amount_sats": 21,
      "expires_in_seconds": 600,
      "next_step": "..."
    }

To complete the call, you MUST:

1. Check that the price (`amount_sats`) is reasonable for the value you're
   trying to extract. If the price seems unreasonable for the task, REFUSE
   and explain to the user — don't burn sats reflexively.
2. Pay the invoice by calling `nwc_pay_invoice` with the `invoice` string.
3. Once paid, RETRY the original tool with the SAME args PLUS the
   `payment_hash` from step 1.
4. The second call returns the upstream tool's real result.

NEVER claim a fact obtained from a paid tool without including a receipt in
your final answer. Receipts list: tool name, sats paid, timestamp, payment_hash
(first 12 chars is enough). Receipts go in a `## Receipts` markdown section
at the end of your answer.

# Budget discipline

You have a per-task budget cap and a per-day budget cap (the operator
configures these). The wallet enforces its own caps too. If you hit a cap,
stop and report the situation honestly — don't try workarounds.

# Free tools

Free tools (price 0) pass through immediately. Use them liberally:
`nwc_get_balance`, `nwc_decode_invoice`, `nostr_list_relays`, etc.

# Citation discipline

For every external fact you state, cite the tool that gave it to you.
Example: "BTC block height: 894,271 (via paywall_bitcoin_block_height,
21 sats paid)."

If the user just asks for an opinion or a chat, you do not need to call
any tools — just answer. Tools are for facts, transactions, and external
state.
"""


def _resolve_model(config: AgentConfig) -> Model | KnownModelName | str:
    """Turn the NOSTR_MERCHANT_MODEL env string into something `Agent(model=...)` accepts.

    Pydantic AI accepts the model-string form directly via `infer_model`,
    so we just pass through. Provider-specific env vars (ANTHROPIC_API_KEY,
    OPENAI_API_KEY, OLLAMA_BASE_URL) are read by Pydantic AI's provider
    classes from `os.environ` — but values from our `.env` file only live on
    the config object, so we bridge them across first. Without this, an `ask`
    only works if the operator manually `export`ed the keys.
    """
    config.apply_provider_env()
    return config.NOSTR_MERCHANT_MODEL


def build_agent(
    config: AgentConfig,
    *,
    process_tool_call: Callable[..., Awaitable[Any]] | None = None,
    model_override: Model | KnownModelName | str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> Agent[None, str]:
    """Assemble the Pydantic AI Agent.

    `process_tool_call` is the per-server middleware hook. Typically built
    by `workflows.research` so it can close over the budget tracker + audit
    log instances for this particular run.

    `model_override` lets tests inject a `TestModel` without round-tripping
    through env vars.
    """
    model = model_override if model_override is not None else _resolve_model(config)
    mcp_servers = build_mcp_servers(config, process_tool_call=process_tool_call)
    agent: Agent[None, str] = Agent(
        model=model,
        toolsets=list(mcp_servers),
        system_prompt=system_prompt,
    )
    return agent


def make_process_tool_call(
    *,
    budget: BudgetTracker,
    audit: AuditLog,
    max_tool_price: int,
    tool_allowlist: set[str] | None,
    read_only: bool,
) -> Callable[..., Awaitable[Any]]:
    """Build the middleware closure attached to every MCP server.

    Pipeline per tool call:
      1. tool_allowlist check (when set)
      2. read_only check (refuse priced tools — heuristic: any tool from the
         `paywall` server, or a previously-issued `payment_required` response)
      3. forward to the upstream tool
      4. inspect the result for `payment_required`:
           - if found: enforce `AGENT_MAX_TOOL_PRICE` and budget caps; if
             the price is acceptable, leave the response intact for the LLM
             to act on; if not, replace it with a refusal payload
           - if not: pass through
      5. for `nwc_pay_invoice` calls: record the spend after a successful
         settlement
      6. audit every step
    """

    async def process(
        ctx: Any,
        call_tool: Callable[..., Awaitable[Any]],
        tool_name: str,
        args: dict[str, Any],
    ) -> Any:
        # 1. allowlist
        if tool_allowlist is not None and tool_name not in tool_allowlist:
            await audit.record_tool_call(
                tool=tool_name,
                outcome="blocked",
                input={"tool_allowlist_active": True},
                blocked_reason=f"tool {tool_name!r} not in AGENT_TOOL_ALLOWLIST",
            )
            return {
                "error": "tool_not_in_allowlist",
                "message": (
                    f"The tool {tool_name!r} is not in the agent's "
                    f"AGENT_TOOL_ALLOWLIST. The operator must add it to enable use."
                ),
            }

        # 2. read-only check (best-effort heuristic: only blocks paywall_*
        #    tools and the nwc_pay_* tool family directly)
        if read_only and _is_likely_priced(tool_name):
            await audit.record_tool_call(
                tool=tool_name,
                outcome="blocked",
                input={"agent_read_only": True},
                blocked_reason="AGENT_READ_ONLY=true",
            )
            return {
                "error": "agent_read_only",
                "message": (
                    "Agent is in read-only mode (AGENT_READ_ONLY=true). "
                    f"Tool {tool_name!r} refused."
                ),
            }

        # 3. forward
        try:
            result = await call_tool(tool_name, args)
        except Exception as err:
            await audit.record_tool_call(
                tool=tool_name,
                outcome="error",
                input={"args_keys": sorted(args.keys())},
                error=f"{type(err).__name__}: {err}",
            )
            raise

        # 4. payment_required inspection
        payment_required = _extract_payment_required(result)
        if payment_required is not None:
            price_sats = payment_required.get("amount_sats")
            if isinstance(price_sats, int) and price_sats > 0:
                if price_sats > max_tool_price:
                    await audit.record_budget_block(
                        tool=tool_name,
                        sats=price_sats,
                        reason=(
                            f"price {price_sats} exceeds "
                            f"AGENT_MAX_TOOL_PRICE={max_tool_price}"
                        ),
                    )
                    return {
                        "error": "agent_max_tool_price_exceeded",
                        "price_sats": price_sats,
                        "cap_sats": max_tool_price,
                        "message": (
                            f"Tool {tool_name!r} demands {price_sats} sats but "
                            f"AGENT_MAX_TOOL_PRICE={max_tool_price}. Refused."
                        ),
                    }
                ok, reason = budget.can_spend(price_sats)
                if not ok:
                    await audit.record_budget_block(
                        tool=tool_name,
                        sats=price_sats,
                        reason=reason or "budget_check_failed",
                    )
                    return {
                        "error": "agent_budget_exceeded",
                        "price_sats": price_sats,
                        "reason": reason,
                        "message": (
                            f"Tool {tool_name!r} requested {price_sats} sats "
                            f"but agent budget refuses: {reason}. "
                            f"Stop and report to the user."
                        ),
                    }
            # Audit the issue-invoice step.
            await audit.record_tool_call(
                tool=tool_name,
                outcome="ok",
                input={"args_keys": sorted(args.keys()), "stage": "invoice_received"},
                result={
                    "amount_sats": price_sats,
                    "payment_hash": payment_required.get("payment_hash"),
                },
            )
            return result

        # 5. nwc_pay_invoice spend bookkeeping
        if tool_name == "nwc_pay_invoice":
            spent = _parse_paid_sats(result)
            if spent > 0:
                budget.record_spend(sats=spent, tool=tool_name)
            await audit.record_tool_call(
                tool=tool_name,
                outcome="ok",
                input={"args_keys": sorted(args.keys())},
                result={"sats_recorded": spent},
            )
            return result

        # 6. default — passthrough audit
        await audit.record_tool_call(
            tool=tool_name,
            outcome="ok",
            input={"args_keys": sorted(args.keys())},
            result={"stage": "passthrough"},
        )
        return result

    return process


def _is_likely_priced(tool_name: str) -> bool:
    """Heuristic for read-only mode — refuse obvious-spend tools."""
    if tool_name.startswith("paywall_"):
        return True
    return tool_name.startswith("nwc_pay") or tool_name == "nwc_multi_pay_invoice"


def _extract_payment_required(result: Any) -> dict[str, Any] | None:
    """Return the inner `payment_required` payload if the result is one, else None.

    Pydantic AI surfaces tool results in a few shapes (dict, string, structured
    content list). We look for the common shapes used by paywall-mcp and
    paywall-mcp-test.
    """
    if isinstance(result, dict) and result.get("error") == "payment_required":
        return result
    if isinstance(result, str):
        # paywall-mcp returns its body as a JSON string sometimes; try to parse.
        try:
            import json

            parsed = json.loads(result)
            if isinstance(parsed, dict) and parsed.get("error") == "payment_required":
                return parsed
        except (ValueError, TypeError):
            return None
    return None


def _parse_paid_sats(result: Any) -> int:
    """Extract sats actually paid from a successful `nwc_pay_invoice` response.

    Best-effort: looks at common fields (`amount_sats`, `amount`, `paid_sats`).
    Returns 0 if it can't tell.
    """
    candidate: Any = None
    if isinstance(result, dict):
        candidate = result
    elif isinstance(result, str):
        try:
            import json

            candidate = json.loads(result)
        except (ValueError, TypeError):
            return 0
    if not isinstance(candidate, dict):
        return 0
    for key in ("amount_sats", "paid_sats", "amount"):
        val = candidate.get(key)
        if isinstance(val, int) and val >= 0:
            return val
    return 0
