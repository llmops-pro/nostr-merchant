"""Typer CLI entry-point for nostr-merchant.

Commands per design doc §8:
  - `nostr-merchant ask "<question>"`         — main demo loop
  - `nostr-merchant doctor`                   — MCP server health probes
  - `nostr-merchant budget`                   — current spend snapshot
  - `nostr-merchant audit [--tail N]`         — recent audit entries
  - `nostr-merchant config-print`             — effective config (secrets masked)
  - `nostr-merchant version`                  — installed version
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from . import __version__
from .budget import BudgetTracker
from .config import AgentConfig
from .mcp_servers import doctor_check
from .workflows.engagement import run_inbox
from .workflows.research import run_research

app = typer.Typer(
    name="nostr-merchant",
    help="Reference Python agent for the LLMOps.Pro sovereign-AI substrate.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
console = Console()


def _load_config() -> AgentConfig:
    """Load config + halt with a friendly stderr message on failure."""
    try:
        return AgentConfig()
    except Exception as err:
        console.print(
            Panel(
                f"[red]Config load failed:[/red]\n\n{err}\n\n"
                f"[dim]Check ~/.nostr-merchant/.env or the env vars in your shell.[/dim]",
                title="nostr-merchant",
                border_style="red",
            ),
        )
        raise typer.Exit(code=1) from err


# -------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the installed nostr-merchant version."""
    typer.echo(__version__)


@app.command()
def ask(
    question: Annotated[
        str,
        typer.Argument(help="The question or task for the agent."),
    ],
) -> None:
    """Ask the agent a question — the main demo loop.

    The agent will plan, call MCP tools (free + paid), pay invoices via the
    `nwc` server when needed, and return a written answer with receipts
    for any paid facts.
    """
    config = _load_config()
    console.print(
        Panel(
            f"[bold]{question}[/bold]\n\n"
            f"[dim]model: {config.NOSTR_MERCHANT_MODEL}  ·  "
            f"per-task cap: {config.AGENT_MAX_SATS_PER_TASK} sats  ·  "
            f"per-day cap: {config.AGENT_MAX_SATS_PER_DAY} sats[/dim]",
            title="nostr-merchant ask",
            border_style="cyan",
        ),
    )
    try:
        result = asyncio.run(run_research(question, config=config))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130) from None
    except Exception as err:
        console.print(
            Panel(
                f"[red]{type(err).__name__}: {err}[/red]\n\n"
                f"[dim]Audit log: {config.AGENT_AUDIT_PATH}[/dim]",
                title="agent error",
                border_style="red",
            ),
        )
        raise typer.Exit(code=1) from err

    console.print(Panel(result.answer, title="answer", border_style="green"))

    table = Table(title="budget", show_header=True, header_style="bold cyan")
    table.add_column("window", style="dim")
    table.add_column("before")
    table.add_column("after")
    table.add_column("delta")
    table.add_row(
        "this task (sats)",
        str(result.budget_before.per_task_spent_sats),
        str(result.budget_after.per_task_spent_sats),
        str(
            result.budget_after.per_task_spent_sats
            - result.budget_before.per_task_spent_sats,
        ),
    )
    table.add_row(
        "today (sats, rolling 24h)",
        str(result.budget_before.spent_today_sats),
        str(result.budget_after.spent_today_sats),
        str(
            result.budget_after.spent_today_sats
            - result.budget_before.spent_today_sats,
        ),
    )
    table.add_row(
        "lifetime (sats)",
        str(result.budget_before.spent_lifetime_sats),
        str(result.budget_after.spent_lifetime_sats),
        str(
            result.budget_after.spent_lifetime_sats
            - result.budget_before.spent_lifetime_sats,
        ),
    )
    console.print(table)
    console.print(f"[dim]Audit log: {config.AGENT_AUDIT_PATH}[/dim]")


@app.command()
def inbox(
    since: Annotated[
        int,
        typer.Option("--since", "-s", help="Look back this many hours."),
    ] = 48,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Max inbound items to triage."),
    ] = 20,
) -> None:
    """Triage replies/mentions on your recent NOSTR posts and draft responses.

    READ-ONLY: the agent is restricted to nostr-ops-mcp's read tools, so it cannot
    publish — it produces a review queue of drafts. You post the approved ones yourself.
    """
    config = _load_config()
    console.print(
        Panel(
            f"[bold]Gathering engagement — last {since}h, up to {limit} items[/bold]\n\n"
            f"[dim]model: {config.NOSTR_MERCHANT_MODEL}  ·  "
            f"READ-ONLY (drafts only — nothing is published)[/dim]",
            title="nostr-merchant inbox",
            border_style="cyan",
        ),
    )
    try:
        result = asyncio.run(
            run_inbox(
                config=config,
                since_hours=since,
                limit=limit,
                on_progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
            ),
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130) from None
    except Exception as err:
        console.print(
            Panel(
                f"[red]{type(err).__name__}: {err}[/red]\n\n"
                f"[dim]Audit log: {config.AGENT_AUDIT_PATH}[/dim]",
                title="inbox error",
                border_style="red",
            ),
        )
        raise typer.Exit(code=1) from err

    console.print(
        Panel(
            result.queue,
            title=f"engagement queue — {result.item_count} item(s), drafts only (nothing posted)",
            border_style="green",
        ),
    )
    console.print(
        "[dim]v1 is read-only — nothing was published. Post the replies you approve "
        "yourself (an approval-gated `inbox --post` is the next step).[/dim]",
    )


@app.command()
def doctor() -> None:
    """Ping every configured MCP server. Prints a status table."""
    config = _load_config()
    specs = config.mcp_server_specs()
    console.print(
        Panel(
            f"[bold]Probing {len(specs)} MCP server(s)...[/bold]",
            title="nostr-merchant doctor",
            border_style="cyan",
        ),
    )

    async def run_all() -> None:
        results = await asyncio.gather(
            *(doctor_check(spec) for spec in specs),
            return_exceptions=False,
        )
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("server", style="bold")
        table.add_column("status")
        table.add_column("tools")
        table.add_column("notes")
        for spec, res in zip(specs, results, strict=True):
            launch = f"{spec.command} {' '.join(spec.args)}"
            if res.ok:
                tools_preview = ", ".join(res.tool_names[:6])
                if len(res.tool_names) > 6:
                    tools_preview += f", +{len(res.tool_names) - 6} more"
                table.add_row(
                    res.name,
                    "[green]ok[/green]",
                    str(res.tool_count),
                    tools_preview or "[dim]none[/dim]",
                )
            else:
                table.add_row(
                    res.name,
                    "[red]fail[/red]",
                    "—",
                    f"[red]{res.error}[/red]\n[dim]{launch}[/dim]",
                )
        console.print(table)

    asyncio.run(run_all())


@app.command()
def budget() -> None:
    """Print current spend snapshot — what's used, what's remaining."""
    config = _load_config()
    tracker = BudgetTracker(
        path=config.AGENT_BUDGET_PATH,
        max_per_task_sats=config.AGENT_MAX_SATS_PER_TASK,
        max_per_day_sats=config.AGENT_MAX_SATS_PER_DAY,
    )
    snap = tracker.snapshot()
    table = Table(title="nostr-merchant budget", show_header=True, header_style="bold cyan")
    table.add_column("window", style="dim")
    table.add_column("spent (sats)")
    table.add_column("cap (sats)")
    table.add_column("remaining (sats)")
    table.add_row(
        "this task",
        str(snap.per_task_spent_sats),
        str(config.AGENT_MAX_SATS_PER_TASK),
        str(snap.per_task_remaining_sats),
    )
    table.add_row(
        "today (rolling 24h)",
        str(snap.spent_today_sats),
        str(config.AGENT_MAX_SATS_PER_DAY),
        str(snap.today_remaining_sats),
    )
    table.add_row(
        "lifetime",
        str(snap.spent_lifetime_sats),
        "[dim]unbounded[/dim]",
        "—",
    )
    console.print(table)
    console.print(f"[dim]Budget file: {config.AGENT_BUDGET_PATH}[/dim]")


@app.command()
def audit(
    tail: Annotated[
        int,
        typer.Option("--tail", "-n", help="How many trailing entries to print."),
    ] = 20,
) -> None:
    """Pretty-print the trailing N entries from the audit log."""
    config = _load_config()
    path: Path = config.AGENT_AUDIT_PATH
    if not path.exists():
        console.print(f"[yellow]No audit log yet at {path}.[/yellow]")
        raise typer.Exit(code=0)
    lines = path.read_text(encoding="utf-8").splitlines()
    recent = [line for line in lines[-tail:] if line.strip()]
    if not recent:
        console.print(f"[yellow]Audit log at {path} is empty.[/yellow]")
        return
    table = Table(
        title=f"audit log — last {len(recent)} entries",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("ts", style="dim", overflow="fold")
    table.add_column("kind", overflow="fold")
    table.add_column("outcome")
    table.add_column("tool", overflow="fold")
    table.add_column("detail", overflow="fold")
    for line in recent:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = str(entry.get("ts", ""))
        kind = str(entry.get("kind", ""))
        outcome = str(entry.get("outcome", ""))
        tool = str(entry.get("tool", "")) or "—"
        detail = entry.get("error") or entry.get("blocked_reason") or ""
        if not detail and "result" in entry:
            detail = json.dumps(entry["result"], separators=(",", ":"))[:80]
        outcome_style = {
            "ok": "[green]ok[/green]",
            "error": "[red]error[/red]",
            "blocked": "[yellow]blocked[/yellow]",
        }.get(outcome, outcome)
        table.add_row(ts, kind, outcome_style, tool, str(detail))
    console.print(table)


@app.command("config-print")
def config_print() -> None:
    """Dump effective config (secrets masked)."""
    config = _load_config()
    raw = config.model_dump(mode="json")
    for secret_key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if raw.get(secret_key):
            raw[secret_key] = "***"
    json_str = json.dumps(raw, indent=2, default=str)
    console.print(
        Syntax(json_str, "json", theme="ansi_dark", line_numbers=False, word_wrap=True),
    )
