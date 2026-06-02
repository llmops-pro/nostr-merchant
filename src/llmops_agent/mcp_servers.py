"""MCP server launch + lifecycle helpers.

Wraps Pydantic AI's `MCPServerStdio` for our config-driven launch model.
Each MCP server spec from `AgentConfig.mcp_server_specs()` becomes one
`MCPServerStdio` instance.

We deliberately do NOT set `tool_prefix` — the substrate's tools are
already self-prefixed at the server level (e.g. `nwc_get_balance`,
`nostr_publish_event`, `marketplace_create_or_update_stall`), so adding
pydantic-ai's prefix would double-prefix to `nwc_nwc_get_balance`.

Pydantic AI manages the child-process lifecycle (start on context-enter,
stop on context-exit). Each MCP server reads its own `.env` from its own
install location, per the convention we established across the substrate.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic_ai.mcp import MCPServerStdio

if TYPE_CHECKING:
    from .config import AgentConfig, McpServerSpec


# A "process_tool_call" callback — receives the run-context, the actual call
# function, the tool name, and the args; returns whatever the tool returned.
# We type this loosely because Pydantic AI's exact type uses internal types
# we don't import here.
ProcessToolCallT = Callable[..., Awaitable[Any]]


def build_mcp_servers(
    config: AgentConfig,
    *,
    process_tool_call: ProcessToolCallT | None = None,
) -> list[MCPServerStdio]:
    """Construct a `MCPServerStdio` per spec from the config.

    `process_tool_call` is the optional middleware hook attached to every
    server. When provided, every tool call routes through this callback
    (e.g., for budget enforcement + audit logging).
    """
    specs = config.mcp_server_specs()
    servers: list[MCPServerStdio] = []
    for spec in specs:
        kwargs: dict[str, Any] = {
            "command": spec.command,
            "args": list(spec.args),
            "id": spec.name,
        }
        if spec.cwd is not None:
            kwargs["cwd"] = spec.cwd
        if spec.env is not None:
            kwargs["env"] = spec.env
        if process_tool_call is not None:
            kwargs["process_tool_call"] = process_tool_call
        servers.append(MCPServerStdio(**kwargs))
    return servers


async def doctor_check(spec: McpServerSpec, *, timeout: float = 6.0) -> DoctorResult:
    """Spawn the MCP server briefly and confirm it answers `tools/list`.

    Returns a structured result so the CLI can render a status table.
    Never raises — failures are encoded in the result.
    """
    kwargs: dict[str, Any] = {
        "command": spec.command,
        "args": list(spec.args),
        "id": spec.name,
        "timeout": timeout,
    }
    if spec.cwd is not None:
        kwargs["cwd"] = spec.cwd
    if spec.env is not None:
        kwargs["env"] = spec.env

    server = MCPServerStdio(**kwargs)
    try:
        async with asyncio.timeout(timeout + 4):
            async with server:
                tool_list = await server.list_tools()
                names = sorted(t.name for t in tool_list)
                return DoctorResult(
                    name=spec.name,
                    ok=True,
                    tool_count=len(names),
                    tool_names=names,
                    error=None,
                )
    except Exception as err:
        return DoctorResult(
            name=spec.name,
            ok=False,
            tool_count=0,
            tool_names=[],
            error=f"{type(err).__name__}: {err}",
        )


class DoctorResult:
    """Outcome of a single MCP server health probe."""

    __slots__ = ("error", "name", "ok", "tool_count", "tool_names")

    def __init__(
        self,
        *,
        name: str,
        ok: bool,
        tool_count: int,
        tool_names: list[str],
        error: str | None,
    ) -> None:
        self.name = name
        self.ok = ok
        self.tool_count = tool_count
        self.tool_names = tool_names
        self.error = error
