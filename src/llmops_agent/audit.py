"""NDJSON audit log for llmops-agent.

Same shape as the substrate MCP servers' audit logs so that tailing the
agent + MCP audit logs side by side reads coherently. One JSON object per
line, with a fixed core schema and free-form `input` / `result` / `error`
extension fields.

Core schema:

    {
      "ts": "2026-05-31T20:14:33.412Z",         # ISO-8601 UTC
      "kind": "tool_call" | "llm_call" | "startup" | "shutdown" | "budget_block",
      "outcome": "ok" | "error" | "blocked",
      "tool": "...",                            # set for tool_call entries
      "input": {...},                            # tool args, prompt summary, etc.
      "result": {...},                           # tool output summary or LLM result summary
      "error": "...",                            # set when outcome="error"
      "blocked_reason": "..."                    # set when outcome="blocked"
    }
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _iso_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class AuditLog:
    """Append-only NDJSON writer with a small async lock to serialize writes.

    The lock is process-local — multi-process scenarios (which we do not
    support in v0.1) would need a file lock instead.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    async def record(
        self,
        *,
        kind: str,
        outcome: str,
        tool: str | None = None,
        input: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        blocked_reason: str | None = None,
    ) -> None:
        """Append a structured event. Never raises — IO failures go to stderr."""
        event: dict[str, Any] = {
            "ts": _iso_now(),
            "kind": kind,
            "outcome": outcome,
        }
        if tool is not None:
            event["tool"] = tool
        if input is not None:
            event["input"] = input
        if result is not None:
            event["result"] = result
        if error is not None:
            event["error"] = error
        if blocked_reason is not None:
            event["blocked_reason"] = blocked_reason

        line = json.dumps(event, separators=(",", ":"), default=str) + "\n"
        async with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                # File IO is sync; the lock serializes calls so concurrent
                # tasks don't interleave their lines.
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
            except OSError as err:
                # Audit log failure must NEVER take down the agent. Print
                # to stderr and continue.
                import sys

                print(
                    f"llmops-agent: audit append failed ({type(err).__name__}: {err})",
                    file=sys.stderr,
                )

    # ---- Convenience wrappers for the call sites that will use this ----

    async def record_tool_call(
        self,
        *,
        tool: str,
        outcome: str,
        input: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        blocked_reason: str | None = None,
    ) -> None:
        await self.record(
            kind="tool_call",
            outcome=outcome,
            tool=tool,
            input=input,
            result=result,
            error=error,
            blocked_reason=blocked_reason,
        )

    async def record_llm_call(
        self,
        *,
        outcome: str,
        input: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        await self.record(
            kind="llm_call",
            outcome=outcome,
            input=input,
            result=result,
            error=error,
        )

    async def record_budget_block(
        self,
        *,
        tool: str,
        sats: int,
        reason: str,
    ) -> None:
        await self.record(
            kind="budget_block",
            outcome="blocked",
            tool=tool,
            input={"sats": sats},
            blocked_reason=reason,
        )

    async def record_startup(self, snapshot: dict[str, Any]) -> None:
        await self.record(kind="startup", outcome="ok", result=snapshot)

    async def record_shutdown(self, signal: str) -> None:
        await self.record(kind="shutdown", outcome="ok", result={"signal": signal})
