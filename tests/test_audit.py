"""Tests for `llmops_agent.audit`.

NDJSON shape verification + concurrent-write safety (the async lock should
serialize concurrent record calls so lines never interleave).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from llmops_agent.audit import AuditLog


def read_lines(path: Path) -> list[dict[str, object]]:
    """Read a JSON-line-per-entry file into a list of dicts."""
    if not path.exists():
        return []
    out: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        assert isinstance(parsed, dict)
        out.append(parsed)
    return out


@pytest.fixture
def audit_path(tmp_path: Path) -> Path:
    return tmp_path / "audit.log"


class TestNdjsonShape:
    @pytest.mark.asyncio
    async def test_tool_call_ok_shape(self, audit_path: Path) -> None:
        log = AuditLog(audit_path)
        await log.record_tool_call(
            tool="nwc_get_balance",
            outcome="ok",
            input={"price_sats": 0},
            result={"balance_sats": 12345},
        )
        lines = read_lines(audit_path)
        assert len(lines) == 1
        e = lines[0]
        assert isinstance(e["ts"], str)
        assert e["ts"].endswith("Z")
        assert e["kind"] == "tool_call"
        assert e["outcome"] == "ok"
        assert e["tool"] == "nwc_get_balance"
        assert e["input"] == {"price_sats": 0}
        assert e["result"] == {"balance_sats": 12345}

    @pytest.mark.asyncio
    async def test_blocked_entry_includes_reason(self, audit_path: Path) -> None:
        log = AuditLog(audit_path)
        await log.record_budget_block(
            tool="rare_alpha_signal", sats=10_000, reason="agent_max_tool_price_exceeded"
        )
        lines = read_lines(audit_path)
        e = lines[0]
        assert e["kind"] == "budget_block"
        assert e["outcome"] == "blocked"
        assert e["blocked_reason"] == "agent_max_tool_price_exceeded"
        assert e["input"] == {"sats": 10_000}

    @pytest.mark.asyncio
    async def test_error_entry_includes_error_field(self, audit_path: Path) -> None:
        log = AuditLog(audit_path)
        await log.record_tool_call(
            tool="upstream_thing",
            outcome="error",
            input={"price_sats": 21},
            error="connection_refused",
        )
        lines = read_lines(audit_path)
        e = lines[0]
        assert e["outcome"] == "error"
        assert e["error"] == "connection_refused"

    @pytest.mark.asyncio
    async def test_optional_fields_omitted_when_unset(self, audit_path: Path) -> None:
        log = AuditLog(audit_path)
        await log.record_llm_call(outcome="ok")
        e = read_lines(audit_path)[0]
        assert e["kind"] == "llm_call"
        assert e["outcome"] == "ok"
        assert "tool" not in e
        assert "input" not in e
        assert "result" not in e
        assert "error" not in e
        assert "blocked_reason" not in e


class TestConcurrentWrites:
    @pytest.mark.asyncio
    async def test_no_interleaved_lines(self, audit_path: Path) -> None:
        log = AuditLog(audit_path)
        # Fire many concurrent record calls — the async lock should
        # serialize them so every line on disk parses cleanly.
        async def one(i: int) -> None:
            await log.record_tool_call(
                tool=f"tool_{i}",
                outcome="ok",
                input={"i": i},
                result={"echoed": i},
            )

        await asyncio.gather(*(one(i) for i in range(50)))
        lines = read_lines(audit_path)
        assert len(lines) == 50
        # Each line should round-trip cleanly via JSON.
        tools = {e["tool"] for e in lines}
        assert tools == {f"tool_{i}" for i in range(50)}


class TestStartupAndShutdown:
    @pytest.mark.asyncio
    async def test_startup_entry(self, audit_path: Path) -> None:
        log = AuditLog(audit_path)
        await log.record_startup({"model": "ollama:qwen3:8b", "mcp_count": 5})
        e = read_lines(audit_path)[0]
        assert e["kind"] == "startup"
        assert e["outcome"] == "ok"
        assert e["result"] == {"model": "ollama:qwen3:8b", "mcp_count": 5}

    @pytest.mark.asyncio
    async def test_shutdown_entry(self, audit_path: Path) -> None:
        log = AuditLog(audit_path)
        await log.record_shutdown("SIGTERM")
        e = read_lines(audit_path)[0]
        assert e["kind"] == "shutdown"
        assert e["result"] == {"signal": "SIGTERM"}
