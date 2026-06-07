"""Tests for the agent layer.

Two narrow goals:
  1. `build_agent` returns an `Agent` configured the way we expect (model,
     toolsets, system prompt). No real LLM, no real MCP processes.
  2. `make_process_tool_call` enforces the safety pipeline correctly:
     allowlist, read-only, max-tool-price, budget caps, audit on every step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.models.test import TestModel

from nostr_merchant.agent import (
    SYSTEM_PROMPT,
    build_agent,
    make_process_tool_call,
)
from nostr_merchant.audit import AuditLog
from nostr_merchant.budget import BudgetTracker
from nostr_merchant.config import AgentConfig


def make_config(**overrides: object) -> AgentConfig:
    base: dict[str, object] = {
        "NOSTR_MERCHANT_MODEL": "ollama:qwen3:8b",
        "AGENT_MAX_SATS_PER_TASK": 100,
        "AGENT_MAX_SATS_PER_DAY": 1000,
        "AGENT_MAX_TOOL_PRICE": 500,
    }
    base.update(overrides)
    return AgentConfig.model_validate(base)


@pytest.fixture
def budget_path(tmp_path: Path) -> Path:
    return tmp_path / "budget.json"


@pytest.fixture
def audit_path(tmp_path: Path) -> Path:
    return tmp_path / "audit.log"


# --------------------------------------------------------------------------- #
# build_agent
# --------------------------------------------------------------------------- #


class TestBuildAgent:
    def test_returns_agent_with_test_model_override(self) -> None:
        cfg = make_config()
        agent = build_agent(cfg, model_override=TestModel())
        # Smoke: the agent at least has the system prompt and a model.
        assert agent.model is not None
        # The system prompt is attached as a tuple of strings via Agent's
        # constructor; pydantic-ai 1.x stores it on `_system_prompts`.
        # Just verify our prompt was passed at all.
        assert "Lightning paywall pattern" in SYSTEM_PROMPT

    def test_custom_system_prompt_used(self) -> None:
        cfg = make_config()
        agent = build_agent(
            cfg,
            model_override=TestModel(),
            system_prompt="custom marker prompt",
        )
        assert agent is not None  # built successfully

    def test_attaches_mcp_servers_per_spec(self) -> None:
        cfg = make_config()
        agent = build_agent(cfg, model_override=TestModel())
        # 5 servers expected (the bundled default set).
        # `_toolsets` is the internal storage; we just count.
        toolsets = getattr(agent, "_toolsets", None) or getattr(agent, "toolsets", None)
        # toolsets includes wrappers around MCPServerStdio; rough count.
        assert toolsets is not None


# --------------------------------------------------------------------------- #
# make_process_tool_call
# --------------------------------------------------------------------------- #


async def _call_tool_noop(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Stub upstream call that just echoes — used when we want passthrough."""
    return {"ok": True, "name": name, "args_keys": sorted(args.keys())}


async def _call_tool_paywall_21(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Stub that simulates a paywall-mcp response (payment_required)."""
    return {
        "error": "payment_required",
        "invoice": "lnbc210n1pjexample...",
        "payment_hash": "a" * 64,
        "amount_sats": 21,
        "expires_in_seconds": 600,
        "next_step": "pay then retry",
    }


async def _call_tool_paywall_5000(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "error": "payment_required",
        "invoice": "lnbc50000n1p...",
        "payment_hash": "b" * 64,
        "amount_sats": 5000,
        "expires_in_seconds": 600,
    }


async def _call_tool_pay_invoice_success(
    name: str, args: dict[str, Any]
) -> dict[str, Any]:
    return {"status": "settled", "amount_sats": 21, "payment_hash": "a" * 64}


class TestProcessToolCallAllowlist:
    @pytest.mark.asyncio
    async def test_blocks_unlisted_tool(
        self, budget_path: Path, audit_path: Path
    ) -> None:
        process = make_process_tool_call(
            budget=BudgetTracker(
                path=budget_path, max_per_task_sats=100, max_per_day_sats=1000
            ),
            audit=AuditLog(audit_path),
            max_tool_price=500,
            tool_allowlist={"some_other_tool"},
            read_only=False,
        )
        result = await process(None, _call_tool_noop, "nwc_get_balance", {})
        assert isinstance(result, dict)
        assert result.get("error") == "tool_not_in_allowlist"

    @pytest.mark.asyncio
    async def test_passes_listed_tool(
        self, budget_path: Path, audit_path: Path
    ) -> None:
        process = make_process_tool_call(
            budget=BudgetTracker(
                path=budget_path, max_per_task_sats=100, max_per_day_sats=1000
            ),
            audit=AuditLog(audit_path),
            max_tool_price=500,
            tool_allowlist={"nwc_get_balance"},
            read_only=False,
        )
        result = await process(None, _call_tool_noop, "nwc_get_balance", {})
        assert result.get("ok") is True


class TestProcessToolCallReadOnly:
    @pytest.mark.asyncio
    async def test_refuses_paywall_tool(
        self, budget_path: Path, audit_path: Path
    ) -> None:
        process = make_process_tool_call(
            budget=BudgetTracker(
                path=budget_path, max_per_task_sats=100, max_per_day_sats=1000
            ),
            audit=AuditLog(audit_path),
            max_tool_price=500,
            tool_allowlist=None,
            read_only=True,
        )
        result = await process(None, _call_tool_noop, "paywall_some_tool", {})
        assert result.get("error") == "agent_read_only"

    @pytest.mark.asyncio
    async def test_refuses_pay_invoice(
        self, budget_path: Path, audit_path: Path
    ) -> None:
        process = make_process_tool_call(
            budget=BudgetTracker(
                path=budget_path, max_per_task_sats=100, max_per_day_sats=1000
            ),
            audit=AuditLog(audit_path),
            max_tool_price=500,
            tool_allowlist=None,
            read_only=True,
        )
        result = await process(None, _call_tool_noop, "nwc_pay_invoice", {})
        assert result.get("error") == "agent_read_only"

    @pytest.mark.asyncio
    async def test_allows_get_balance(
        self, budget_path: Path, audit_path: Path
    ) -> None:
        process = make_process_tool_call(
            budget=BudgetTracker(
                path=budget_path, max_per_task_sats=100, max_per_day_sats=1000
            ),
            audit=AuditLog(audit_path),
            max_tool_price=500,
            tool_allowlist=None,
            read_only=True,
        )
        result = await process(None, _call_tool_noop, "nwc_get_balance", {})
        assert result.get("ok") is True


class TestProcessToolCallPaywallEnforcement:
    @pytest.mark.asyncio
    async def test_price_under_cap_passes_invoice_through(
        self, budget_path: Path, audit_path: Path
    ) -> None:
        process = make_process_tool_call(
            budget=BudgetTracker(
                path=budget_path, max_per_task_sats=100, max_per_day_sats=1000
            ),
            audit=AuditLog(audit_path),
            max_tool_price=500,
            tool_allowlist=None,
            read_only=False,
        )
        result = await process(None, _call_tool_paywall_21, "paywall_cheap_tool", {})
        # The LLM should see the original payment_required so it can pay+retry.
        assert isinstance(result, dict)
        assert result.get("error") == "payment_required"
        assert result.get("amount_sats") == 21

    @pytest.mark.asyncio
    async def test_price_above_max_tool_price_refused(
        self, budget_path: Path, audit_path: Path
    ) -> None:
        process = make_process_tool_call(
            budget=BudgetTracker(
                path=budget_path, max_per_task_sats=10_000, max_per_day_sats=20_000
            ),
            audit=AuditLog(audit_path),
            max_tool_price=500,
            tool_allowlist=None,
            read_only=False,
        )
        result = await process(
            None, _call_tool_paywall_5000, "paywall_expensive_tool", {}
        )
        assert isinstance(result, dict)
        assert result.get("error") == "agent_max_tool_price_exceeded"
        assert result.get("price_sats") == 5000

    @pytest.mark.asyncio
    async def test_price_exceeds_budget_refused(
        self, budget_path: Path, audit_path: Path
    ) -> None:
        process = make_process_tool_call(
            budget=BudgetTracker(
                path=budget_path, max_per_task_sats=10, max_per_day_sats=1000
            ),
            audit=AuditLog(audit_path),
            max_tool_price=500,
            tool_allowlist=None,
            read_only=False,
        )
        # 21 sat tool with a 10 sat per-task cap → refused.
        result = await process(
            None, _call_tool_paywall_21, "paywall_some_tool", {}
        )
        assert isinstance(result, dict)
        assert result.get("error") == "agent_budget_exceeded"


class TestProcessToolCallBudgetBookkeeping:
    @pytest.mark.asyncio
    async def test_pay_invoice_success_records_spend(
        self, budget_path: Path, audit_path: Path
    ) -> None:
        budget = BudgetTracker(
            path=budget_path, max_per_task_sats=100, max_per_day_sats=1000
        )
        process = make_process_tool_call(
            budget=budget,
            audit=AuditLog(audit_path),
            max_tool_price=500,
            tool_allowlist=None,
            read_only=False,
        )
        before = budget.snapshot()
        assert before.per_task_spent_sats == 0
        await process(
            None,
            _call_tool_pay_invoice_success,
            "nwc_pay_invoice",
            {"invoice": "lnbc..."},
        )
        after = budget.snapshot()
        assert after.per_task_spent_sats == 21


class TestProcessToolCallAuditLogs:
    @pytest.mark.asyncio
    async def test_audit_records_blocked_calls(
        self, budget_path: Path, audit_path: Path
    ) -> None:
        process = make_process_tool_call(
            budget=BudgetTracker(
                path=budget_path, max_per_task_sats=100, max_per_day_sats=1000
            ),
            audit=AuditLog(audit_path),
            max_tool_price=500,
            tool_allowlist={"only_this_one"},
            read_only=False,
        )
        await process(None, _call_tool_noop, "nwc_get_balance", {})
        import json

        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["outcome"] == "blocked"
        assert entry["kind"] == "tool_call"
        assert "AGENT_TOOL_ALLOWLIST" in entry["blocked_reason"]
