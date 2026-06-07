"""CLI smoke tests via typer.testing.CliRunner.

These don't spawn real MCP servers or hit a real LLM — they exercise the
CLI plumbing (typer wiring, config loading, table rendering) and confirm
that commands at least produce the right shape of output.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nostr_merchant.cli import app


@pytest.fixture
def isolated_env(tmp_path: Path) -> Iterator[None]:
    """Point AGENT_AUDIT_PATH / AGENT_BUDGET_PATH at tmp_path for the test."""
    keys = (
        "AGENT_AUDIT_PATH",
        "AGENT_BUDGET_PATH",
        "AGENT_MAX_SATS_PER_TASK",
        "AGENT_MAX_SATS_PER_DAY",
        "AGENT_MAX_TOOL_PRICE",
        "NOSTR_MERCHANT_MODEL",
        "AGENT_TOOL_ALLOWLIST",
        "AGENT_READ_ONLY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    )
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    os.environ["AGENT_AUDIT_PATH"] = str(tmp_path / "audit.log")
    os.environ["AGENT_BUDGET_PATH"] = str(tmp_path / "budget.json")
    os.environ["NOSTR_MERCHANT_MODEL"] = "ollama:qwen3:8b"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestVersion:
    def test_prints_version(self, isolated_env: None) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        from nostr_merchant import __version__

        assert __version__ in result.stdout


class TestBudgetCommand:
    def test_prints_table_with_zero_spend(self, isolated_env: None) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["budget"])
        assert result.exit_code == 0
        # Rich tables include the column labels in plain text.
        assert "this task" in result.stdout
        assert "today" in result.stdout
        assert "lifetime" in result.stdout


class TestAuditCommand:
    def test_says_empty_when_no_log(self, isolated_env: None) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["audit"])
        assert result.exit_code == 0
        assert "No audit log" in result.stdout

    def test_renders_recent_entries(self, isolated_env: None) -> None:
        runner = CliRunner()
        audit_path = Path(os.environ["AGENT_AUDIT_PATH"])
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            '{"ts":"2026-05-31T20:14:33.000Z","kind":"tool_call",'
            '"outcome":"ok","tool":"nwc__get_balance","result":{"balance":42}}\n',
            encoding="utf-8",
        )
        result = runner.invoke(app, ["audit", "--tail", "5"])
        assert result.exit_code == 0
        # Renders the kind in the output table.
        assert "tool_call" in result.stdout


class TestConfigPrint:
    def test_masks_secrets(self, isolated_env: None) -> None:
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-real-secret"
        runner = CliRunner()
        result = runner.invoke(app, ["config-print"])
        assert result.exit_code == 0
        assert "sk-ant-real-secret" not in result.stdout
        # Rich's Syntax may break the key across lines or include markup; just
        # check the substring is present (whether `"***"` or `"***\n"` etc.)
        assert "***" in result.stdout

    def test_dumps_known_fields(self, isolated_env: None) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["config-print"])
        assert result.exit_code == 0
        assert "NOSTR_MERCHANT_MODEL" in result.stdout
        assert "AGENT_MAX_SATS_PER_TASK" in result.stdout


class TestInvalidConfigFailsLoudly:
    def test_bad_model_string_exits_nonzero(self, isolated_env: None) -> None:
        os.environ["NOSTR_MERCHANT_MODEL"] = "garbage_no_colon"
        runner = CliRunner()
        result = runner.invoke(app, ["budget"])
        assert result.exit_code == 1
        # The rich panel mentions "Config load failed".
        assert "Config load failed" in result.stdout
