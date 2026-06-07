"""Tests for `nostr_merchant.config`.

The config object is a Pydantic Settings instance — most of the
interesting behavior is in the validators and the derived helpers.
"""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from nostr_merchant.config import AgentConfig


def make(**overrides: object) -> AgentConfig:
    """Construct an AgentConfig bypassing env file loading, with overrides.

    `_env_file=None` is load-bearing: without it, pydantic-settings still
    reads the operator's real `~/.nostr-merchant/.env`, leaking whatever the
    user has configured there (e.g. NOSTR_MERCHANT_SUBSTRATE_SKIP) into every test
    config and making the suite non-hermetic.
    """
    base: dict[str, object] = {
        "NOSTR_MERCHANT_MODEL": "ollama:qwen3:8b",
        "AGENT_MAX_SATS_PER_TASK": 100,
        "AGENT_MAX_SATS_PER_DAY": 1000,
        "AGENT_MAX_TOOL_PRICE": 500,
    }
    base.update(overrides)
    return AgentConfig(_env_file=None, **base)  # type: ignore[call-arg, arg-type]


class TestModelString:
    def test_accepts_ollama_default(self) -> None:
        cfg = make(NOSTR_MERCHANT_MODEL="ollama:qwen3:8b")
        assert cfg.llm_provider() == "ollama"
        assert cfg.llm_model_name() == "qwen3:8b"

    def test_accepts_anthropic(self) -> None:
        cfg = make(NOSTR_MERCHANT_MODEL="anthropic:claude-haiku-4-5-20251001")
        assert cfg.llm_provider() == "anthropic"
        assert cfg.llm_model_name() == "claude-haiku-4-5-20251001"

    def test_rejects_missing_colon(self) -> None:
        with pytest.raises(ValidationError, match="<provider>:<model>"):
            make(NOSTR_MERCHANT_MODEL="qwen3")

    def test_rejects_unknown_provider(self) -> None:
        with pytest.raises(ValidationError, match="Unrecognized LLM provider"):
            make(NOSTR_MERCHANT_MODEL="madeup:model")


class TestBudgetInvariants:
    def test_per_task_cannot_exceed_per_day(self) -> None:
        with pytest.raises(ValidationError, match="AGENT_MAX_SATS_PER_TASK"):
            make(AGENT_MAX_SATS_PER_TASK=500, AGENT_MAX_SATS_PER_DAY=100)

    def test_equal_per_task_and_per_day_is_allowed(self) -> None:
        cfg = make(AGENT_MAX_SATS_PER_TASK=100, AGENT_MAX_SATS_PER_DAY=100)
        assert cfg.AGENT_MAX_SATS_PER_TASK == 100
        assert cfg.AGENT_MAX_SATS_PER_DAY == 100

    def test_negative_cap_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make(AGENT_MAX_SATS_PER_TASK=-1)


class TestLogLevel:
    def test_normalizes_to_upper(self) -> None:
        cfg = make(AGENT_LOG_LEVEL="debug")
        assert cfg.AGENT_LOG_LEVEL == "DEBUG"

    def test_rejects_garbage(self) -> None:
        with pytest.raises(ValidationError, match="AGENT_LOG_LEVEL"):
            make(AGENT_LOG_LEVEL="LOUD")


class TestToolAllowlist:
    def test_none_when_unset(self) -> None:
        cfg = make()
        assert cfg.tool_allowlist() is None

    def test_parses_csv(self) -> None:
        cfg = make(AGENT_TOOL_ALLOWLIST="a, b ,c")
        assert cfg.tool_allowlist() == {"a", "b", "c"}

    def test_returns_none_for_blank(self) -> None:
        cfg = make(AGENT_TOOL_ALLOWLIST="  ,  ")
        assert cfg.tool_allowlist() is None


class TestMcpServerSpecs:
    def test_default_set_is_the_five_substrate_servers(self) -> None:
        cfg = make()
        names = [spec.name for spec in cfg.mcp_server_specs()]
        assert names == ["nwc", "nostr", "marketplace", "albyhub", "paywall"]
        for spec in cfg.mcp_server_specs():
            assert spec.command == "npx"
            assert spec.args[0] == "-y"

    def test_override_via_json_array(self) -> None:
        cfg = make(
            NOSTR_MERCHANT_MCP_SERVERS='[{"name":"only","command":"node","args":["a.js"]}]'
        )
        specs = cfg.mcp_server_specs()
        assert len(specs) == 1
        assert specs[0].name == "only"
        assert specs[0].command == "node"
        assert specs[0].args == ["a.js"]

    def test_invalid_json_raises(self) -> None:
        cfg = make(NOSTR_MERCHANT_MCP_SERVERS="{not json}")
        with pytest.raises(ValueError, match="not valid JSON"):
            cfg.mcp_server_specs()

    def test_non_array_raises(self) -> None:
        cfg = make(NOSTR_MERCHANT_MCP_SERVERS='{"name":"x","command":"y"}')
        with pytest.raises(ValueError, match="JSON array"):
            cfg.mcp_server_specs()


class TestSubstrateRoot:
    def test_local_builds_emit_node_command(self, tmp_path: object) -> None:
        cfg = make(NOSTR_MERCHANT_SUBSTRATE_ROOT=str(tmp_path))
        specs = cfg.mcp_server_specs()
        assert len(specs) == 5
        for spec in specs:
            assert spec.command == "node"
            assert spec.args[0].endswith("/dist/index.js")
        # `nostr` spec name maps to the `nostr-ops-mcp` directory.
        nostr_spec = next(s for s in specs if s.name == "nostr")
        assert "nostr-ops-mcp" in nostr_spec.args[0]
        assert nostr_spec.cwd is not None
        assert nostr_spec.cwd.endswith("nostr-ops-mcp")


class TestSubstrateSkip:
    def test_no_skip_when_unset(self) -> None:
        cfg = make()
        assert cfg.substrate_skip() == set()
        assert len(cfg.mcp_server_specs()) == 5

    def test_single_skip_filters_npx_defaults(self) -> None:
        cfg = make(NOSTR_MERCHANT_SUBSTRATE_SKIP="albyhub")
        assert cfg.substrate_skip() == {"albyhub"}
        names = [s.name for s in cfg.mcp_server_specs()]
        assert "albyhub" not in names
        assert len(names) == 4

    def test_multiple_skips_csv(self) -> None:
        cfg = make(NOSTR_MERCHANT_SUBSTRATE_SKIP="albyhub, paywall")
        assert cfg.substrate_skip() == {"albyhub", "paywall"}
        names = [s.name for s in cfg.mcp_server_specs()]
        assert "albyhub" not in names
        assert "paywall" not in names
        assert len(names) == 3

    def test_skip_filters_substrate_root_specs(self, tmp_path: object) -> None:
        cfg = make(
            NOSTR_MERCHANT_SUBSTRATE_ROOT=str(tmp_path),
            NOSTR_MERCHANT_SUBSTRATE_SKIP="nostr",
        )
        names = [s.name for s in cfg.mcp_server_specs()]
        assert "nostr" not in names
        assert len(names) == 4
        # The remaining 4 should still be `node`-spawned (substrate-root mode).
        for spec in cfg.mcp_server_specs():
            assert spec.command == "node"

    def test_skip_does_not_apply_to_full_override(self) -> None:
        cfg = make(
            NOSTR_MERCHANT_MCP_SERVERS='[{"name":"albyhub","command":"x","args":[]}]',
            NOSTR_MERCHANT_SUBSTRATE_SKIP="albyhub",  # ignored when full override is set
        )
        specs = cfg.mcp_server_specs()
        # Full override wins — albyhub still present.
        assert [s.name for s in specs] == ["albyhub"]

    def test_unknown_name_raises(self) -> None:
        cfg = make(NOSTR_MERCHANT_SUBSTRATE_SKIP="bogus")
        with pytest.raises(ValueError, match="unrecognized names"):
            cfg.substrate_skip()

    def test_case_and_whitespace_tolerated(self) -> None:
        cfg = make(NOSTR_MERCHANT_SUBSTRATE_SKIP=" Albyhub ,  PAYWALL ")
        assert cfg.substrate_skip() == {"albyhub", "paywall"}


class TestApplyProviderEnv:
    def test_bridges_config_values_into_environ(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        cfg = make(
            ANTHROPIC_API_KEY="sk-ant-test",
            OPENAI_API_KEY="sk-openai-test",
            OLLAMA_BASE_URL="http://localhost:11434/v1",
        )
        cfg.apply_provider_env()
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test"
        assert os.environ["OPENAI_API_KEY"] == "sk-openai-test"
        assert os.environ["OLLAMA_BASE_URL"] == "http://localhost:11434/v1"

    def test_does_not_overwrite_existing_environ(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An explicit shell export must win over the .env-sourced value.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
        cfg = make(ANTHROPIC_API_KEY="sk-ant-from-dotenv")
        cfg.apply_provider_env()
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-shell"

    def test_skips_unset_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = make()  # no provider keys set
        cfg.apply_provider_env()
        assert "OPENAI_API_KEY" not in os.environ
