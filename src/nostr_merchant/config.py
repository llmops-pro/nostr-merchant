"""Env-driven configuration for nostr-merchant.

Mirrors the safety-knob conventions of the substrate MCP servers (read-only
mode, per-call budget cap, daily budget cap, etc.) but applies them at the
agent layer — on top of nwc-mcp's own caps, not in place of them.

Env vars are documented in `.env.example` and in the design doc §6.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class McpServerSpec(BaseSettings):
    """One MCP server launch spec — name, command, args, optional cwd/env."""

    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] | None = None


def _default_audit_path() -> Path:
    return Path.home() / ".nostr-merchant" / "audit.log"


def _default_budget_path() -> Path:
    return Path.home() / ".nostr-merchant" / "budget.json"


def _default_replied_path() -> Path:
    return Path.home() / ".nostr-merchant" / "replied.json"


class AgentConfig(BaseSettings):
    """All env-driven config for nostr-merchant.

    The fields here are read from the process environment (or the `.env` file
    at `~/.nostr-merchant/.env`). They are validated by Pydantic, so any
    misconfiguration fails loudly at startup rather than mid-task.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=str(Path.home() / ".nostr-merchant" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ---- LLM backend ----
    NOSTR_MERCHANT_MODEL: str = Field(
        default="ollama:qwen3:8b",
        description=(
            "Pydantic AI model string. Examples: 'ollama:qwen3:8b', "
            "'anthropic:claude-haiku-4-5-20251001', 'openai:gpt-5-mini'."
        ),
    )
    OLLAMA_BASE_URL: str | None = Field(
        default=None,
        description=(
            "Base URL for the Ollama HTTP API. Defaults to http://localhost:11434/v1 "
            "if unset (Pydantic AI's default). Set if Ollama runs on a non-standard "
            "host or port."
        ),
    )
    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None

    # ---- MCP servers ----
    NOSTR_MERCHANT_MCP_SERVERS: str | None = Field(
        default=None,
        description=(
            "JSON array of MCP server specs: "
            '[{"name":"nwc","command":"npx","args":["-y","nwc-mcp"]}, ...]. '
            "When unset, the bundled defaults are chosen based on "
            "NOSTR_MERCHANT_SUBSTRATE_ROOT (see below)."
        ),
    )
    NOSTR_MERCHANT_SUBSTRATE_ROOT: Path | None = Field(
        default=None,
        description=(
            "Path to the parent directory containing the five MCP-server "
            "build directories as siblings (`nwc-mcp/`, `nostr-ops-mcp/`, "
            "`marketplace-mcp/`, `albyhub-admin-mcp/`, `paywall-mcp/`). "
            "When set AND `NOSTR_MERCHANT_MCP_SERVERS` is unset, the agent spawns "
            "the substrate from your LOCAL builds via "
            "`node <root>/<name>/dist/index.js`, so each server reads its "
            "own `.env` from its own build directory. Use this in dev. "
            "When BOTH unset, the agent falls back to `npx -y <name>` and "
            "relies on each server picking its config up from the process "
            "env (the npm-installed binaries don't have their own `.env` "
            "files in the npx cache)."
        ),
    )
    NOSTR_MERCHANT_SUBSTRATE_SKIP: str | None = Field(
        default=None,
        description=(
            "Optional CSV of substrate spec names to exclude from the "
            "default list (whether sourced from NOSTR_MERCHANT_SUBSTRATE_ROOT or "
            "the npx fallback). Useful when one server's config is broken "
            "or its upstream dependency is unavailable. Recognized names: "
            "nwc, nostr, marketplace, albyhub, paywall. Example: "
            "'NOSTR_MERCHANT_SUBSTRATE_SKIP=albyhub' to launch only the other four."
        ),
    )

    # ---- Agent-layer safety knobs ----
    AGENT_READ_ONLY: bool = Field(
        default=False,
        description="When true, the agent refuses to call any priced MCP tool.",
    )
    AGENT_MAX_SATS_PER_TASK: int = Field(
        default=100,
        ge=0,
        description="Single `ask` invocation can spend at most this many sats.",
    )
    AGENT_MAX_SATS_PER_DAY: int = Field(
        default=1_000,
        ge=0,
        description="Rolling 24h budget across all tasks.",
    )
    AGENT_MAX_TOOL_PRICE: int = Field(
        default=500,
        ge=0,
        description=(
            "Refuse any individual paid tool whose invoice asks more than this, "
            "regardless of remaining budget."
        ),
    )
    AGENT_TOOL_ALLOWLIST: str | None = Field(
        default=None,
        description=(
            "Optional comma-separated tool name allowlist. When set, only "
            "listed tools are callable; all others (free or paid) are refused."
        ),
    )
    AGENT_CONFIRM_THRESHOLD_SATS: int = Field(
        default=100,
        ge=0,
        description=(
            "In interactive mode, pause and ask the operator before paying "
            "any tool whose price exceeds this threshold."
        ),
    )

    # ---- Logging / persistence paths ----
    AGENT_AUDIT_PATH: Path = Field(default_factory=_default_audit_path)
    AGENT_BUDGET_PATH: Path = Field(default_factory=_default_budget_path)
    AGENT_REPLIED_PATH: Path = Field(default_factory=_default_replied_path)
    AGENT_LOG_LEVEL: str = Field(default="INFO")

    # ---- Validators ----

    @field_validator("NOSTR_MERCHANT_MODEL")
    @classmethod
    def _validate_model_string(cls, v: str) -> str:
        if ":" not in v:
            msg = (
                f"NOSTR_MERCHANT_MODEL must be of the form '<provider>:<model>' "
                f"(e.g. 'ollama:qwen3:8b'). Got: {v!r}"
            )
            raise ValueError(msg)
        provider = v.split(":", 1)[0]
        known = {"ollama", "anthropic", "openai", "google-gla", "groq", "mistral"}
        if provider not in known:
            msg = (
                f"Unrecognized LLM provider {provider!r}. Known providers: "
                f"{sorted(known)}. (If pydantic-ai added a new provider since "
                f"this release, update nostr_merchant/config.py.)"
            )
            raise ValueError(msg)
        return v

    @field_validator("AGENT_LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        upper = v.upper()
        if upper not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            msg = f"AGENT_LOG_LEVEL must be one of DEBUG/INFO/WARNING/ERROR/CRITICAL, got {v!r}"
            raise ValueError(msg)
        return upper

    @model_validator(mode="after")
    def _check_budget_invariants(self) -> AgentConfig:
        if self.AGENT_MAX_SATS_PER_TASK > self.AGENT_MAX_SATS_PER_DAY:
            msg = (
                "AGENT_MAX_SATS_PER_TASK "
                f"({self.AGENT_MAX_SATS_PER_TASK}) cannot exceed "
                "AGENT_MAX_SATS_PER_DAY "
                f"({self.AGENT_MAX_SATS_PER_DAY}). Tighten one or the other."
            )
            raise ValueError(msg)
        return self

    # ---- Derived helpers ----

    def apply_provider_env(self) -> None:
        """Bridge provider credentials/endpoints into ``os.environ``.

        Pydantic AI's provider classes (Anthropic, OpenAI, Ollama) read their
        credentials from the *process environment*, not from this config
        object. Values loaded from the ``.env`` file populate the fields here
        but never reach ``os.environ`` on their own — so an `ask` would fail
        with a missing-key/missing-base-url error unless the operator had
        also `export`ed them by hand. This method closes that gap.

        An explicit value already present in the real environment is left
        untouched, so a shell `export` still wins over the ``.env`` file.
        """
        for key, value in (
            ("ANTHROPIC_API_KEY", self.ANTHROPIC_API_KEY),
            ("OPENAI_API_KEY", self.OPENAI_API_KEY),
            ("OLLAMA_BASE_URL", self.OLLAMA_BASE_URL),
        ):
            if value and not os.environ.get(key):
                os.environ[key] = value

    def llm_provider(self) -> str:
        """Provider half of NOSTR_MERCHANT_MODEL (e.g., 'ollama')."""
        return self.NOSTR_MERCHANT_MODEL.split(":", 1)[0]

    def llm_model_name(self) -> str:
        """Model half of NOSTR_MERCHANT_MODEL (e.g., 'qwen3:8b')."""
        return self.NOSTR_MERCHANT_MODEL.split(":", 1)[1]

    def tool_allowlist(self) -> set[str] | None:
        """Parse AGENT_TOOL_ALLOWLIST into a set, or None when unset."""
        raw = self.AGENT_TOOL_ALLOWLIST
        if not raw:
            return None
        names = {name.strip() for name in raw.split(",") if name.strip()}
        return names or None

    def substrate_skip(self) -> set[str]:
        """Parse NOSTR_MERCHANT_SUBSTRATE_SKIP into a normalized set of spec names."""
        raw = self.NOSTR_MERCHANT_SUBSTRATE_SKIP
        if not raw:
            return set()
        names = {n.strip().lower() for n in raw.split(",") if n.strip()}
        unknown = names - set(_SUBSTRATE_DIR_BY_NAME.keys())
        if unknown:
            msg = (
                "NOSTR_MERCHANT_SUBSTRATE_SKIP contains unrecognized names: "
                f"{sorted(unknown)}. Known: {sorted(_SUBSTRATE_DIR_BY_NAME.keys())}."
            )
            raise ValueError(msg)
        return names

    def mcp_server_specs(self) -> list[McpServerSpec]:
        """Resolve the MCP server launch specs.

        Resolution order:
          1. `NOSTR_MERCHANT_MCP_SERVERS` JSON env var (highest priority — full override;
             the skip list does NOT apply when this is used).
          2. `NOSTR_MERCHANT_SUBSTRATE_ROOT` env var (local-builds mode for dev), with
             the `NOSTR_MERCHANT_SUBSTRATE_SKIP` set filtered out.
          3. `npx -y <name>` fallback, with the skip set filtered out.
        """
        raw = self.NOSTR_MERCHANT_MCP_SERVERS
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as err:
                msg = f"NOSTR_MERCHANT_MCP_SERVERS is not valid JSON: {err}"
                raise ValueError(msg) from err
            if not isinstance(parsed, list):
                msg = "NOSTR_MERCHANT_MCP_SERVERS must be a JSON array of server specs"
                raise ValueError(msg)
            return [McpServerSpec.model_validate(item) for item in parsed]
        skip = self.substrate_skip()
        if self.NOSTR_MERCHANT_SUBSTRATE_ROOT is not None:
            return [
                spec
                for spec in _substrate_root_specs(self.NOSTR_MERCHANT_SUBSTRATE_ROOT)
                if spec.name not in skip
            ]
        npx_defaults = [
            McpServerSpec(name="nwc", command="npx", args=["-y", "nwc-mcp"]),
            McpServerSpec(name="nostr", command="npx", args=["-y", "nostr-ops-mcp"]),
            McpServerSpec(name="marketplace", command="npx", args=["-y", "marketplace-mcp"]),
            McpServerSpec(name="albyhub", command="npx", args=["-y", "albyhub-admin-mcp"]),
            McpServerSpec(name="paywall", command="npx", args=["-y", "paywall-mcp"]),
        ]
        return [spec for spec in npx_defaults if spec.name not in skip]


# Map MCP server spec names → directory names on disk.
# All match except `nostr`, whose folder is `nostr-ops-mcp` (the npm package
# name `nostr-mcp` was already taken, but the folder retained the npm-name
# convention `nostr-ops-mcp` after the rename pass).
_SUBSTRATE_DIR_BY_NAME: dict[str, str] = {
    "nwc": "nwc-mcp",
    "nostr": "nostr-ops-mcp",
    "marketplace": "marketplace-mcp",
    "albyhub": "albyhub-admin-mcp",
    "paywall": "paywall-mcp",
}


def _substrate_root_specs(root: Path) -> list[McpServerSpec]:
    """Build the five-server spec list pointing at local builds under `root`."""
    root = root.expanduser().resolve()
    specs: list[McpServerSpec] = []
    for spec_name, dir_name in _SUBSTRATE_DIR_BY_NAME.items():
        build = root / dir_name / "dist" / "index.js"
        specs.append(
            McpServerSpec(
                name=spec_name,
                command="node",
                args=[str(build)],
                cwd=str(root / dir_name),
            ),
        )
    return specs
