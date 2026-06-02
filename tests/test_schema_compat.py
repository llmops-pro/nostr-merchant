"""Regression guard: every substrate tool's inputSchema must be compatible
with the hosted-LLM providers we support (OpenAI/Ollama + Anthropic).

Two real bugs motivated this test:

  * `marketplace-mcp` shipped a `specs` field whose array `items` was a JSON
    Schema *tuple* (a list of schemas). Pydantic AI's
    `OpenAIJsonSchemaTransformer` crashes on that with
    `'list' object has no attribute 'get'` — so any OpenAI/Ollama-backed run
    blew up the moment that tool was in the toolset.
  * `nostr-ops-mcp` shipped `nostr_query_events` with property keys
    `#e/#p/#d/#t`. Anthropic's API rejects tool property names that don't
    match `^[a-zA-Z0-9_-]{1,64}$`, failing every `ask` with a 400.

Neither surfaced under Claude Code's MCP path; both only bit once the agent
talked to a raw provider API. This test reproduces both checks locally.

It is an *integration* test: it needs the TS servers built
(`<root>/<server>/dist/index.js`) and their own `.env` files present. When a
server's build is missing the whole test is skipped; when a built server
fails to spawn (e.g. missing config) that one server is skipped — but a
schema that a built, running server actually exposes is a hard failure.
"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.providers.anthropic import AnthropicJsonSchemaTransformer

from llmops_agent.config import (
    _SUBSTRATE_DIR_BY_NAME,
    McpServerSpec,
    _substrate_root_specs,
)

# Anthropic's documented constraint on tool input_schema property names.
_ANTHROPIC_PROP_KEY = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _substrate_root() -> Path:
    # tests/ -> python-agent/ -> NOSTR/  (the parent holding the server dirs)
    return Path(__file__).resolve().parents[2]


def _collect_property_keys(schema: Any, path: str = "") -> list[tuple[str, str]]:
    """Recursively yield (json-path, property-name) for every `properties` key.

    Walks into nested objects, array items (single or tuple-form), schema
    combinators, and `$defs`, so a bad key buried anywhere is still caught.
    """
    found: list[tuple[str, str]] = []
    if not isinstance(schema, dict):
        return found

    props = schema.get("properties")
    if isinstance(props, dict):
        for key, sub in props.items():
            found.append((f"{path}/properties/{key}", str(key)))
            found.extend(_collect_property_keys(sub, f"{path}/properties/{key}"))

    for kw in ("items", "additionalProperties", "not"):
        sub = schema.get(kw)
        if isinstance(sub, dict):
            found.extend(_collect_property_keys(sub, f"{path}/{kw}"))

    for kw in ("items", "prefixItems", "allOf", "anyOf", "oneOf"):
        val = schema.get(kw)
        if isinstance(val, list):
            for i, sub in enumerate(val):
                found.extend(_collect_property_keys(sub, f"{path}/{kw}/{i}"))

    for kw in ("$defs", "definitions"):
        defs = schema.get(kw)
        if isinstance(defs, dict):
            for name, sub in defs.items():
                found.extend(_collect_property_keys(sub, f"{path}/{kw}/{name}"))

    return found


def _built_specs() -> list[McpServerSpec]:
    """Substrate specs whose local `dist/index.js` exists (built)."""
    root = _substrate_root()
    return [
        spec
        for spec in _substrate_root_specs(root)
        if (root / _SUBSTRATE_DIR_BY_NAME[spec.name] / "dist" / "index.js").exists()
    ]


_SPECS = _built_specs()


@pytest.mark.skipif(
    not _SPECS,
    reason="no substrate builds found (run `pnpm build` in each *-mcp dir)",
)
@pytest.mark.parametrize("spec", _SPECS, ids=lambda s: s.name)
async def test_tool_schemas_are_provider_compatible(spec: McpServerSpec) -> None:
    kwargs: dict[str, Any] = {
        "command": spec.command,
        "args": list(spec.args),
        "id": spec.name,
        "timeout": 10.0,
    }
    if spec.cwd is not None:
        kwargs["cwd"] = spec.cwd
    if spec.env is not None:
        kwargs["env"] = spec.env

    server = MCPServerStdio(**kwargs)
    try:
        async with server:
            tools = await server.list_tools()
    except Exception as err:  # spawn/config failure is not a schema problem
        pytest.skip(f"{spec.name} did not start (likely missing .env): {err}")

    assert tools, f"{spec.name} exposed zero tools"

    for tool in tools:
        schema = tool.inputSchema

        # Bug #1: the OpenAI/Ollama transformer must walk the schema cleanly.
        try:
            OpenAIJsonSchemaTransformer(deepcopy(schema)).walk()
        except Exception as err:
            pytest.fail(
                f"{spec.name}:{tool.name} inputSchema breaks "
                f"OpenAIJsonSchemaTransformer (bug-#1 class — e.g. a tuple-typed "
                f"array `items`): {err}"
            )

        # Defensive: the Anthropic transformer must also walk it cleanly.
        try:
            AnthropicJsonSchemaTransformer(deepcopy(schema)).walk()
        except Exception as err:
            pytest.fail(
                f"{spec.name}:{tool.name} inputSchema breaks "
                f"AnthropicJsonSchemaTransformer: {err}"
            )

        # Bug #2: every property name must satisfy Anthropic's key pattern.
        for json_path, key in _collect_property_keys(schema):
            assert _ANTHROPIC_PROP_KEY.match(key), (
                f"{spec.name}:{tool.name} has illegal property name {key!r} at "
                f"{json_path} — Anthropic requires ^[a-zA-Z0-9_-]{{1,64}}$ "
                f"(bug-#2 class — e.g. NIP-01 `#e`/`#p` tag-filter keys)."
            )
