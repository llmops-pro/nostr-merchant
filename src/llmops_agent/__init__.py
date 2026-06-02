"""llmops-agent — reference Python agent for the LLMOps.Pro sovereign-AI substrate.

Consumes the five MCP servers (nwc-mcp, nostr-ops-mcp, marketplace-mcp,
albyhub-admin-mcp, paywall-mcp) and demonstrates the agent-pays-paid-MCP loop
end-to-end. Ollama-first, API-pluggable. Built on pydantic-ai.

See `03-python-reference-agent-design.md` in the parent project for the design.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    # Single source of truth: the version baked into the installed distribution
    # (from pyproject.toml). Avoids the hand-synced-constant drift where the
    # CLI reports a stale version after a release bump.
    __version__ = _pkg_version("llmops-agent")
except PackageNotFoundError:  # running from a raw checkout, not installed
    __version__ = "0.0.0+unknown"
