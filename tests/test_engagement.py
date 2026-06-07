"""Tests for the engagement inbox workflow (v1: deterministic gather + tool-less draft)."""

from __future__ import annotations

from typer.testing import CliRunner

from nostr_merchant.cli import app
from nostr_merchant.workflows.engagement import GATHER_READ_TOOLS, _e_tags, _events, _text


class TestGatherIsReadOnly:
    """v1's safety story: the workflow only ever calls read tools, and drafting uses no tools."""

    def test_gather_only_uses_read_tools(self) -> None:
        # The two tools the gather calls must be read-only — never publish/DM/spend/mutate.
        forbidden = ("publish", "send", "delete", "pay", "confirm", "create", "update", "decrypt")
        for tool in GATHER_READ_TOOLS:
            assert not any(marker in tool for marker in forbidden), (
                f"{tool!r} looks like a write/spend tool — the inbox must stay read-only"
            )
        assert "nostr_query_events" in GATHER_READ_TOOLS
        assert "nostr_get_pubkey" in GATHER_READ_TOOLS


class TestResultParsing:
    def test_events_extracts_list(self) -> None:
        payload = '{"count":2,"events":[{"id":"a"},{"id":"b"}]}'
        evs = _events(payload)
        assert [e["id"] for e in evs] == ["a", "b"]

    def test_events_tolerates_garbage(self) -> None:
        assert _events("not json") == []
        assert _events('{"no":"events"}') == []

    def test_e_tags_pulls_e_tag_targets(self) -> None:
        ev = {"tags": [["e", "deadbeef"], ["p", "abc"], ["e", "cafe"], ["x"]]}
        assert _e_tags(ev) == ["deadbeef", "cafe"]

    def test_text_reads_mcp_content_block(self) -> None:
        import types

        res = types.SimpleNamespace(content=[types.SimpleNamespace(text="hello")])
        assert _text(res) == "hello"


class TestInboxCommandRegistered:
    def test_inbox_help_lists_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["inbox", "--help"])
        assert result.exit_code == 0
        assert "--since" in result.stdout
        assert "--limit" in result.stdout
