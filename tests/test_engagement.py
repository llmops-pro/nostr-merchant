"""Tests for the engagement inbox workflow (deterministic gather + structured draft, v2 post)."""

from __future__ import annotations

from typer.testing import CliRunner

from nostr_merchant.cli import app
from nostr_merchant.workflows.engagement import (
    GATHER_READ_TOOLS,
    DraftedReply,
    InboxItem,
    _e_tags,
    _events,
    _text,
    render_queue,
)


class TestGatherIsReadOnly:
    """The gather only ever calls read tools; posting is separate and operator-gated."""

    def test_gather_only_uses_read_tools(self) -> None:
        forbidden = ("publish", "send", "delete", "pay", "confirm", "create", "update", "decrypt")
        for tool in GATHER_READ_TOOLS:
            assert not any(marker in tool for marker in forbidden), (
                f"{tool!r} looks like a write/spend tool — the gather must stay read-only"
            )
        assert "nostr_query_events" in GATHER_READ_TOOLS
        assert "nostr_get_pubkey" in GATHER_READ_TOOLS


class TestResultParsing:
    def test_events_extracts_list(self) -> None:
        evs = _events('{"count":2,"events":[{"id":"a"},{"id":"b"}]}')
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


class TestStructuredDrafts:
    def test_drafted_reply_defaults(self) -> None:
        d = DraftedReply(event_id="abc", action="skip", reason="bot")
        assert d.text == "" and d.reason == "bot"

    def test_render_queue_shows_drafts_and_skips_with_summary(self) -> None:
        items = {
            "aaa": InboxItem(
                event_id="aaa",
                author="auth1",
                author_pubkey="auth1-full-hex",
                content="great point about sovereignty",
                created_at=2,
                relation="reply",
                on_post_excerpt="my post",
            ),
            "bbb": InboxItem(
                event_id="bbb",
                author="auth2",
                author_pubkey="auth2-full-hex",
                content="buy my coin",
                created_at=1,
                relation="mention",
                on_post_excerpt="",
            ),
        }
        drafts = [
            DraftedReply(event_id="aaa", action="draft", text="appreciate it"),
            DraftedReply(event_id="bbb", action="skip", reason="spam"),
        ]
        out = render_queue(items, drafts)
        assert "DRAFT: appreciate it" in out
        assert "SKIP: spam" in out
        assert "1 drafted · 1 skipped" in out


class TestInboxCommandRegistered:
    def test_inbox_help_lists_options_including_post(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["inbox", "--help"])
        assert result.exit_code == 0
        assert "--since" in result.stdout
        assert "--limit" in result.stdout
        assert "--post" in result.stdout
