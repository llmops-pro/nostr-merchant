"""Tests for the engagement inbox workflow (deterministic gather + structured draft, v2 post)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nostr_merchant.cli import app
from nostr_merchant.workflows.engagement import (
    GATHER_READ_TOOLS,
    DraftedReply,
    InboxItem,
    _e_tags,
    _events,
    _text,
    append_outreach_ledger,
    append_replied_ledger,
    build_inbox_ledger_entry,
    load_replied_ledger,
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


class TestRepliedLedger:
    """The persistent ledger makes dedup survive the --since window and relay flakiness."""

    def test_missing_file_is_empty_set(self, tmp_path: Path) -> None:
        assert load_replied_ledger(tmp_path / "nope.json") == set()

    def test_append_then_load_roundtrips(self, tmp_path: Path) -> None:
        p = tmp_path / "replied.json"
        append_replied_ledger(p, ["aaa", "bbb"])
        assert load_replied_ledger(p) == {"aaa", "bbb"}

    def test_append_is_additive_across_calls(self, tmp_path: Path) -> None:
        p = tmp_path / "replied.json"
        append_replied_ledger(p, ["aaa"])
        append_replied_ledger(p, ["bbb", "ccc"])
        assert load_replied_ledger(p) == {"aaa", "bbb", "ccc"}

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        p = tmp_path / "deep" / "nested" / "replied.json"
        append_replied_ledger(p, ["aaa"])
        assert load_replied_ledger(p) == {"aaa"}

    def test_append_dedupes_and_drops_empties(self, tmp_path: Path) -> None:
        p = tmp_path / "replied.json"
        append_replied_ledger(p, ["aaa", "aaa", "", "bbb"])
        assert load_replied_ledger(p) == {"aaa", "bbb"}

    def test_empty_append_writes_nothing(self, tmp_path: Path) -> None:
        p = tmp_path / "replied.json"
        append_replied_ledger(p, [])
        assert not p.exists()

    def test_corrupt_line_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        p = tmp_path / "replied.json"
        p.write_text(
            '{"event_id":"aaa","ts":1}\nnot json at all\n{"event_id":"bbb","ts":2}\n',
            encoding="utf-8",
        )
        assert load_replied_ledger(p) == {"aaa", "bbb"}


class TestInboxCommandRegistered:
    def test_inbox_help_lists_options_including_post(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["inbox", "--help"])
        assert result.exit_code == 0
        assert "--since" in result.stdout
        assert "--limit" in result.stdout
        assert "--post" in result.stdout


class TestDraftedReplyClassification:
    def test_business_relevant_defaults_false(self) -> None:
        d = DraftedReply(event_id="x", action="draft", text="hi")
        assert d.business_relevant is False

    def test_business_relevant_settable(self) -> None:
        d = DraftedReply(event_id="x", action="draft", text="hi", business_relevant=True)
        assert d.business_relevant is True


class TestBuildInboxLedgerEntry:
    def test_entry_shape_and_counts(self) -> None:
        posted = [
            {
                "event_id": "aa" * 32, "reply_to": "ee" * 32, "to": "bb" * 32,
                "business_relevant": True, "reply_text": "our reply", "in_reply_to_excerpt": "their q",
            },
            {
                "event_id": "cc" * 32, "reply_to": "ff" * 32, "to": "dd" * 32,
                "business_relevant": False, "reply_text": "lol", "in_reply_to_excerpt": "banter",
            },
        ]
        e = build_inbox_ledger_entry(model="anthropic:claude-sonnet-4-6", posted=posted)
        assert e["type"] == "post"
        assert e["channel"] == "nostr"
        assert e["status"] == "done"
        assert e["auto_logged"] is True
        assert e["id"].endswith(tuple("0123456789"))  # date-inbox-HHMMSS
        assert "1 business-relevant, 1 social" in e["summary"]
        assert len(e["replies"]) == 2
        # the 0.3.4 fix: event_id is OUR reply, reply_to is the inbound event, plus our text
        assert e["replies"][0]["event_id"] == "aa" * 32
        assert e["replies"][0]["reply_to"] == "ee" * 32
        assert e["replies"][0]["reply_text"] == "our reply"
        assert e["replies"][0]["business_relevant"] is True
        assert set(e["links"]) == {"reply_1", "reply_2"}


class TestAppendOutreachLedger:
    def _seed(self, p: Path) -> None:
        p.write_text(
            json.dumps({"schema": "nostr-business-ledger/v1", "entries": [{"id": "old"}]}),
            encoding="utf-8",
        )

    def test_prepends_newest_first(self, tmp_path: Path) -> None:
        p = tmp_path / "ledger.json"
        self._seed(p)
        status = append_outreach_ledger(p, {"id": "new"})
        assert "ledger += new" in status
        data = json.loads(p.read_text(encoding="utf-8"))
        assert [e["id"] for e in data["entries"]] == ["new", "old"]

    def test_missing_file_skipped_not_created(self, tmp_path: Path) -> None:
        p = tmp_path / "nope.json"
        status = append_outreach_ledger(p, {"id": "new"})
        assert "not found" in status
        assert not p.exists()

    def test_corrupt_ledger_not_overwritten(self, tmp_path: Path) -> None:
        p = tmp_path / "ledger.json"
        p.write_text("{ this is not json", encoding="utf-8")
        status = append_outreach_ledger(p, {"id": "new"})
        assert "unreadable" in status
        assert p.read_text(encoding="utf-8") == "{ this is not json"  # untouched

    def test_unexpected_shape_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "ledger.json"
        p.write_text(json.dumps({"no_entries": True}), encoding="utf-8")
        status = append_outreach_ledger(p, {"id": "new"})
        assert "shape unexpected" in status
