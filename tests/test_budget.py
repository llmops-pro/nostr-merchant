"""Tests for `llmops_agent.budget`.

The tracker is file-backed and uses a clock that can be overridden via the
`now` constructor arg, so we can test the rolling-window math
deterministically without sleeping.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmops_agent.budget import BudgetTracker, SpendEntry


@pytest.fixture
def budget_path(tmp_path: Path) -> Path:
    return tmp_path / "budget.json"


class TestCanSpend:
    def test_within_caps_permits(self, budget_path: Path) -> None:
        tracker = BudgetTracker(
            path=budget_path, max_per_task_sats=100, max_per_day_sats=1000
        )
        ok, reason = tracker.can_spend(21)
        assert ok is True
        assert reason is None

    def test_zero_is_always_permitted(self, budget_path: Path) -> None:
        tracker = BudgetTracker(
            path=budget_path, max_per_task_sats=0, max_per_day_sats=0
        )
        ok, reason = tracker.can_spend(0)
        assert ok is True
        assert reason is None

    def test_negative_refused(self, budget_path: Path) -> None:
        tracker = BudgetTracker(
            path=budget_path, max_per_task_sats=100, max_per_day_sats=1000
        )
        ok, reason = tracker.can_spend(-5)
        assert ok is False
        assert reason is not None
        assert "non_negative" in reason

    def test_per_task_zero_refuses_any_spend(self, budget_path: Path) -> None:
        tracker = BudgetTracker(
            path=budget_path, max_per_task_sats=0, max_per_day_sats=1000
        )
        ok, reason = tracker.can_spend(1)
        assert ok is False
        assert reason is not None
        assert "agent_max_sats_per_task=0" in reason

    def test_per_task_cap_enforced_across_calls(self, budget_path: Path) -> None:
        tracker = BudgetTracker(
            path=budget_path, max_per_task_sats=100, max_per_day_sats=10_000
        )
        tracker.record_spend(sats=60, tool="t1")
        ok, reason = tracker.can_spend(50)  # would total 110 > 100
        assert ok is False
        assert reason is not None
        assert "agent_max_sats_per_task_exceeded" in reason

    def test_per_day_cap_enforced(self, budget_path: Path) -> None:
        # Use a generous per-task cap so we can stack spends across "tasks"
        # to exceed the per-day cap.
        tracker = BudgetTracker(
            path=budget_path,
            max_per_task_sats=1000,
            max_per_day_sats=200,
            now=1_000_000.0,
        )
        tracker.record_spend(sats=150, tool="t1")
        tracker.reset_per_task()
        ok, reason = tracker.can_spend(75)  # 150 + 75 > 200
        assert ok is False
        assert reason is not None
        assert "agent_max_sats_per_day_exceeded" in reason


class TestRecordAndSnapshot:
    def test_record_then_snapshot_reflects(self, budget_path: Path) -> None:
        tracker = BudgetTracker(
            path=budget_path,
            max_per_task_sats=100,
            max_per_day_sats=1000,
            now=2_000_000.0,
        )
        tracker.record_spend(sats=21, tool="bitcoin_block_height")
        snap = tracker.snapshot()
        assert snap.spent_today_sats == 21
        assert snap.spent_lifetime_sats == 21
        assert snap.per_task_spent_sats == 21
        assert snap.per_task_remaining_sats == 79
        assert snap.today_remaining_sats == 979

    def test_zero_or_negative_record_is_noop(self, budget_path: Path) -> None:
        tracker = BudgetTracker(
            path=budget_path, max_per_task_sats=100, max_per_day_sats=1000
        )
        tracker.record_spend(sats=0, tool="x")
        tracker.record_spend(sats=-5, tool="x")
        snap = tracker.snapshot()
        assert snap.spent_today_sats == 0
        assert snap.per_task_spent_sats == 0
        assert not budget_path.exists()  # nothing flushed to disk either

    def test_per_task_resets_but_persistent_remains(self, budget_path: Path) -> None:
        tracker = BudgetTracker(
            path=budget_path,
            max_per_task_sats=100,
            max_per_day_sats=1000,
            now=3_000_000.0,
        )
        tracker.record_spend(sats=21, tool="t1")
        tracker.reset_per_task()
        snap = tracker.snapshot()
        assert snap.per_task_spent_sats == 0
        assert snap.spent_today_sats == 21  # day window still has it


class TestRollingWindow:
    def test_entries_older_than_24h_drop_out(self, tmp_path: Path) -> None:
        path = tmp_path / "budget.json"
        # Write three entries directly: one yesterday-yesterday, one inside
        # the window, one fresh.
        DAY = 24 * 60 * 60
        now = 10_000_000.0
        entries = [
            SpendEntry(ts=now - 2 * DAY, sats=500, tool="ancient"),
            SpendEntry(ts=now - DAY / 2, sats=100, tool="recent"),
            SpendEntry(ts=now - 60, sats=21, tool="fresh"),
        ]
        with path.open("w", encoding="utf-8") as fh:
            import json

            for e in entries:
                fh.write(json.dumps(e.to_dict()) + "\n")

        tracker = BudgetTracker(
            path=path, max_per_task_sats=1000, max_per_day_sats=10_000, now=now
        )
        snap = tracker.snapshot()
        # Today window = last 24h: only "recent" + "fresh" count.
        assert snap.spent_today_sats == 121
        # Lifetime includes all three.
        assert snap.spent_lifetime_sats == 621


class TestPersistenceRoundtrip:
    def test_a_second_tracker_sees_prior_spends(self, budget_path: Path) -> None:
        t1 = BudgetTracker(
            path=budget_path,
            max_per_task_sats=100,
            max_per_day_sats=1000,
            now=5_000_000.0,
        )
        t1.record_spend(sats=21, tool="t1")
        t1.record_spend(sats=50, tool="t2")

        # Brand new tracker at the same `now` should read the file.
        t2 = BudgetTracker(
            path=budget_path,
            max_per_task_sats=100,
            max_per_day_sats=1000,
            now=5_000_000.0,
        )
        snap = t2.snapshot()
        assert snap.spent_today_sats == 71
        assert snap.spent_lifetime_sats == 71

    def test_corrupt_lines_are_skipped(self, budget_path: Path) -> None:
        budget_path.parent.mkdir(parents=True, exist_ok=True)
        budget_path.write_text(
            "\n".join(
                [
                    '{"ts": 1.0, "sats": 10, "tool": "ok1"}',
                    "not json at all",
                    '{"ts": 2.0, "sats": "bad", "tool": "t"}',  # bad sats type
                    '{"ts": 3.0, "sats": 5, "tool": "ok2"}',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        tracker = BudgetTracker(
            path=budget_path,
            max_per_task_sats=1000,
            max_per_day_sats=10_000,
            now=4.0,  # all entries are recent in this fake clock
        )
        snap = tracker.snapshot()
        assert snap.spent_lifetime_sats == 15  # only ok1 + ok2 count
