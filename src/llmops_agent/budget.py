"""Persistent rolling-window budget tracker for the agent layer.

Sits on top of nwc-mcp's own budget caps. Two layers of defense — if the
agent layer is compromised, the wallet layer still holds the line. Same
philosophy as the kind allowlist / read-only / two-step-confirm patterns
in the substrate.

Persistence: a JSON file (default `~/.llmops-agent/budget.json`) that records
every settled spend with timestamp + amount + tool name. Reads aggregate the
file on demand (rolling-window math at query time, not at write time). The
file is small in any realistic agent lifetime — bounded by the per-day cap,
not by uptime — so we don't bother with rotation in v0.1.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpendEntry:
    """A single settled spend event. Persisted as one JSON object on disk."""

    ts: float  # unix seconds, float for sub-second precision
    sats: int  # always non-negative
    tool: str  # tool name that consumed the budget

    def to_dict(self) -> dict[str, object]:
        return {"ts": self.ts, "sats": self.sats, "tool": self.tool}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SpendEntry:
        ts = data.get("ts")
        sats = data.get("sats")
        tool = data.get("tool")
        if not isinstance(ts, (int, float)):
            msg = f"SpendEntry.ts must be a number, got {type(ts).__name__}"
            raise ValueError(msg)
        if not isinstance(sats, int) or sats < 0:
            msg = f"SpendEntry.sats must be a non-negative int, got {sats!r}"
            raise ValueError(msg)
        if not isinstance(tool, str) or not tool:
            msg = f"SpendEntry.tool must be a non-empty string, got {tool!r}"
            raise ValueError(msg)
        return cls(ts=float(ts), sats=sats, tool=tool)


@dataclass(frozen=True)
class BudgetSnapshot:
    """A point-in-time view across all configured windows."""

    spent_today_sats: int
    spent_lifetime_sats: int
    today_remaining_sats: int
    per_task_remaining_sats: int
    per_task_spent_sats: int


class BudgetTracker:
    """File-backed rolling-window budget tracker.

    Construction does not perform IO — the file is read lazily on the first
    query or record call. This makes the tracker cheap to instantiate in
    tests and CLI startup paths.

    Caps are passed in at construction (typically from `AgentConfig`).
    Negative or zero caps mean "no spending allowed" (everything refused).
    """

    SECONDS_PER_DAY = 24 * 60 * 60

    def __init__(
        self,
        *,
        path: Path,
        max_per_task_sats: int,
        max_per_day_sats: int,
        now: float | None = None,
    ) -> None:
        self._path = path
        self._max_per_task = int(max_per_task_sats)
        self._max_per_day = int(max_per_day_sats)
        # Per-task spend lives in memory only — reset between agent
        # invocations. Lifetime + 24h windows are persisted.
        self._per_task_sats = 0
        # Override clock for tests.
        self._now_fn: Callable[[], float] = (
            (lambda: float(now)) if now is not None else time.time
        )

    # ---- Read path ----

    def snapshot(self) -> BudgetSnapshot:
        """Aggregate the persisted log against the configured caps."""
        entries = self._load()
        now = self._now_fn()
        cutoff = now - self.SECONDS_PER_DAY

        spent_today = sum(e.sats for e in entries if e.ts >= cutoff)
        spent_lifetime = sum(e.sats for e in entries)

        return BudgetSnapshot(
            spent_today_sats=spent_today,
            spent_lifetime_sats=spent_lifetime,
            today_remaining_sats=max(0, self._max_per_day - spent_today),
            per_task_spent_sats=self._per_task_sats,
            per_task_remaining_sats=max(0, self._max_per_task - self._per_task_sats),
        )

    def can_spend(self, sats: int) -> tuple[bool, str | None]:
        """Check whether a proposed spend fits within all configured caps.

        Returns `(True, None)` when permitted, `(False, reason)` when blocked.
        """
        if sats < 0:
            return False, f"spend_must_be_non_negative ({sats})"
        if sats == 0:
            return True, None
        if self._max_per_task <= 0:
            return False, "agent_max_sats_per_task=0 — all spending disabled"
        if self._max_per_day <= 0:
            return False, "agent_max_sats_per_day=0 — all spending disabled"

        snap = self.snapshot()
        if self._per_task_sats + sats > self._max_per_task:
            return (
                False,
                (
                    f"agent_max_sats_per_task_exceeded "
                    f"(would spend {self._per_task_sats + sats}, cap {self._max_per_task})"
                ),
            )
        if snap.spent_today_sats + sats > self._max_per_day:
            return (
                False,
                (
                    f"agent_max_sats_per_day_exceeded "
                    f"(would spend {snap.spent_today_sats + sats} today, cap {self._max_per_day})"
                ),
            )
        return True, None

    # ---- Write path ----

    def record_spend(self, *, sats: int, tool: str) -> None:
        """Persist a settled spend. Caller is responsible for calling
        `can_spend()` first when atomicity matters.
        """
        if sats <= 0:
            return  # nothing to record
        entry = SpendEntry(ts=self._now_fn(), sats=int(sats), tool=tool)
        self._append(entry)
        self._per_task_sats += entry.sats

    def reset_per_task(self) -> None:
        """Clear the per-task spend counter — call at the start of each
        new agent invocation.
        """
        self._per_task_sats = 0

    # ---- Persistence helpers ----

    def _load(self) -> list[SpendEntry]:
        if not self._path.exists():
            return []
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError:
            return []
        out: list[SpendEntry] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            try:
                out.append(SpendEntry.from_dict(data))
            except ValueError:
                continue
        return out

    def _append(self, entry: SpendEntry) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_dict()) + "\n")
