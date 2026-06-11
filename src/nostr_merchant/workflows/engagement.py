"""Engagement inbox — triage replies/mentions on my NOSTR posts and draft (v1) or post (v2) replies.

Architecture: **deterministic gather + structured single-call draft**, with hard timeouts.

  - Gathering runs in Python, calling nostr-ops-mcp READ tools directly via pydantic-ai's
    `MCPServerStdio.direct_call_tool`, wrapped in `asyncio.timeout` so it can never hang.
  - Drafting is ONE LLM call (no tools) returning STRUCTURED drafts (draft|skip per item).
  - v1 (`inbox`): read-only — prints the drafted queue, posts nothing.
  - v2 (`inbox --post`): the CLI walks each draft (approve/edit/skip), a final confirm gate,
    then posts approved replies as NIP-10 replies via `nostr_publish_text_note`. The operator's
    per-item approval IS the gate — draft, don't autobot.

See 05-engagement-workflow-design.md.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

from ..agent import _resolve_model
from ..audit import AuditLog
from ..config import AgentConfig

# The only READ tools the gather calls. Posting (v2) uses nostr_publish_text_note, gated by the
# operator's per-item approval in the CLI + nostr-ops-mcp's own kind-allowlist + rate-limit.
GATHER_READ_TOOLS: tuple[str, ...] = ("nostr_get_pubkey", "nostr_query_events")
POST_TOOL = "nostr_publish_text_note"

GATHER_TIMEOUT_S = 75
MY_KINDS = [1, 30023]
NOTE_KIND = 1

DRAFT_PROMPT = """\
You are nostr-merchant's engagement assistant. For each inbound item below, decide whether to
draft a reply or skip it, and return a structured result — one entry per item.

# Voice (for drafts)
Terse, technical, sovereignty-aligned. Plain. No marketing-speak, no hype, no emoji spam. Match a
thoughtful FOSS / bitcoin / NOSTR builder. A good reply adds something — answers a question,
acknowledges a real point, extends the thread. **Match the language the person wrote in.** Never a
generic "thanks!".

# Decide
- action="draft", `text` = your reply — for items worth a genuine response.
- action="skip", `reason` = one line — for spam, bots, one-word low-effort replies, truncated
  auto-summaries, or anything you'd have nothing real to add to.
- `business_relevant` = true ONLY when a drafted reply advances the business (agent payments,
  NWC/L402, the kit, the playbook, a sale, a grant, a genuine prospect or technical question
  about what we build); false for social/community banter or off-topic chat.

Copy each item's event_id EXACTLY into your result.
"""


class DraftedReply(BaseModel):
    """One structured decision from the LLM, keyed to an inbound event."""

    event_id: str = Field(description="The inbound event id this addresses — copy it exactly.")
    action: Literal["draft", "skip"]
    text: str = Field(default="", description="The reply, in the operator's voice (when drafting).")
    reason: str = Field(default="", description="One-line reason (when skipping).")
    business_relevant: bool = Field(
        default=False,
        description=(
            "True if this reply advances the business (agent payments, NWC/L402, the kit, the "
            "playbook, a sale, a grant, a real prospect); False for social/community banter, "
            "off-topic chat, or anything not business-advancing."
        ),
    )


class DraftQueue(BaseModel):
    items: list[DraftedReply]


@dataclass
class InboxItem:
    """One open inbound item (a reply to my post, or a mention of me)."""

    event_id: str
    author: str  # short, for display
    author_pubkey: str  # full hex — needed to tag the parent author on a reply
    content: str
    created_at: int
    relation: str  # "reply" | "mention"
    on_post_excerpt: str


@dataclass
class InboxResult:
    queue: str
    drafts: list[DraftedReply]
    items_by_id: dict[str, InboxItem]
    since_hours: int
    my_post_count: int
    item_count: int


def _text(result: Any) -> str:
    """Coerce a tool result (str | dict | list-of-content-parts | wrapper) to a text payload."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, (bytes, bytearray)):
        return result.decode("utf-8", "replace")
    if isinstance(result, dict):
        return json.dumps(result)
    if isinstance(result, list):
        parts: list[str] = []
        for p in result:
            if isinstance(p, str):
                parts.append(p)
                continue
            t = getattr(p, "text", None)
            if t is None and isinstance(p, dict):
                t = p.get("text")
            if isinstance(t, str):
                parts.append(t)
        if parts:
            return "".join(parts)
    content = getattr(result, "content", None)
    if content is not None:
        return _text(content)
    t = getattr(result, "text", None)
    if isinstance(t, str):
        return t
    return str(result)


def _events(result: Any) -> list[dict[str, Any]]:
    data: Any = result
    if not isinstance(data, (dict, list)):
        try:
            data = json.loads(_text(result))
        except (ValueError, TypeError):
            return []
    if isinstance(data, dict):
        evs = data.get("events")
        return [e for e in evs if isinstance(e, dict)] if isinstance(evs, list) else []
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    return []


def _e_tags(ev: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for tag in ev.get("tags", []):
        if isinstance(tag, list) and len(tag) >= 2 and tag[0] == "e" and isinstance(tag[1], str):
            out.append(tag[1])
    return out


def load_replied_ledger(path: Path) -> set[str]:
    """Read the persistent set of event ids we've ever replied to.

    Stored as NDJSON (one `{"event_id": ..., "ts": ...}` per line) so appends are atomic and a
    single corrupt line can never wipe the ledger. Missing file → empty set.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return set()
    out: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # tolerate a corrupt line; keep the rest of the ledger
        eid = rec.get("event_id") if isinstance(rec, dict) else None
        if isinstance(eid, str) and eid:
            out.add(eid)
    return out


def append_replied_ledger(path: Path, event_ids: Iterable[str]) -> None:
    """Append event ids we've just replied to. Best-effort: IO failure never breaks a post run."""
    new = [e for e in dict.fromkeys(event_ids) if e]  # de-dupe, preserve order, drop empties
    if not new:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        now = int(time.time())
        lines = "".join(json.dumps({"event_id": e, "ts": now}) + "\n" for e in new)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(lines)
    except OSError:
        pass  # the relay-derived `answered` set still covers the in-window case


def build_inbox_ledger_entry(*, model: str, posted: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a nostr-business-ledger/v1 entry summarising one `inbox --post` session.

    `posted` is the list of SUCCESSFULLY published replies, each a dict with keys: `event_id`
    (OUR posted reply), `reply_to` (the inbound event we replied to), `to` (inbound author hex),
    `business_relevant` (bool), `reply_text` (what we said), `in_reply_to_excerpt` (what they
    said). FACTS only — Claude Code / the operator annotate the business-relevant ones later.
    """
    now = time.localtime()
    date = time.strftime("%Y-%m-%d", now)
    sid = time.strftime("%H%M%S", now)
    biz = sum(1 for p in posted if p.get("business_relevant"))
    soc = len(posted) - biz
    plural = "y" if len(posted) == 1 else "ies"
    links = {
        f"reply_{i + 1}": (
            f"{p.get('event_id', '')} → reply to {str(p.get('reply_to', ''))[:12]}… "
            f"(from {str(p.get('to', ''))[:12]}…)"
        )
        for i, p in enumerate(posted)
    }
    replies = [
        {
            "event_id": p.get("event_id", ""),  # our reply
            "reply_to": p.get("reply_to", ""),  # the inbound event
            "to": str(p.get("to", ""))[:12],
            "business_relevant": bool(p.get("business_relevant")),
            "reply_text": str(p.get("reply_text", ""))[:200],
            "in_reply_to_excerpt": str(p.get("in_reply_to_excerpt", ""))[:120],
        }
        for p in posted
    ]
    return {
        "id": f"{date}-inbox-{sid}",
        "date": date,
        "type": "post",
        "channel": "nostr",
        "status": "done",
        "summary": (
            f"Inbox pass ({model}): {len(posted)} repl{plural} posted "
            f"({biz} business-relevant, {soc} social). Auto-logged by `inbox --post`."
        ),
        "links": links,
        "auto_logged": True,
        "replies": replies,
        "context": (
            "Auto-logged by `nostr-merchant inbox --post` — facts only. Claude Code / the operator "
            "review and annotate the business-relevant ones (flags, accuracy checks, follow-ups). "
            "Social/banter replies are tagged business_relevant=false so the brief can hide them."
        ),
        "action_items": [],
        "next_check": None,
    }


def append_outreach_ledger(path: Path, entry: dict[str, Any]) -> str:
    """Prepend `entry` to the outreach ledger's `entries` array (newest-first), atomically.

    Defensive: never overwrites a ledger it can't parse (protects the hand-curated file).
    Returns a short status string; never raises into the caller.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return f"ledger not found at {path} — skipped"
    except (OSError, ValueError) as err:
        return f"ledger unreadable ({type(err).__name__}) — skipped, not overwritten"
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        return "ledger shape unexpected (no 'entries' list) — skipped"
    data["entries"].insert(0, entry)
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError as err:
        return f"ledger write failed ({type(err).__name__})"
    return f"ledger += {entry.get('id', '?')}"


def _nostr_server(config: AgentConfig) -> MCPServerStdio:
    spec = next((s for s in config.mcp_server_specs() if s.name == "nostr"), None)
    if spec is None:
        msg = "No 'nostr' server in the substrate — can't run engagement."
        raise RuntimeError(msg)
    kwargs: dict[str, Any] = {
        "command": spec.command,
        "args": list(spec.args),
        "id": "nostr",
        "timeout": 12,
    }
    if spec.cwd is not None:
        kwargs["cwd"] = spec.cwd
    if spec.env is not None:
        kwargs["env"] = spec.env
    return MCPServerStdio(**kwargs)


async def _gather(
    config: AgentConfig,
    *,
    since_ts: int,
    limit: int,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[int, list[InboxItem]]:
    """Query nostr-ops-mcp READ tools directly (no LLM) for open replies + mentions."""
    report = on_progress or (lambda _m: None)
    server = _nostr_server(config)
    report("connecting to nostr-ops-mcp…")
    async with asyncio.timeout(GATHER_TIMEOUT_S), server:
        tools = await server.list_tools()
        report(f"connected · {len(tools)} tools available")

        async def query(label: str, args: dict[str, Any]) -> list[dict[str, Any]]:
            evs = _events(await server.direct_call_tool("nostr_query_events", args))
            report(f"  {label}: {len(evs)}")
            return evs

        pk = json.loads(_text(await server.direct_call_tool("nostr_get_pubkey", {})) or "{}")
        mypub = pk.get("pubkey_hex")
        if not isinstance(mypub, str) or not mypub:
            msg = "nostr_get_pubkey returned no pubkey_hex — is a signer configured in nostr-ops-mcp/.env?"
            raise RuntimeError(msg)
        report(f"identified as {mypub[:12]}… · querying")

        my_posts = await query(
            "my recent posts",
            {"authors": [mypub], "kinds": MY_KINDS, "since": since_ts, "limit": 25},
        )
        post_by_id = {p["id"]: p for p in my_posts if isinstance(p.get("id"), str)}

        replies: list[dict[str, Any]] = []
        if post_by_id:
            replies = await query(
                "replies to my posts",
                {
                    "kinds": [NOTE_KIND],
                    "e_tag": list(post_by_id.keys()),
                    "since": since_ts,
                    "limit": 200,
                },
            )
        mentions = await query(
            "mentions of me",
            {"kinds": [NOTE_KIND], "p_tag": [mypub], "since": since_ts, "limit": 100},
        )
        my_outbound = await query(
            "my outbound notes",
            {"authors": [mypub], "kinds": [NOTE_KIND], "since": since_ts, "limit": 200},
        )
        # Window-bounded dedup (relay-derived) plus the persistent ledger (survives the --since
        # window and relay flakiness).
        answered: set[str] = {eid for ev in my_outbound for eid in _e_tags(ev)}
        ledger = load_replied_ledger(config.AGENT_REPLIED_PATH)
        if ledger:
            report(f"  ledger: {len(ledger)} already-replied")
        answered |= ledger

        seen: set[str] = set()
        items: list[InboxItem] = []
        for ev in [*replies, *mentions]:
            eid = ev.get("id")
            if (
                not isinstance(eid, str)
                or eid in seen
                or eid in post_by_id
                or eid in answered
                or ev.get("pubkey") == mypub
            ):
                continue
            seen.add(eid)
            target = next((t for t in reversed(_e_tags(ev)) if t in post_by_id), None)
            excerpt = (post_by_id[target].get("content") or "")[:80] if target else ""
            pubkey = str(ev.get("pubkey", ""))
            items.append(
                InboxItem(
                    event_id=eid,
                    author=pubkey[:12],
                    author_pubkey=pubkey,
                    content=(ev.get("content") or "")[:400],
                    created_at=int(ev.get("created_at", 0) or 0),
                    relation="reply" if target else "mention",
                    on_post_excerpt=excerpt,
                ),
            )

        items.sort(key=lambda i: i.created_at, reverse=True)
        return len(my_posts), items[:limit]


async def _draft(
    items: list[InboxItem],
    config: AgentConfig,
    model_override: str | None = None,
) -> list[DraftedReply]:
    """One LLM call (no tools) → structured draft/skip decision per item."""
    agent: Agent[None, DraftQueue] = Agent(
        model=_resolve_model(config, model_override),
        output_type=DraftQueue,
        system_prompt=DRAFT_PROMPT,
    )
    blocks = []
    for it in items:
        ctx = (
            f"reply on my post: “{it.on_post_excerpt}…”"
            if it.relation == "reply"
            else "mention of me"
        )
        blocks.append(
            f"event_id: {it.event_id}\n  {ctx}\n  they said: {it.content}",
        )
    result = await agent.run("Items:\n\n" + "\n\n".join(blocks))
    known = {it.event_id for it in items}
    return [d for d in result.output.items if d.event_id in known]


def render_queue(items_by_id: dict[str, InboxItem], drafts: list[DraftedReply]) -> str:
    """Human-readable review queue derived from the structured drafts."""
    if not drafts:
        return "(no drafts)"
    lines: list[str] = []
    drafted = skipped = 0
    for n, d in enumerate(drafts, 1):
        it = items_by_id.get(d.event_id)
        if it is None:
            continue
        ctx = f"reply on “{it.on_post_excerpt}…”" if it.relation == "reply" else "mention"
        head = f"{n}. from {it.author}… · {ctx} · event {d.event_id[:12]}…"
        if d.action == "skip":
            skipped += 1
            lines.append(f"{head}\n   SKIP: {d.reason}")
        else:
            drafted += 1
            lines.append(f"{head}\n   they said: {it.content[:160]}\n   DRAFT: {d.text}")
    summary = f"\n{len(drafts)} item(s) · {drafted} drafted · {skipped} skipped"
    return "\n\n".join(lines) + summary


async def _post_replies(
    config: AgentConfig,
    approved: list[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    """Publish approved replies as NIP-10 replies via nostr_publish_text_note.

    `approved` is a list of (reply_to_event_id, reply_to_author_hex, text). Audited.
    """
    audit = AuditLog(config.AGENT_AUDIT_PATH)
    server = _nostr_server(config)
    out: list[dict[str, Any]] = []
    posted_ids: list[str] = []
    async with asyncio.timeout(GATHER_TIMEOUT_S), server:
        await server.list_tools()  # prime the client
        for event_id, author_pubkey, text in approved:
            try:
                raw = await server.direct_call_tool(
                    POST_TOOL,
                    {
                        "content": text,
                        "reply_to_event_id": event_id,
                        "reply_to_author": author_pubkey,
                    },
                )
                parsed = json.loads(_text(raw) or "{}")
                ok = isinstance(parsed, dict) and "error" not in parsed
                if ok:
                    posted_ids.append(event_id)
                out.append({"reply_to": event_id, "ok": ok, "result": parsed})
                await audit.record_tool_call(
                    tool=POST_TOOL,
                    outcome="ok" if ok else "error",
                    input={"reply_to_event_id": event_id},
                    result=parsed if ok else None,
                    error=None if ok else json.dumps(parsed)[:200],
                )
            except Exception as err:
                out.append({"reply_to": event_id, "ok": False, "error": f"{type(err).__name__}: {err}"})
                await audit.record_tool_call(
                    tool=POST_TOOL,
                    outcome="error",
                    input={"reply_to_event_id": event_id},
                    error=f"{type(err).__name__}: {err}",
                )
    # Persist the parents we successfully answered so they never re-surface, regardless of
    # how far back a future --since reaches or whether relays still serve our outbound notes.
    append_replied_ledger(config.AGENT_REPLIED_PATH, posted_ids)
    return out


async def run_inbox(
    *,
    config: AgentConfig,
    since_hours: int = 48,
    limit: int = 20,
    model_override: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> InboxResult:
    """Gather open replies/mentions and draft responses. Gathering + drafting only — never posts."""
    audit = AuditLog(config.AGENT_AUDIT_PATH)
    since_ts = int(time.time()) - since_hours * 3600
    await audit.record_startup(
        {
            "kind": "engagement_inbox",
            "model": model_override or config.NOSTR_MERCHANT_MODEL,
            "since_hours": since_hours,
            "limit": limit,
        },
    )

    try:
        my_post_count, items = await _gather(
            config, since_ts=since_ts, limit=limit, on_progress=on_progress,
        )
    except TimeoutError as err:
        await audit.record_llm_call(outcome="error", input={"since_hours": since_hours}, error="gather timed out")
        msg = f"Gather timed out after {GATHER_TIMEOUT_S}s — nostr-ops-mcp may be unreachable."
        raise RuntimeError(msg) from err

    items_by_id = {it.event_id: it for it in items}
    if on_progress is not None:
        on_progress(
            f"gathered {my_post_count} recent post(s) · {len(items)} open item(s)"
            + (" · drafting…" if items else ""),
        )

    if not items:
        queue = (
            f"No new replies or mentions in the last {since_hours}h "
            f"across your {my_post_count} recent post(s)."
        )
        await audit.record_llm_call(outcome="ok", input={"since_hours": since_hours}, result={"items": 0})
        return InboxResult(queue, [], items_by_id, since_hours, my_post_count, 0)

    try:
        drafts = await _draft(items, config, model_override)
        await audit.record_llm_call(outcome="ok", input={"items": len(items)}, result={"drafts": len(drafts)})
    except Exception as err:
        await audit.record_llm_call(outcome="error", input={"items": len(items)}, error=f"{type(err).__name__}: {err}")
        raise

    return InboxResult(
        render_queue(items_by_id, drafts), drafts, items_by_id, since_hours, my_post_count, len(items),
    )
