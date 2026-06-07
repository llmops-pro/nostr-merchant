"""Engagement inbox — triage replies/mentions on my NOSTR posts and draft responses.

v1 architecture: **deterministic gather + single-call draft**, with hard timeouts.

  - Gathering runs IN PYTHON, calling nostr-ops-mcp's READ tools directly by name through
    pydantic-ai's `MCPServerStdio.direct_call_tool` (the same client the agent + `doctor` use —
    proven to work). It is the *raw* `mcp` stdio client that deadlocks on this setup, not this
    one. The whole gather is wrapped in `asyncio.timeout(GATHER_TIMEOUT_S)` so it can never hang
    the CLI — worst case it errors fast.
  - The LLM does ONE thing: draft replies for the already-gathered items, via an Agent built
    with NO toolsets — so it cannot loop, publish, DM, or spend.

Nothing is published. The operator reviews the queue and posts approved replies. A later
`inbox --post` adds an approval-gated publish step. See 05-engagement-workflow-design.md.
Governing principle: draft, don't autobot.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

from ..agent import _resolve_model
from ..audit import AuditLog
from ..config import AgentConfig

# The ONLY tools this workflow calls — both read-only. Drafting uses no tools at all.
GATHER_READ_TOOLS: tuple[str, ...] = ("nostr_get_pubkey", "nostr_query_events")

GATHER_TIMEOUT_S = 75  # hard ceiling on the whole gather — it must never hang the CLI again
MY_KINDS = [1, 30023]  # my posts: text notes + long-form articles
NOTE_KIND = 1

DRAFT_PROMPT = """\
You are nostr-merchant's engagement assistant. The operator's recent NOSTR posts received the
replies/mentions listed below. Draft an authentic reply to each, OR mark it SKIP.

# Voice
Terse, technical, sovereignty-aligned. Plain. No marketing-speak, no hype, no emoji spam. Match
how a thoughtful FOSS / bitcoin / NOSTR builder actually talks. A good reply adds something —
answers a question, acknowledges a real point, extends the thread. Never a generic "thanks!".
SKIP spam, bots, one-word low-effort replies, or anything you'd have nothing real to add to.

# Output (markdown)
A numbered queue, one entry per item, in the order given:
- **From:** <author> — **on:** <which post / mention, one line>
- **They said:** <short quote>
- **Event:** <event id>
- **Draft:** <your reply>   — or   **SKIP:** <one-line reason>

End with a single line: "N items · M drafted · K skipped".
"""


@dataclass
class InboxItem:
    """One open inbound item (a reply to my post, or a mention of me)."""

    event_id: str
    author: str
    content: str
    created_at: int
    relation: str  # "reply" | "mention"
    on_post_excerpt: str


@dataclass
class InboxResult:
    queue: str
    since_ts: int
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
    """Extract the `events` list from a query result (already-parsed dict/list, or JSON text)."""
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


def _nostr_server(config: AgentConfig) -> MCPServerStdio:
    spec = next((s for s in config.mcp_server_specs() if s.name == "nostr"), None)
    if spec is None:
        msg = "No 'nostr' server in the substrate — can't gather engagement."
        raise RuntimeError(msg)
    kwargs: dict[str, Any] = {
        "command": spec.command,
        "args": list(spec.args),
        "id": "nostr",
        "timeout": 12,  # per-operation timeout inside pydantic-ai's client
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
    """Query nostr-ops-mcp READ tools directly (no LLM) for open replies + mentions.

    Five sub-second queries, all read-only, bounded by ``GATHER_TIMEOUT_S``. Emits granular
    progress so a stall is visible (and pinpointable) rather than a blind wait.
    """
    report = on_progress or (lambda _m: None)
    server = _nostr_server(config)
    report("connecting to nostr-ops-mcp…")
    async with asyncio.timeout(GATHER_TIMEOUT_S), server:
        # list_tools first: the operation `doctor` uses successfully. It also primes
        # pydantic-ai's client so direct_call_tool can resolve tool names.
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
        answered: set[str] = {eid for ev in my_outbound for eid in _e_tags(ev)}

        # Assemble open items deterministically (still inside the server context — pure/fast).
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
            items.append(
                InboxItem(
                    event_id=eid,
                    author=str(ev.get("pubkey", ""))[:12],
                    content=(ev.get("content") or "")[:400],
                    created_at=int(ev.get("created_at", 0) or 0),
                    relation="reply" if target else "mention",
                    on_post_excerpt=excerpt,
                ),
            )

        items.sort(key=lambda i: i.created_at, reverse=True)
        return len(my_posts), items[:limit]


async def _draft(items: list[InboxItem], config: AgentConfig) -> str:
    """One LLM call, NO tools — turn gathered items into a reviewable draft queue."""
    agent: Agent[None, str] = Agent(model=_resolve_model(config), system_prompt=DRAFT_PROMPT)
    blocks = []
    for n, it in enumerate(items, 1):
        ctx = (
            f"reply on my post: “{it.on_post_excerpt}…”"
            if it.relation == "reply"
            else "mention of me"
        )
        blocks.append(
            f"{n}. [{it.relation}] event {it.event_id}\n"
            f"   author {it.author}…  ·  {ctx}\n"
            f"   they said: {it.content}",
        )
    result = await agent.run("Items:\n\n" + "\n\n".join(blocks))
    return str(result.output)


async def run_inbox(
    *,
    config: AgentConfig,
    since_hours: int = 48,
    limit: int = 20,
    on_progress: Callable[[str], None] | None = None,
) -> InboxResult:
    """Gather open replies/mentions on the operator's recent posts and draft responses.

    READ-ONLY end to end: gathering calls only nostr read tools; drafting uses a tool-less
    Agent. Nothing is published.
    """
    audit = AuditLog(config.AGENT_AUDIT_PATH)
    since_ts = int(time.time()) - since_hours * 3600
    await audit.record_startup(
        {
            "kind": "engagement_inbox",
            "model": config.NOSTR_MERCHANT_MODEL,
            "since_hours": since_hours,
            "limit": limit,
            "read_only": True,
        },
    )

    try:
        my_post_count, items = await _gather(
            config, since_ts=since_ts, limit=limit, on_progress=on_progress,
        )
    except TimeoutError as err:
        await audit.record_llm_call(
            outcome="error",
            input={"since_hours": since_hours},
            error="gather timed out",
        )
        msg = f"Gather timed out after {GATHER_TIMEOUT_S}s — nostr-ops-mcp may be unreachable."
        raise RuntimeError(msg) from err

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
        await audit.record_llm_call(
            outcome="ok",
            input={"since_hours": since_hours, "limit": limit},
            result={"items": 0},
        )
        return InboxResult(queue, since_ts, since_hours, my_post_count, 0)

    try:
        queue = await _draft(items, config)
        await audit.record_llm_call(
            outcome="ok",
            input={"items": len(items)},
            result={"queue_length": len(queue)},
        )
    except Exception as err:
        await audit.record_llm_call(
            outcome="error",
            input={"items": len(items)},
            error=f"{type(err).__name__}: {err}",
        )
        raise

    return InboxResult(queue, since_ts, since_hours, my_post_count, len(items))
