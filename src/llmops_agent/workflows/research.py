"""Self-paying research agent — the v0.1 reference workflow.

Entry point for `llmops-agent ask`. Builds the agent, runs it against the
user's question, audits every tool call + LLM completion, returns the
final answer + a structured budget snapshot.

Per-task budget resets at the start of every call.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agent import build_agent, make_process_tool_call
from ..audit import AuditLog
from ..budget import BudgetSnapshot, BudgetTracker
from ..config import AgentConfig


@dataclass
class ResearchResult:
    """What `run_research` returns. Tight, JSON-serializable shape."""

    answer: str
    budget_before: BudgetSnapshot
    budget_after: BudgetSnapshot


async def run_research(
    question: str,
    *,
    config: AgentConfig,
) -> ResearchResult:
    """Run the agent against a single user question."""
    audit = AuditLog(config.AGENT_AUDIT_PATH)
    budget = BudgetTracker(
        path=config.AGENT_BUDGET_PATH,
        max_per_task_sats=config.AGENT_MAX_SATS_PER_TASK,
        max_per_day_sats=config.AGENT_MAX_SATS_PER_DAY,
    )
    budget.reset_per_task()
    budget_before = budget.snapshot()

    await audit.record_startup(
        {
            "kind": "research_task",
            "model": config.LLMOPS_MODEL,
            "question_length": len(question),
            "budget_before": {
                "spent_today_sats": budget_before.spent_today_sats,
                "today_remaining_sats": budget_before.today_remaining_sats,
                "per_task_remaining_sats": budget_before.per_task_remaining_sats,
            },
        },
    )

    process = make_process_tool_call(
        budget=budget,
        audit=audit,
        max_tool_price=config.AGENT_MAX_TOOL_PRICE,
        tool_allowlist=config.tool_allowlist(),
        read_only=config.AGENT_READ_ONLY,
    )
    agent = build_agent(config, process_tool_call=process)

    try:
        async with agent:
            result = await agent.run(question)
        answer = str(result.output)
        await audit.record_llm_call(
            outcome="ok",
            input={"question_length": len(question)},
            result={"answer_length": len(answer)},
        )
    except Exception as err:
        await audit.record_llm_call(
            outcome="error",
            input={"question_length": len(question)},
            error=f"{type(err).__name__}: {err}",
        )
        raise

    budget_after = budget.snapshot()
    return ResearchResult(
        answer=answer,
        budget_before=budget_before,
        budget_after=budget_after,
    )
