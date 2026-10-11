"""Opt-in live checks on the real Claude Agent SDK: ``LIVE=1 uv run pytest tests/test_live.py``.

They use LiveKit's own testing helpers (RunResult assertions and ``judge()``).
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
from dataclasses import replace

import pytest
from livekit.agents import AgentSession

from durable_text.agents import FrontDesk, Scheduler
from durable_text.config import Settings
from durable_text.envsafe import ALLOWED_KEYS
from durable_text.framework import TurnState
from durable_text.llm.base import LLMRequest, ToolSpec
from durable_text.llm.claude_sdk import ClaudeSDKLLM
from durable_text.llm.factory import detect_mode

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("LIVE") != "1", reason="set LIVE=1 to run live checks"),
]


@pytest.fixture(autouse=True)
def allowlisted_env(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    for key in list(os.environ):
        if key not in ALLOWED_KEYS:
            monkeypatch.delenv(key)
    if detect_mode(replace(settings, replay_forced=False)).mode != "live":
        pytest.skip("no Claude Code login and no ANTHROPIC_API_KEY")


def model() -> ClaudeSDKLLM:
    return ClaudeSDKLLM(os.environ.get("DTA_MODEL", "claude-sonnet-5-5"))


async def test_front_desk_hands_off_and_scheduler_offers_slots(state: TurnState) -> None:
    llm = model()
    async with AgentSession[TurnState](llm=llm, userdata=state) as session:
        await session.start(FrontDesk(state, resumed=True))
        result = await session.run(user_input="hi, my dishwasher is leaking all over the kitchen floor")
        result.expect.next_event().is_function_call(name="transfer_to_scheduler")
        result.expect.next_event().is_function_call_output()
        result.expect.next_event().is_agent_handoff(new_agent_type=Scheduler)
        result.expect.contains_function_call(name="list_slots")
        turn_calls = len(llm.calls)
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(llm, intent="Offers the customer one or more visit time windows to choose from")
        )
    assert all(c.source == "live" for c in llm.calls)
    assert turn_calls == 3  # handoff returns only the agent: no extra call for the old agent


async def test_status_question_reads_booking_and_does_not_promise(
    state: TurnState, conn: sqlite3.Connection
) -> None:
    conn.execute(
        "INSERT INTO delegations (id, thread_id, from_agent, to_agent, task_json, status, created_ms) "
        "VALUES ('d1', 't1', 'scheduler', 'dispatch', '{}', 'needs_approval', 0)"
    )
    state.values.update(
        {
            "issue": "dishwasher leaking",
            "held_slot": "tue-1400",
            "held_window": "Tue 13 Oct 14:00-16:00",
            "delegation_id": "d1",
            "dispatch_status": "pending",
        }
    )
    llm = model()
    async with AgentSession[TurnState](llm=llm, userdata=state) as session:
        await session.start(Scheduler(state, resumed=True))
        result = await session.run(user_input="any update on my repair?")
        result.expect.next_event().is_function_call(name="check_booking")
        result.expect.next_event().is_function_call_output()
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Says the time is held but the technician is not confirmed yet, without naming one",
            )
        )


async def test_cancelled_call_leaves_no_cli_child() -> None:
    llm = model()
    req = LLMRequest(
        agent_id="scheduler",
        system="Reply with a long friendly paragraph.",
        transcript=["[user] tell me about dishwasher care"],
        tools=[ToolSpec(name="noop")],
    )
    task = asyncio.create_task(llm.complete(req))
    await asyncio.sleep(2.5)  # the CLI child is up and talking to the model
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.5)
    children = subprocess.run(["pgrep", "-P", str(os.getpid())], capture_output=True, text=True).stdout
    assert children.strip() == ""
