from __future__ import annotations

from datetime import datetime

import pytest
from livekit.agents import llm

from durable_text.agents import Scheduler
from durable_text.framework import (
    DurableAgent,
    TurnState,
    WakeEvent,
    agent_class,
    delegate,
    on_event,
    registered,
    schedule,
)


def test_registry_and_lookup() -> None:
    assert agent_class("scheduler") is Scheduler
    assert set(registered()) >= {"front_desk", "scheduler", "dispatch"}
    with pytest.raises(KeyError):
        agent_class("nobody")


def test_on_event_collects_handlers_including_inherited(state: TurnState) -> None:
    class Base(DurableAgent):
        agent_id = "test_base"

        def instructions_for(self, state: TurnState) -> str:
            return "base"

        @on_event("a.happened")
        async def a(self, state: TurnState, event: WakeEvent) -> str | None:
            return "a"

    class Child(Base):
        agent_id = "test_child"

        @on_event("b.happened")
        async def b(self, state: TurnState, event: WakeEvent) -> str | None:
            return None

    child = Child(state)
    assert child.handles("a.happened") and child.handles("b.happened")
    assert not Base(state).handles("b.happened")
    assert child.id == "test_child"
    assert '<agent id="test_child"/>' in child.instructions


def test_delegate_ids_continue_from_existing_count(state: TurnState) -> None:
    state.delegation_seq = 2
    assert delegate(state, to="dispatch", task={"x": 1}) == "d3"
    assert delegate(state, to="dispatch", task={"x": 2}) == "d4"
    with pytest.raises(ValueError):
        delegate(state, to="nobody", task={})


def test_schedule_buffers_a_timer_on_the_sim_clock(state: TurnState) -> None:
    spec = schedule(state, "visit_reminder", at=datetime(2026, 10, 13, 12, 0, 30), payload={"a": 1})
    assert spec.due == "2026-10-13T12:00"
    assert state.effects.timers == [spec]


def test_handoff_copy_drops_the_dangling_handoff_call(state: TurnState) -> None:
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="leak")
    ctx.items.append(llm.FunctionCall(call_id="c1", name="list_slots", arguments="{}"))
    ctx.items.append(llm.FunctionCallOutput(call_id="c1", name="list_slots", output="[]", is_error=False))
    ctx.items.append(llm.FunctionCall(call_id="c2", name="hold_slot", arguments="{}"))  # no output yet
    nxt = Scheduler(state, chat_ctx=ctx).handoff_to(Scheduler, state)
    calls = [i.call_id for i in nxt.chat_ctx.items if i.type == "function_call"]
    assert calls == ["c1"]


def test_restored_agent_carries_its_chat_history(state: TurnState) -> None:
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="hi")
    ctx.add_message(role="assistant", content="hello")
    restored = llm.ChatContext.from_dict(ctx.to_dict(exclude_timestamp=False))
    agent = Scheduler(state, chat_ctx=restored, resumed=True)
    texts = [i.text_content for i in agent.chat_ctx.items if i.type == "message"]
    assert texts[-2:] == ["hi", "hello"]
