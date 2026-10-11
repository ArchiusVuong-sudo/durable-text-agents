"""Every agent tool, called directly with a TurnState: validation and buffered effects."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest
from livekit.agents import ToolError

from durable_text import agents
from durable_text.agents import Dispatch, FrontDesk, Scheduler, resolve_day
from durable_text.framework import TurnState


def ctx(state: TurnState) -> Any:
    return SimpleNamespace(userdata=state)


TASK = {
    "skill": "dishwasher",
    "zone": "East",
    "slot_id": "tue-1400",
    "start": "2026-10-13T14:00",
    "end": "2026-10-13T16:00",
    "window": "Tue 13 Oct 14:00-16:00",
    "issue": "leak",
}


def test_resolve_day(state: TurnState) -> None:
    assert resolve_day(state, "tomorrow") == "tue"
    assert resolve_day(state, "Thursday") == "thu"
    assert resolve_day(state, "2026-10-14") == "wed"
    assert resolve_day(state, "today") == "mon"
    with pytest.raises(ToolError):
        resolve_day(state, "someday")


async def test_list_slots_returns_only_open_future_windows(state: TurnState) -> None:
    out = json.loads(await Scheduler(state).list_slots(ctx(state), "tomorrow"))
    assert out["day"] == "tue"
    assert [s["slot_id"] for s in out["open"]] == ["tue-0800", "tue-1000", "tue-1400"]
    assert out["open"][2]["window"] == "Tue 13 Oct 14:00-16:00"
    assert json.loads(await Scheduler(state).list_slots(ctx(state), "today"))["open"] == []


async def test_hold_slot_buffers_effect_and_values(state: TurnState) -> None:
    out = json.loads(await Scheduler(state).hold_slot(ctx(state), "TUE-1400"))
    assert out == {"held": "tue-1400", "window": "Tue 13 Oct 14:00-16:00"}
    assert state.effects.holds == ["tue-1400"]
    assert state.values["held_slot"] == "tue-1400"
    row = state.conn.execute("SELECT status FROM slots WHERE id = 'tue-1400'").fetchone()
    assert row["status"] == "open"  # nothing written until the turn commits


@pytest.mark.parametrize(("slot", "msg"), [("tue-1600", "no longer open"), ("xyz", "No slot")])
async def test_hold_slot_rejects_bad_slots(state: TurnState, slot: str, msg: str) -> None:
    with pytest.raises(ToolError, match=msg):
        await Scheduler(state).hold_slot(ctx(state), slot)


async def test_request_dispatch_needs_a_hold_then_delegates(state: TurnState) -> None:
    s = Scheduler(state)
    with pytest.raises(ToolError, match="Hold a slot"):
        await s.request_dispatch(ctx(state), "leak")
    await s.hold_slot(ctx(state), "tue-1400")
    out = json.loads(await s.request_dispatch(ctx(state), "leak under the door"))
    assert out == {"delegation": "d1", "status": "sent to dispatch"}
    [d] = state.effects.delegations
    assert d.to == "dispatch" and d.task["slot_id"] == "tue-1400" and d.task["zone"] == "East"
    again = json.loads(await s.request_dispatch(ctx(state), "again"))
    assert again["status"] == "already requested"
    assert len(state.effects.delegations) == 1
    with pytest.raises(ToolError, match="already in progress"):
        await s.hold_slot(ctx(state), "tue-0800")


async def test_check_booking_reads_delegation_status(state: TurnState, conn: sqlite3.Connection) -> None:
    s = Scheduler(state)
    assert json.loads(await s.check_booking(ctx(state)))["dispatch"] == "none"
    state.values.update({"delegation_id": "d1", "held_window": "Tue 13 Oct 14:00-16:00"})
    conn.execute(
        "INSERT INTO delegations (id, thread_id, from_agent, to_agent, task_json, status, created_ms) "
        "VALUES ('d1', 't1', 'scheduler', 'dispatch', '{}', 'needs_approval', 0)"
    )
    out = json.loads(await s.check_booking(ctx(state)))
    assert "waiting for a dispatcher" in out["dispatch"]


async def test_transfer_to_scheduler_returns_the_next_agent(state: TurnState) -> None:
    nxt = await FrontDesk(state).transfer_to_scheduler(ctx(state), "dishwasher leaking")
    assert isinstance(nxt, Scheduler)
    assert state.values["issue"] == "dishwasher leaking"
    with pytest.raises(ToolError):
        await FrontDesk(state).transfer_to_scheduler(ctx(state), "  ")


async def test_list_technicians_marks_zone_and_availability(state: TurnState) -> None:
    state.values["task"] = TASK
    out = json.loads(await Dispatch(state).list_technicians(ctx(state), "dishwasher", "East"))
    fits = [t["name"] for t in out["technicians"] if t["in_zone"] and t["free_for_slot"]]
    assert fits == ["Priya Nair"]  # the synthetic data keeps the pick stable
    assert {t["name"] for t in out["technicians"]} == {
        "Priya Nair",
        "Grace Okafor",
        "Daniel Moreau",
        "Lena Fischer",
    }


@pytest.mark.parametrize(
    ("tech", "msg"),
    [
        ("tech-nobody", "Unknown technician"),
        ("tech-tomas", "does not have the dishwasher skill"),
        ("tech-daniel", "not free"),
        ("tech-grace", "zone"),
    ],
)
async def test_propose_assignment_validates(state: TurnState, tech: str, msg: str) -> None:
    state.values["task"] = TASK
    with pytest.raises(ToolError, match=msg):
        await Dispatch(state).propose_assignment(ctx(state), tech, "because")
    assert state.effects.proposal is None


async def test_propose_assignment_records_proposal_and_asks_for_no_reply(state: TurnState) -> None:
    state.values["task"] = TASK
    result = await Dispatch(state).propose_assignment(ctx(state), "tech-priya", "East zone, free, 4.9")
    assert result is None  # LiveKit then skips the follow-up LLM call
    p = state.effects.proposal
    assert p is not None and p.technician_name == "Priya Nair" and p.slot_id == "tue-1400"


async def test_wake_handlers_schedule_timers(state: TurnState) -> None:
    from durable_text.framework import WakeEvent

    event = WakeEvent(
        type="dispatch.approved",
        data={
            "technician_name": "Priya Nair",
            "window": TASK["window"],
            "start": TASK["start"],
            "end": TASK["end"],
        },
    )
    text = await Scheduler(state).dispatch_approved(state, event)
    assert "Priya Nair is confirmed" in text
    assert [(t.kind, t.due) for t in state.effects.timers] == [
        ("visit_reminder", "2026-10-13T12:00"),
        ("visit_followup", "2026-10-14T10:00"),
    ]
    assert state.values["dispatch_status"] == "confirmed"


def test_agents_are_registered() -> None:
    assert {a.agent_id for a in agents.AGENTS} == {"front_desk", "scheduler", "dispatch"}
    assert Dispatch.headless and not Scheduler.headless
