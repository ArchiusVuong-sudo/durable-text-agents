"""The three agents of the synthetic repair desk: plain LiveKit agents plus the durable layer."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from livekit.agents import RunContext, ToolError, function_tool

from . import clock, db, seed
from .framework import DurableAgent, Proposal, TurnState, WakeEvent, delegate, on_event, schedule

WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _customer_line(state: TurnState) -> str:
    c = state.customer
    return f"Customer on file: {c['name']}, {c['zone']} zone, {c['appliance_detail']}."


def resolve_day(state: TurnState, day: str) -> str:
    """Accept 'tomorrow', 'today', a weekday name or an ISO date; return a weekday key."""
    text = day.strip().lower()
    if text in ("today", ""):
        return state.now.strftime("%a").lower()
    if text == "tomorrow":
        return (state.now + timedelta(days=1)).strftime("%a").lower()
    for key in WEEKDAYS:
        if text.startswith(key):
            return key
    try:
        return datetime.fromisoformat(text[:10]).strftime("%a").lower()
    except ValueError:
        raise ToolError(f"Unknown day {day!r}. Use 'tomorrow' or a weekday like 'Tuesday'.") from None


def _slot(state: TurnState, slot_id: str) -> dict[str, Any]:
    r = db.row(state.conn, "SELECT * FROM slots WHERE id = ?", slot_id.strip().lower())
    if r is None:
        raise ToolError(f"No slot {slot_id!r}. Call list_slots first and use a slot_id from it.")
    return dict(r)


class FrontDesk(DurableAgent):
    agent_id = "front_desk"

    def instructions_for(self, state: TurnState) -> str:
        return (
            "You are the front desk of Kettle & Coil Repair, a fictional appliance repair shop, "
            f"texting with a customer. {_customer_line(state)} "
            "Your only job is to understand the problem. As soon as the customer describes a "
            "fault that needs a technician visit, call transfer_to_scheduler with a one-line "
            "summary. Do not offer times yourself. If they only say hello, ask what needs fixing."
        )

    @function_tool
    async def transfer_to_scheduler(self, context: RunContext[TurnState], issue: str) -> DurableAgent:
        """Hand the conversation to the scheduler once the customer needs a technician visit.

        Args:
            issue: one-line summary of the fault, for example 'dishwasher leaking onto the floor'
        """
        if not issue.strip():
            raise ToolError("issue must be a short summary of the fault")
        context.userdata.values["issue"] = issue.strip()[:200]
        # Returning only the next agent (no message) skips a follow-up LLM call here.
        return self.handoff_to(Scheduler, context.userdata)


class Scheduler(DurableAgent):
    agent_id = "scheduler"

    def instructions_for(self, state: TurnState) -> str:
        issue = state.values.get("issue", "not recorded yet")
        return (
            "You are the scheduler at Kettle & Coil Repair (fictional), texting with the customer. "
            f"Now: {clock.human(state.now)}. {_customer_line(state)} Issue: {issue}.\n"
            "Rules:\n"
            "1. To offer times, call list_slots for the day they want (tomorrow if they did not "
            "say) and offer the open windows in one short text.\n"
            "2. When the customer picks a time, call hold_slot with its slot_id, then call "
            "request_dispatch, then text that the time is held and dispatch is confirming a "
            "technician, and that you will text once it is confirmed.\n"
            "3. If they change their mind, hold the new slot instead (one hold at a time).\n"
            "4. For status questions call check_booking and answer from it. Never name or promise "
            "a technician before dispatch is approved.\n"
            "Keep every text short and friendly."
        )

    @function_tool
    async def list_slots(self, context: RunContext[TurnState], day: str) -> str:
        """List open 2-hour visit windows for one day.

        Args:
            day: 'tomorrow', a weekday such as 'Tuesday', or a date like 2026-10-13
        """
        state = context.userdata
        key = resolve_day(state, day)
        rows = db.rows(
            state.conn,
            "SELECT * FROM slots WHERE lower(day) = ? AND status = 'open' ORDER BY start_sim",
            key,
        )
        open_slots = [
            {"slot_id": r["id"], "window": clock.window(clock.parse(r["start_sim"]), clock.parse(r["end_sim"]))}
            for r in rows
            if clock.parse(r["start_sim"]) > state.now
        ]
        return json.dumps({"day": key, "open": open_slots})

    @function_tool
    async def hold_slot(self, context: RunContext[TurnState], slot_id: str) -> str:
        """Hold one visit window for this customer (replaces any earlier hold).

        Args:
            slot_id: a slot_id returned by list_slots, for example 'tue-1400'
        """
        state = context.userdata
        slot = _slot(state, slot_id)
        if slot["status"] != "open" and slot.get("thread_id") != state.thread_id:
            raise ToolError(f"{slot['id']} is no longer open. Call list_slots again.")
        if clock.parse(slot["start_sim"]) <= state.now:
            raise ToolError(f"{slot['id']} is in the past.")
        if state.values.get("dispatch_status") in ("pending", "confirmed"):
            raise ToolError("A technician request is already in progress for the held slot.")
        state.effects.holds = [slot["id"]]
        state.values["held_slot"] = slot["id"]
        window = clock.window(clock.parse(slot["start_sim"]), clock.parse(slot["end_sim"]))
        state.values["held_window"] = window
        return json.dumps({"held": slot["id"], "window": window})

    @function_tool
    async def request_dispatch(self, context: RunContext[TurnState], notes: str) -> str:
        """Ask dispatch to pick a technician for the held slot. A dispatcher approves the pick.

        Args:
            notes: short notes for the technician about the fault
        """
        state = context.userdata
        slot_id = state.values.get("held_slot")
        if not slot_id:
            raise ToolError("Hold a slot with hold_slot before requesting dispatch.")
        if state.values.get("dispatch_status") in ("pending", "confirmed"):
            return json.dumps({"delegation": state.values.get("delegation_id"), "status": "already requested"})
        slot = _slot(state, slot_id)
        task = {
            "skill": state.customer["appliance"],
            "zone": state.customer["zone"],
            "slot_id": slot_id,
            "start": slot["start_sim"],
            "end": slot["end_sim"],
            "window": state.values.get("held_window"),
            "issue": state.values.get("issue", ""),
            "notes": notes.strip()[:300],
        }
        delegation_id = delegate(state, to="dispatch", task=task)
        state.values["delegation_id"] = delegation_id
        state.values["dispatch_status"] = "pending"
        return json.dumps({"delegation": delegation_id, "status": "sent to dispatch"})

    @function_tool
    async def check_booking(self, context: RunContext[TurnState]) -> str:
        """Current booking status for this customer: held slot and dispatch progress."""
        state = context.userdata
        delegation_id = state.values.get("delegation_id")
        status = "none"
        if delegation_id:
            r = db.row(state.conn, "SELECT status FROM delegations WHERE id = ?", delegation_id)
            status = {
                "running": "dispatch is picking a technician",
                "needs_approval": "dispatch proposed a technician, waiting for a dispatcher to confirm",
                "approved": "technician confirmed",
                "failed": "dispatch could not find a technician",
            }.get(r["status"] if r else "", "requested")
        return json.dumps(
            {
                "held_window": state.values.get("held_window"),
                "dispatch": status,
                "technician": state.values.get("technician"),
            }
        )

    @on_event("dispatch.approved")
    async def dispatch_approved(self, state: TurnState, event: WakeEvent) -> str:
        d = event.data
        start, end = clock.parse(d["start"]), clock.parse(d["end"])
        state.values["dispatch_status"] = "confirmed"
        state.values["technician"] = d["technician_name"]
        reminder = {"technician_name": d["technician_name"], "window": d["window"]}
        schedule(state, "visit_reminder", at=start - timedelta(hours=2), payload=reminder)
        schedule(state, "visit_followup", at=end + timedelta(hours=18), payload=reminder)
        return (
            f"Event from dispatch: {d['technician_name']} is confirmed for {d['window']}. "
            "Text the customer this confirmation in one short message and say we will text a "
            "reminder before the visit."
        )

    @on_event("timer.visit_reminder")
    async def visit_reminder(self, state: TurnState, event: WakeEvent) -> str:
        d = event.data
        return (
            f"Event from a reminder timer: {d['technician_name']} arrives {d['window']}. "
            "Text a short reminder and ask the customer to clear space around the appliance."
        )

    @on_event("timer.visit_followup")
    async def visit_followup(self, state: TurnState, event: WakeEvent) -> str:
        d = event.data
        return (
            f"Event from a follow-up timer: the visit was {d['window']}. "
            "Ask in one short text how the repair went."
        )


class Dispatch(DurableAgent):
    """Headless: runs in its own AgentSession, never texts the customer."""

    agent_id = "dispatch"
    headless = True

    def instructions_for(self, state: TurnState) -> str:
        task = state.values.get("task", {})
        return (
            "You are the dispatch agent of Kettle & Coil Repair (fictional). You never talk to "
            "customers. Pick one technician for the task: call list_technicians with the "
            "required skill and zone, then call propose_assignment for the best technician who "
            "is free for the slot, with a one-sentence reason. "
            f"Task: {json.dumps(task, sort_keys=True)}"
        )

    @function_tool
    async def list_technicians(self, context: RunContext[TurnState], skill: str, zone: str) -> str:
        """Technicians with a skill, with their zone, rating and whether they are free for the task slot.

        Args:
            skill: appliance skill, for example 'dishwasher'
            zone: service zone of the customer, for example 'East'
        """
        state = context.userdata
        task = state.values.get("task", {})
        start, end = clock.parse(task["start"]), clock.parse(task["end"])
        out = []
        for r in db.rows(state.conn, "SELECT * FROM technicians ORDER BY id"):
            skills = json.loads(r["skills_json"])
            if skill.strip().lower() not in skills:
                continue
            out.append(
                {
                    "technician_id": r["id"],
                    "name": r["name"],
                    "zone": r["zone"],
                    "rating": r["rating"],
                    "in_zone": r["zone"].lower() == zone.strip().lower(),
                    "free_for_slot": seed.technician_free(json.loads(r["windows_json"]), start, end),
                }
            )
        return json.dumps({"slot": task.get("window"), "technicians": out})

    @function_tool
    async def propose_assignment(
        self, context: RunContext[TurnState], technician_id: str, reason: str
    ) -> None:
        """Propose one technician for the task slot. A human dispatcher approves it.

        Args:
            technician_id: a technician_id from list_technicians
            reason: one sentence on why this technician fits
        """
        state = context.userdata
        task = state.values.get("task", {})
        r = db.row(state.conn, "SELECT * FROM technicians WHERE id = ?", technician_id.strip())
        if r is None:
            raise ToolError(f"Unknown technician_id {technician_id!r}.")
        if task.get("skill") not in json.loads(r["skills_json"]):
            raise ToolError(f"{r['name']} does not have the {task.get('skill')} skill.")
        start, end = clock.parse(task["start"]), clock.parse(task["end"])
        if not seed.technician_free(json.loads(r["windows_json"]), start, end):
            raise ToolError(f"{r['name']} is not free for {task.get('window')}.")
        if r["zone"].lower() != str(task.get("zone", "")).lower():
            raise ToolError(f"{r['name']} works in the {r['zone']} zone, not {task.get('zone')}.")
        state.effects.proposal = Proposal(
            technician_id=r["id"],
            technician_name=r["name"],
            slot_id=task["slot_id"],
            window=task["window"],
            reason=reason.strip()[:300] or "Best match for skill, zone and time.",
        )
        # Returning None asks LiveKit for no follow-up reply: the headless run ends here.
        return None


AGENTS: tuple[type[DurableAgent], ...] = (FrontDesk, Scheduler, Dispatch)
