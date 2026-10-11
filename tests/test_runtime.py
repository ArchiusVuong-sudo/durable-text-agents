"""Transactional turns with a scripted LLM: handoff, delegation, approval, timers, supersede, recovery."""

from __future__ import annotations

import asyncio
import json
import sqlite3

from durable_text import channel, db
from durable_text.config import Settings
from durable_text.llm.base import CompletionLLM, LLMCall, LLMRequest

from .conftest import Harness, SlowScripted


def thread(conn: sqlite3.Connection) -> sqlite3.Row:
    r = db.row(conn, "SELECT * FROM threads")
    assert r is not None
    return r


async def book_visit(h: Harness) -> None:
    h.text("hi, my dishwasher is leaking")
    h.text("water all over the kitchen floor")
    await h.drain()
    h.text("afternoon works")
    await h.drain()


async def test_coalesced_texts_make_one_turn_with_handoff(harness: Harness) -> None:
    harness.text("hi, my dishwasher is leaking")
    harness.text("water all over the kitchen floor")
    await harness.drain()
    [turn] = harness.turns("turn")
    assert turn["status"] == "committed"
    assert json.loads(turn["input_json"])["coalesced"] == 2
    trace = json.loads(turn["trace_json"])
    assert {"t": "handoff", "from": "front_desk", "to": "scheduler"} in trace
    assert [s["name"] for s in trace if s["t"] == "call"] == ["transfer_to_scheduler", "list_slots"]
    t = thread(harness.conn)
    assert (t["active_agent"], t["checkpoint_version"]) == ("scheduler", 1)
    [reply] = harness.outbound()
    assert reply["turn_id"] == turn["id"] and reply["encoding"] == "GSM-7"
    consumed = db.rows(harness.conn, "SELECT consumed_by_turn FROM messages WHERE direction = 'in'")
    assert {r["consumed_by_turn"] for r in consumed} == {turn["id"]}


async def test_next_turn_resumes_from_checkpoint_with_same_agent_and_history(harness: Harness) -> None:
    await book_visit(harness)
    first, second = harness.turns("turn")
    assert (second["agent_before"], second["checkpoint_from"], second["checkpoint_to"]) == ("scheduler", 1, 2)
    cp = db.row(harness.conn, "SELECT * FROM checkpoints WHERE version = 2")
    assert cp is not None
    items = json.loads(cp["chat_ctx_json"])["items"]
    users = [i["content"] for i in items if i.get("type") == "message" and i.get("role") == "user"]
    assert any("dishwasher" in str(u) for u in users)  # history from turn 1 is still there
    assert json.loads(cp["values_json"])["held_slot"] == "tue-1400"


async def test_delegation_runs_headless_and_waits_for_approval(harness: Harness) -> None:
    await book_visit(harness)
    [d] = db.rows(harness.conn, "SELECT * FROM delegations")
    assert d["status"] == "needs_approval"
    proposal = json.loads(d["result_json"])
    assert proposal["technician_name"] == "Priya Nair"
    [dispatch] = harness.turns("dispatch")
    assert dispatch["status"] == "committed"
    # the headless agent never texts the customer
    assert all(m["turn_id"] != dispatch["id"] for m in harness.outbound())
    slot = db.row(harness.conn, "SELECT status FROM slots WHERE id = 'tue-1400'")
    assert slot is not None and slot["status"] == "held"


async def test_approval_wakes_the_thread_and_schedules_timers(harness: Harness) -> None:
    await book_visit(harness)
    before = len(harness.outbound())
    channel.approve(harness.conn, "d1")
    await harness.drain()
    [wake] = harness.turns("wake")
    assert wake["status"] == "committed"
    out = harness.outbound()
    assert len(out) == before + 1 and "Priya Nair" in out[-1]["body"]
    timers = db.rows(harness.conn, "SELECT kind, due_sim, status FROM timers ORDER BY due_sim")
    assert [tuple(t) for t in timers] == [
        ("visit_reminder", "2026-10-13T12:00", "pending"),
        ("visit_followup", "2026-10-14T10:00", "pending"),
    ]


async def test_timer_fires_after_a_worker_restart(settings: Settings, conn: sqlite3.Connection) -> None:
    h = Harness(settings, conn)
    await book_visit(h)
    channel.approve(conn, "d1")
    await h.drain()
    # a brand-new worker (new runtime, nothing in memory) picks up the durable timer
    h2 = Harness(settings, conn)
    channel.jump_to_next_timer(conn)
    await h2.drain()
    wakes = h2.turns("wake")
    assert len(wakes) == 2 and wakes[-1]["status"] == "committed"
    assert "arrives" in h2.outbound()[-1]["body"]


async def test_stop_suppresses_later_timer_and_sends_nothing(harness: Harness) -> None:
    await book_visit(harness)
    channel.approve(harness.conn, "d1")
    await harness.drain()
    harness.text("STOP")
    sent_before = len(harness.outbound())
    for _ in range(2):
        channel.jump_to_next_timer(harness.conn)
        await harness.drain()
    timers = [r["status"] for r in db.rows(harness.conn, "SELECT status FROM timers ORDER BY due_sim")]
    assert timers == ["suppressed", "suppressed"]
    assert len(harness.outbound()) == sent_before
    assert len(harness.turns("wake")) == 1  # only the approval wake ran


async def test_new_text_supersedes_running_turn_and_rerun_sees_all(
    settings: Settings, conn: sqlite3.Connection
) -> None:
    h = Harness(settings, conn, lambda: SlowScripted(0.3))
    h.text("hi, my dishwasher is leaking")
    await h.drain()
    sent = len(h.outbound())
    h.text("tomorrow morning works")
    for _ in range(20):  # let the turn start and its first model call begin
        await h.step()
        if db.row(conn, "SELECT 1 FROM turns WHERE status = 'running'"):
            break
    h.text("actually afternoon is better")
    await h.drain()
    turns = h.turns("turn")
    assert [t["status"] for t in turns] == ["committed", "superseded", "committed"]
    superseded, rerun = turns[1], turns[2]
    assert all(m["turn_id"] != superseded["id"] for m in h.outbound())  # nothing from it was sent
    assert len(h.outbound()) == sent + 1
    assert json.loads(rerun["input_json"])["texts"] == [
        "tomorrow morning works",
        "actually afternoon is better",
    ]
    assert superseded["checkpoint_to"] is None and rerun["checkpoint_from"] == 1


class Exploding(CompletionLLM):
    def __init__(self) -> None:
        super().__init__("exploding")

    async def complete(self, req: LLMRequest) -> LLMCall:
        raise RuntimeError("model unavailable")


async def test_failed_turn_retries_then_gives_up(settings: Settings, conn: sqlite3.Connection) -> None:
    h = Harness(settings, conn, Exploding)
    h.text("hello")
    for _ in range(3):
        await h.step()
        await asyncio.sleep(0.05)
        conn.execute("UPDATE jobs SET run_after_ms = 0 WHERE status = 'queued'")
        await h.drain()
    job = db.row(conn, "SELECT * FROM jobs")
    assert job is not None and (job["status"], job["attempts"]) == ("failed", 3)
    assert [t["status"] for t in h.turns()] == ["failed"] * 3
    assert h.outbound() == []
    assert thread(conn)["checkpoint_version"] == 0  # nothing committed


async def test_commit_is_fenced_on_the_lease(settings: Settings, conn: sqlite3.Connection) -> None:
    h = Harness(settings, conn, lambda: SlowScripted(0.2))
    h.text("hi, my dishwasher is leaking")
    for _ in range(20):
        await h.step()
        if db.row(conn, "SELECT 1 FROM turns WHERE status = 'running'"):
            break
    conn.execute("UPDATE jobs SET lease_owner = 999999")  # another worker took the job over
    for _ in range(80):
        await h.step()
        if not h.worker.running:
            break
    assert h.outbound() == []
    assert thread(conn)["checkpoint_version"] == 0
    assert h.turns()[0]["status"] == "interrupted"


async def test_dead_worker_jobs_are_reclaimed_with_a_new_attempt(
    settings: Settings, conn: sqlite3.Connection
) -> None:
    h = Harness(settings, conn)
    h.text("hi, my dishwasher is leaking")
    # simulate a worker that claimed the job and then died (SIGKILL): running job, running turn
    conn.execute("INSERT INTO workers (pid, status, started_ms) VALUES (4242, 'killed', 0)")
    conn.execute(
        "UPDATE jobs SET status = 'running', lease_owner = 4242, lease_expires_ms = ?, attempts = 1",
        (db.now_ms() + 60_000,),
    )
    conn.execute(
        "INSERT INTO turns (job_id, thread_id, kind, attempt, status, worker_pid, sim_at, started_ms) "
        "VALUES (1, 't1', 'turn', 1, 'running', 4242, '2026-10-12T16:05', 0)"
    )
    await h.drain()
    old, new = h.turns()
    assert old["status"] == "interrupted" and "4242 died" in old["error"]
    assert (new["status"], new["attempt"], new["worker_pid"]) == ("committed", 2, h.worker.pid)
    assert len(h.outbound()) == 1
