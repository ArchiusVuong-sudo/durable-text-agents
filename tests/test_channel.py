"""Channel rules that run before any agent code: dedupe, keywords, coalescing, supersede."""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta

import pytest

from durable_text import channel, clock, db, seed
from durable_text.channel import InboundSMS

PHONE, BIZ = seed.CUSTOMER["phone"], seed.BUSINESS_NUMBER


def sms(body: str, sid: str | None = None, frm: str = PHONE) -> InboundSMS:
    return InboundSMS(From=frm, To=BIZ, Body=body, **({"MessageSid": sid} if sid else {}))


def jobs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return db.rows(conn, "SELECT * FROM jobs ORDER BY id")


def test_duplicate_message_sid_is_ignored(conn: sqlite3.Connection) -> None:
    assert channel.receive_inbound(conn, sms("hello", "m1"), 3).outcome == "accepted"
    again = channel.receive_inbound(conn, sms("hello", "m1"), 3)
    assert again.outcome == "duplicate"
    assert len(db.rows(conn, "SELECT * FROM messages")) == 1
    assert len(jobs(conn)) == 1
    assert [w["outcome"] for w in db.rows(conn, "SELECT * FROM webhooks ORDER BY id")] == [
        "accepted",
        "duplicate",
    ]


def test_replay_last_webhook_is_a_duplicate(conn: sqlite3.Connection) -> None:
    assert channel.replay_last_webhook(conn, 3) is None
    channel.receive_inbound(conn, sms("hello", "m9"), 3)
    result = channel.replay_last_webhook(conn, 3)
    assert result is not None and result.outcome == "duplicate"


def test_unknown_number_is_ignored(conn: sqlite3.Connection) -> None:
    assert channel.receive_inbound(conn, sms("hi", frm="+13125550199"), 3).outcome == "unknown_number"
    assert jobs(conn) == []


def test_texts_inside_the_window_coalesce_into_one_job(conn: sqlite3.Connection) -> None:
    channel.receive_inbound(conn, sms("one"), 3)
    first_run_after = jobs(conn)[0]["run_after_ms"]
    detail = channel.receive_inbound(conn, sms("two"), 3).detail
    assert "coalesced" in detail
    assert len(jobs(conn)) == 1
    assert jobs(conn)[0]["run_after_ms"] >= first_run_after  # debounce pushes the start


def test_text_during_a_running_turn_requests_supersede(conn: sqlite3.Connection) -> None:
    channel.receive_inbound(conn, sms("one"), 0)
    conn.execute("UPDATE jobs SET status = 'running', lease_owner = 1")
    result = channel.receive_inbound(conn, sms("two"), 0)
    assert "supersedes running job" in result.detail
    running, queued = jobs(conn)
    assert running["cancel_requested"] == 1
    assert queued["status"] == "queued"


def test_stop_opts_out_with_template_and_cancels_pending_turns(conn: sqlite3.Connection) -> None:
    channel.receive_inbound(conn, sms("my oven is broken"), 3)
    result = channel.receive_inbound(conn, sms("STOP"), 3)
    assert result.outcome == "keyword:stop"
    thread = db.row(conn, "SELECT * FROM threads")
    assert thread is not None and thread["opted_out"] == 1
    out = db.rows(conn, "SELECT * FROM messages WHERE direction = 'out'")
    assert len(out) == 1 and out[0]["kind"] == "template" and "unsubscribed" in out[0]["body"]
    assert jobs(conn)[0]["status"] == "cancelled"


def test_opted_out_thread_gets_no_turn_until_start(conn: sqlite3.Connection) -> None:
    channel.receive_inbound(conn, sms("STOP"), 3)
    assert channel.receive_inbound(conn, sms("hello?"), 3).outcome == "opted_out"
    assert jobs(conn) == []
    assert channel.receive_inbound(conn, sms("START"), 3).outcome == "keyword:start"
    thread = db.row(conn, "SELECT * FROM threads")
    assert thread is not None and thread["opted_out"] == 0
    assert channel.receive_inbound(conn, sms("hello again"), 3).outcome == "accepted"


def test_help_answers_from_template_without_a_job(conn: sqlite3.Connection) -> None:
    assert channel.receive_inbound(conn, sms("help"), 3).outcome == "keyword:help"
    assert jobs(conn) == []
    out = db.rows(conn, "SELECT * FROM messages WHERE direction = 'out'")
    assert out[0]["kind"] == "template"


def test_fake_carrier_delivers_and_suppresses_for_opted_out(conn: sqlite3.Connection) -> None:
    now = db.now_ms()
    for kind in ("agent", "template"):
        conn.execute(
            "INSERT INTO messages (thread_id, direction, body, kind, status, sim_at, created_ms) "
            "VALUES ('t1', 'out', 'x', ?, 'queued', '2026-10-12T16:05', ?)",
            (kind, now - 1000),
        )
    conn.execute("UPDATE threads SET opted_out = 1")
    assert channel.deliver_outbox(conn) == 2
    statuses = [r["status"] for r in db.rows(conn, "SELECT status FROM messages ORDER BY id")]
    assert statuses == ["suppressed", "delivered"]


def _proposal(conn: sqlite3.Connection, status: str = "needs_approval") -> None:
    task = {
        "slot_id": "tue-1400",
        "start": "2026-10-13T14:00",
        "end": "2026-10-13T16:00",
        "window": "Tue 13 Oct 14:00-16:00",
        "skill": "dishwasher",
        "zone": "East",
    }
    result = {
        "technician_id": "tech-priya",
        "technician_name": "Priya Nair",
        "slot_id": "tue-1400",
        "window": "Tue 13 Oct 14:00-16:00",
        "reason": "fits",
    }
    conn.execute(
        "INSERT INTO delegations (id, thread_id, from_agent, to_agent, task_json, status, result_json, "
        "created_ms) VALUES ('d1', 't1', 'scheduler', 'dispatch', ?, ?, ?, 0)",
        (json.dumps(task), status, json.dumps(result)),
    )


def test_approve_books_slot_and_enqueues_wake(conn: sqlite3.Connection) -> None:
    _proposal(conn)
    event = channel.approve(conn, "d1")
    assert event["type"] == "dispatch.approved"
    slot = db.row(conn, "SELECT * FROM slots WHERE id = 'tue-1400'")
    assert slot is not None and slot["status"] == "booked" and slot["technician_id"] == "tech-priya"
    wake = jobs(conn)[0]
    assert wake["kind"] == "wake"
    assert json.loads(wake["payload_json"])["event"]["data"]["technician_name"] == "Priya Nair"


def test_approve_twice_is_refused(conn: sqlite3.Connection) -> None:
    _proposal(conn)
    channel.approve(conn, "d1")
    with pytest.raises(channel.ApprovalError):
        channel.approve(conn, "d1")
    with pytest.raises(channel.ApprovalError):
        channel.approve(conn, "nope")


def _timer(conn: sqlite3.Connection, kind: str, due: str) -> None:
    conn.execute(
        "INSERT INTO timers (thread_id, kind, due_sim, payload_json) VALUES ('t1', ?, ?, '{}')", (kind, due)
    )


def test_timers_fire_on_the_simulated_clock_only(conn: sqlite3.Connection) -> None:
    _timer(conn, "visit_reminder", "2026-10-13T12:00")
    assert channel.fire_due_timers(conn) == 0
    channel.advance_clock(conn, 3)
    assert channel.fire_due_timers(conn) == 0
    assert channel.jump_to_next_timer(conn) == "2026-10-13T12:00"
    assert channel.fire_due_timers(conn) == 1
    timer = db.row(conn, "SELECT * FROM timers")
    assert timer is not None and timer["status"] == "fired"
    assert json.loads(jobs(conn)[0]["payload_json"])["event"]["type"] == "timer.visit_reminder"
    assert channel.jump_to_next_timer(conn) is None


def test_timer_for_opted_out_thread_is_suppressed(conn: sqlite3.Connection) -> None:
    _timer(conn, "visit_followup", clock.fmt_iso(clock.SIM_START + timedelta(hours=1)))
    conn.execute("UPDATE threads SET opted_out = 1")
    channel.advance_clock(conn, 2)
    channel.fire_due_timers(conn)
    timer = db.row(conn, "SELECT * FROM timers")
    assert timer is not None and (timer["status"], timer["note"]) == ("suppressed", "opted out")
    assert jobs(conn) == []


def test_inbound_validation() -> None:
    with pytest.raises(ValueError):
        InboundSMS(From="", To=BIZ, Body="x")
    with pytest.raises(ValueError):
        InboundSMS(From=PHONE, To=BIZ, Body="x" * 1601)
