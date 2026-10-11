"""The SMS channel: carrier-style webhook in, outbox out, plus the operator actions.

Text rules live here, before any agent code runs:

* duplicate ``MessageSid`` (a carrier retry) is logged and ignored;
* STOP / START / HELP are answered from templates with no LLM call;
* an opted-out thread gets no outbound texts and its timers are suppressed;
* texts inside the coalesce window become one turn (debounce);
* a text that lands while a turn runs cancels that uncommitted turn, and the
  rerun sees every unread text.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import timedelta
from typing import Any

from pydantic import BaseModel, Field

from . import clock, db
from .sms import TEMPLATES, Keyword, keyword, segment_info


class InboundSMS(BaseModel):
    """Form fields of a carrier-style inbound webhook."""

    From: str = Field(min_length=3, max_length=32)
    To: str = Field(min_length=3, max_length=32)
    Body: str = Field(max_length=1600)
    MessageSid: str = Field(default_factory=lambda: "msg_" + secrets.token_hex(12), max_length=64)


class InboundResult(BaseModel):
    outcome: str
    detail: str = ""
    message_id: int | None = None


def receive_inbound(conn: sqlite3.Connection, sms: InboundSMS, coalesce_s: float) -> InboundResult:
    with db.tx(conn):
        result = _receive(conn, sms, coalesce_s)
        conn.execute(
            "INSERT INTO webhooks (sid, form_json, outcome, created_ms) VALUES (?, ?, ?, ?)",
            (sms.MessageSid, sms.model_dump_json(), result.outcome, db.now_ms()),
        )
        db.emit(conn, "inbound", outcome=result.outcome, sid=sms.MessageSid)
    return result


def _receive(conn: sqlite3.Connection, sms: InboundSMS, coalesce_s: float) -> InboundResult:
    if db.row(conn, "SELECT 1 FROM messages WHERE sid = ?", sms.MessageSid):
        return InboundResult(outcome="duplicate", detail="duplicate MessageSid ignored")
    thread = db.row(conn, "SELECT * FROM threads WHERE phone = ?", sms.From)
    if thread is None:
        return InboundResult(outcome="unknown_number", detail="no thread for this number")
    tid = thread["id"]
    kw = keyword(sms.Body)
    seg = segment_info(sms.Body)
    kind = "keyword" if kw else ("ignored" if thread["opted_out"] else "agent")
    cur = conn.execute(
        "INSERT INTO messages (thread_id, direction, sid, body, kind, status, segments, encoding, "
        "sim_at, created_ms) VALUES (?, 'in', ?, ?, ?, 'received', ?, ?, ?, ?)",
        (tid, sms.MessageSid, sms.Body, kind, seg.segments, seg.encoding,
         clock.fmt_iso(clock.get(conn)), db.now_ms()),
    )
    msg_id = int(cur.lastrowid or 0)
    if kw is not None:
        _handle_keyword(conn, tid, kw)
        return InboundResult(outcome=f"keyword:{kw.value}", detail="answered from template, no LLM call",
                             message_id=msg_id)
    if thread["opted_out"]:
        return InboundResult(outcome="opted_out", detail="thread is opted out: no reply",
                             message_id=msg_id)
    detail = _schedule_turn(conn, tid, coalesce_s)
    return InboundResult(outcome="accepted", detail=detail, message_id=msg_id)


def _schedule_turn(conn: sqlite3.Connection, thread_id: str, coalesce_s: float) -> str:
    run_after = db.now_ms() + int(coalesce_s * 1000)
    notes = []
    running = db.row(
        conn,
        "SELECT id FROM jobs WHERE thread_id = ? AND kind = 'turn' AND status = 'running'",
        thread_id,
    )
    if running is not None:
        conn.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (running["id"],))
        notes.append(f"supersedes running job {running['id']}")
    queued = db.row(
        conn,
        "SELECT id FROM jobs WHERE thread_id = ? AND kind = 'turn' AND status = 'queued'",
        thread_id,
    )
    if queued is not None:
        conn.execute("UPDATE jobs SET run_after_ms = ? WHERE id = ?", (run_after, queued["id"]))
        notes.append(f"coalesced into job {queued['id']}")
    else:
        conn.execute(
            "INSERT INTO jobs (kind, thread_id, run_after_ms, created_ms) VALUES ('turn', ?, ?, ?)",
            (thread_id, run_after, db.now_ms()),
        )
        notes.append(f"turn queued, waits {coalesce_s:g} s for more texts")
    return "; ".join(notes)


def _handle_keyword(conn: sqlite3.Connection, thread_id: str, kw: Keyword) -> None:
    if kw is Keyword.STOP:
        conn.execute("UPDATE threads SET opted_out = 1 WHERE id = ?", (thread_id,))
        # Nothing may reach an opted-out number: drop queued turns, cancel a running one.
        conn.execute(
            "UPDATE jobs SET status = 'cancelled', finished_ms = ? "
            "WHERE thread_id = ? AND kind = 'turn' AND status = 'queued'",
            (db.now_ms(), thread_id),
        )
        conn.execute(
            "UPDATE jobs SET cancel_requested = 1 WHERE thread_id = ? AND kind = 'turn' AND status = 'running'",
            (thread_id,),
        )
    elif kw is Keyword.START:
        conn.execute("UPDATE threads SET opted_out = 0 WHERE id = ?", (thread_id,))
    body = TEMPLATES[kw]
    seg = segment_info(body)
    conn.execute(
        "INSERT INTO messages (thread_id, direction, body, kind, status, segments, encoding, sim_at, "
        "created_ms) VALUES (?, 'out', ?, 'template', 'queued', ?, ?, ?, ?)",
        (thread_id, body, seg.segments, seg.encoding, clock.fmt_iso(clock.get(conn)), db.now_ms()),
    )


def replay_last_webhook(conn: sqlite3.Connection, coalesce_s: float) -> InboundResult | None:
    """Send the newest webhook again with the same MessageSid, as a carrier retry would."""
    last = db.row(conn, "SELECT form_json FROM webhooks WHERE outcome != 'duplicate' ORDER BY id DESC LIMIT 1")
    if last is None:
        return None
    return receive_inbound(conn, InboundSMS.model_validate_json(last["form_json"]), coalesce_s)


# ---------- fake carrier ----------


def deliver_outbox(conn: sqlite3.Connection, min_age_ms: int = 300) -> int:
    """Fake carrier: mark queued outbound texts delivered. Never sends a real SMS."""
    with db.tx(conn):
        pending = db.rows(
            conn,
            "SELECT m.id, m.kind, t.opted_out FROM messages m JOIN threads t ON t.id = m.thread_id "
            "WHERE m.direction = 'out' AND m.status = 'queued' AND m.created_ms <= ?",
            db.now_ms() - min_age_ms,
        )
        for m in pending:
            status = "suppressed" if m["opted_out"] and m["kind"] != "template" else "delivered"
            conn.execute("UPDATE messages SET status = ? WHERE id = ?", (status, m["id"]))
        if pending:
            db.emit(conn, "outbox.delivered", count=len(pending))
    return len(pending)


# ---------- dispatcher approval ----------


class ApprovalError(Exception):
    pass


def approve(conn: sqlite3.Connection, delegation_id: str) -> dict[str, Any]:
    """Human approval: book the slot and wake the customer thread with no inbound text."""
    with db.tx(conn):
        d = db.row(conn, "SELECT * FROM delegations WHERE id = ?", delegation_id)
        if d is None:
            raise ApprovalError("no such delegation")
        if d["status"] != "needs_approval":
            raise ApprovalError(f"delegation is {d['status']}, not waiting for approval")
        proposal = json.loads(d["result_json"])
        task = json.loads(d["task_json"])
        conn.execute(
            "UPDATE delegations SET status = 'approved', decided_ms = ? WHERE id = ?",
            (db.now_ms(), delegation_id),
        )
        conn.execute(
            "UPDATE slots SET status = 'booked', thread_id = ?, technician_id = ? WHERE id = ?",
            (d["thread_id"], proposal["technician_id"], proposal["slot_id"]),
        )
        event = {
            "type": "dispatch.approved",
            "data": {
                "delegation_id": delegation_id,
                "technician_name": proposal["technician_name"],
                "slot_id": proposal["slot_id"],
                "window": proposal["window"],
                "start": task["start"],
                "end": task["end"],
            },
        }
        cur = conn.execute(
            "INSERT INTO jobs (kind, thread_id, payload_json, run_after_ms, created_ms) "
            "VALUES ('wake', ?, ?, ?, ?)",
            (d["thread_id"], json.dumps({"event": event}), db.now_ms(), db.now_ms()),
        )
        db.emit(conn, "delegation.approved", id=delegation_id, wake_job=cur.lastrowid)
    return event


# ---------- simulated clock and durable timers ----------


def advance_clock(conn: sqlite3.Connection, hours: float) -> str:
    with db.tx(conn):
        new = clock.advance(conn, timedelta(hours=hours))
        db.emit(conn, "clock", sim_now=clock.fmt_iso(new))
    return clock.fmt_iso(new)


def jump_to_next_timer(conn: sqlite3.Connection) -> str | None:
    with db.tx(conn):
        nxt = db.row(conn, "SELECT MIN(due_sim) AS due FROM timers WHERE status = 'pending'")
        if nxt is None or nxt["due"] is None:
            return None
        due = clock.parse(nxt["due"])
        if due > clock.get(conn):
            clock.set_(conn, due)
        db.emit(conn, "clock", sim_now=clock.fmt_iso(clock.get(conn)))
        return clock.fmt_iso(clock.get(conn))


def fire_due_timers(conn: sqlite3.Connection) -> int:
    """Turn due timers into wake jobs, or mark them suppressed for opted-out threads."""
    with db.tx(conn):
        now = clock.fmt_iso(clock.get(conn))
        due = db.rows(
            conn,
            "SELECT tm.*, t.opted_out FROM timers tm JOIN threads t ON t.id = tm.thread_id "
            "WHERE tm.status = 'pending' AND tm.due_sim <= ? ORDER BY tm.due_sim, tm.id",
            now,
        )
        for tm in due:
            if tm["opted_out"]:
                conn.execute(
                    "UPDATE timers SET status = 'suppressed', note = 'opted out' WHERE id = ?", (tm["id"],)
                )
                continue
            event = {"type": f"timer.{tm['kind']}", "data": json.loads(tm["payload_json"])}
            cur = conn.execute(
                "INSERT INTO jobs (kind, thread_id, payload_json, run_after_ms, created_ms) "
                "VALUES ('wake', ?, ?, ?, ?)",
                (tm["thread_id"], json.dumps({"event": event}), db.now_ms(), db.now_ms()),
            )
            conn.execute(
                "UPDATE timers SET status = 'fired', fired_job = ? WHERE id = ?", (cur.lastrowid, tm["id"])
            )
        if due:
            db.emit(conn, "timers.fired", count=len(due))
    return len(due)
