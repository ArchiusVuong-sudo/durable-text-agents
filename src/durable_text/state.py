"""Read model for the UI: one JSON snapshot of the whole demo."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from . import clock, db
from .sms import segment_info


def _j(text: str | None, default: Any) -> Any:
    return json.loads(text) if text else default


def snapshot(conn: sqlite3.Connection, *, mode: str, mode_reason: str, model: str,
             fixtures: int | None = None) -> dict[str, Any]:
    thread = db.row(conn, "SELECT * FROM threads LIMIT 1")
    now = clock.get(conn)
    messages = [
        {
            "id": m["id"], "direction": m["direction"], "body": m["body"], "kind": m["kind"],
            "status": m["status"], "segments": m["segments"], "encoding": m["encoding"],
            "label": segment_info(m["body"]).label, "turn_id": m["turn_id"], "sim_at": m["sim_at"],
            "sid": m["sid"],
        }
        for m in db.rows(conn, "SELECT * FROM messages ORDER BY id")
    ]
    turns = []
    for t in db.rows(conn, "SELECT * FROM turns ORDER BY id DESC LIMIT 40"):
        llm_calls = _j(t["llm_json"], [])
        turns.append(
            {
                "id": t["id"], "job_id": t["job_id"], "kind": t["kind"], "attempt": t["attempt"],
                "status": t["status"], "pid": t["worker_pid"], "agent_before": t["agent_before"],
                "agent_after": t["agent_after"], "checkpoint_from": t["checkpoint_from"],
                "checkpoint_to": t["checkpoint_to"], "input": _j(t["input_json"], {}),
                "trace": _j(t["trace_json"], []), "error": t["error"], "sim_at": t["sim_at"],
                "started_ms": t["started_ms"], "ended_ms": t["ended_ms"],
                "llm": [
                    {k: c.get(k) for k in ("source", "model", "duration_ms", "recorded_ms", "input_tokens",
                                           "output_tokens", "cost_usd", "attempts", "agent_id")}
                    for c in llm_calls
                ],
                "cost_usd": round(sum(c.get("cost_usd") or 0 for c in llm_calls), 5),
            }
        )
    webhooks = [
        {"id": w["id"], "sid": w["sid"], "outcome": w["outcome"], "body": _j(w["form_json"], {}).get("Body"),
         "created_ms": w["created_ms"]}
        for w in db.rows(conn, "SELECT * FROM webhooks ORDER BY id DESC LIMIT 12")
    ]
    delegations = []
    for d in db.rows(conn, "SELECT d.*, j.attempts FROM delegations d LEFT JOIN jobs j ON j.id = d.job_id "
                           "ORDER BY d.created_ms DESC"):
        delegations.append(
            {"id": d["id"], "from": d["from_agent"], "to": d["to_agent"], "status": d["status"],
             "task": _j(d["task_json"], {}), "result": _j(d["result_json"], None),
             "attempts": d["attempts"]}
        )
    timers = [
        {"id": tm["id"], "kind": tm["kind"], "due": tm["due_sim"], "due_human": clock.human(clock.parse(tm["due_sim"])),
         "status": tm["status"], "note": tm["note"]}
        for tm in db.rows(conn, "SELECT * FROM timers ORDER BY due_sim, id")
    ]
    workers = [
        {"pid": w["pid"], "status": w["status"], "started_ms": w["started_ms"], "ended_ms": w["ended_ms"]}
        for w in db.rows(conn, "SELECT * FROM workers ORDER BY started_ms DESC LIMIT 6")
    ]
    jobs = [
        dict(j) for j in db.rows(
            conn, "SELECT id, kind, status, attempts, lease_owner, cancel_requested, run_after_ms "
                  "FROM jobs WHERE status IN ('queued', 'running') ORDER BY id")
    ]
    cp = None
    if thread is not None:
        r = db.row(conn, "SELECT * FROM checkpoints WHERE thread_id = ? AND version = ?",
                   thread["id"], thread["checkpoint_version"])
        if r is not None:
            items = _j(r["chat_ctx_json"], {}).get("items", [])
            cp = {"version": r["version"], "agent_id": r["agent_id"], "values": _j(r["values_json"], {}),
                  "items": len(items)}
    last = db.row(conn, "SELECT MAX(id) AS id FROM events")
    committed = [t for t in turns if t["status"] == "committed"]
    return {
        "mode": mode, "mode_reason": mode_reason, "model": model, "fixtures": fixtures,
        "event_id": last["id"] if last and last["id"] else 0,
        "clock": {"sim_now": clock.fmt_iso(now), "human": clock.human(now)},
        "thread": None if thread is None else {
            "id": thread["id"], "customer": _j(thread["customer_json"], {}), "opted_out": bool(thread["opted_out"]),
            "active_agent": thread["active_agent"], "checkpoint_version": thread["checkpoint_version"],
            "business_number": thread["business_number"],
        },
        "checkpoint": cp, "messages": messages, "turns": turns, "webhooks": webhooks,
        "delegations": delegations, "timers": timers, "workers": workers, "jobs": jobs,
        "typing": any(j["kind"] in ("turn", "wake") for j in jobs if j["status"] == "running"),
        "totals": {
            "turns": len(committed),
            "llm_calls": sum(len(t["llm"]) for t in turns),
            "cost_usd": round(sum(t["cost_usd"] for t in turns), 4),
        },
    }
