"""Transactional turns.

One job = one turn. A turn loads the last checkpoint, rebuilds a LiveKit
``AgentSession`` with the active agent and its ``ChatContext``, runs it, and
commits everything in one transaction: the new checkpoint, the outbound texts
(outbox), slot holds, delegations and timers. The commit is fenced on the job
lease and on ``cancel_requested``, so a superseded or orphaned turn writes
nothing but its own "superseded" or "interrupted" status.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from livekit.agents import AgentSession, llm
from livekit.agents.voice.run_result import RunResult

from . import agents as _agents  # noqa: F401 - registers the agents
from . import clock, db
from .framework import DurableAgent, Effects, TurnState, WakeEvent, agent_class
from .llm.base import CompletionLLM
from .sms import segment_info, to_plain_text

log = logging.getLogger(__name__)

CancelReason = Literal["superseded", "shutdown"]
MAX_ATTEMPTS = 3


@dataclass
class Job:
    id: int
    kind: str
    thread_id: str
    payload: dict[str, Any]
    attempts: int

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> Job:
        return cls(r["id"], r["kind"], r["thread_id"], json.loads(r["payload_json"]), r["attempts"])


@dataclass
class TurnOutput:
    agent_after: str
    chat_ctx: dict[str, Any]
    texts: list[str]
    trace: list[dict[str, Any]]
    effects: Effects
    values: dict[str, Any]


@dataclass
class JobRun:
    job: Job
    turn_id: int | None = None
    cancel_reason: CancelReason | None = None
    llm: CompletionLLM | None = None
    progress: list[dict[str, Any]] = field(default_factory=list)


class LeaseLost(Exception):
    """The job is no longer ours (reclaimed by another worker); write nothing."""


class Superseded(Exception):
    """A newer text asked to cancel this turn before it committed."""


def trace_from_run(result: RunResult[Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for ev in result.events:
        if ev.type == "message":
            if ev.item.role == "assistant" and (ev.item.text_content or "").strip():
                steps.append({"t": "message", "text": ev.item.text_content})
        elif ev.type == "function_call":
            steps.append({"t": "call", "name": ev.item.name, "args": _json(ev.item.arguments)})
        elif ev.type == "function_call_output":
            steps.append(
                {
                    "t": "output",
                    "name": ev.item.name,
                    "output": ev.item.output[:600],
                    "error": bool(ev.item.is_error),
                }
            )
        elif ev.type == "agent_handoff" and ev.old_agent is not None:
            steps.append(
                {"t": "handoff", "from": ev.old_agent.id if ev.old_agent else None, "to": ev.new_agent.id}
            )
    return steps


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text


class Runtime:
    def __init__(self, db_path: Path, pid: int, llm_factory: Callable[[], CompletionLLM]) -> None:
        self._db_path = db_path
        self._pid = pid
        self._llm_factory = llm_factory

    # ---------- entry point ----------

    async def run_job(self, run: JobRun) -> str:
        conn = db.connect(self._db_path)
        try:
            if run.job.kind == "turn":
                return await self._customer_turn(conn, run, wake=None)
            if run.job.kind == "wake":
                return await self._customer_turn(conn, run, wake=WakeEvent(**run.job.payload["event"]))
            if run.job.kind == "dispatch":
                return await self._dispatch(conn, run)
            raise ValueError(f"unknown job kind {run.job.kind}")
        except (asyncio.CancelledError, Superseded):
            self._finish_cancelled(conn, run)
            return run.cancel_reason or "cancelled"
        except LeaseLost:
            self._close_turn(conn, run, "interrupted", "lease lost before commit")
            return "lease-lost"
        except Exception as exc:  # the job fails, the worker keeps running
            log.exception("job %s failed", run.job.id)
            self._finish_failed(conn, run, exc)
            return "failed"
        finally:
            conn.close()

    # ---------- customer-facing turns (inbound text or wake event) ----------

    async def _customer_turn(self, conn: sqlite3.Connection, run: JobRun, wake: WakeEvent | None) -> str:
        job = run.job
        thread = db.row(conn, "SELECT * FROM threads WHERE id = ?", job.thread_id)
        assert thread is not None
        inputs: list[sqlite3.Row] = []
        if wake is None:
            inputs = db.rows(
                conn,
                "SELECT * FROM messages WHERE thread_id = ? AND direction = 'in' AND kind = 'agent' "
                "AND consumed_by_turn IS NULL ORDER BY id",
                job.thread_id,
            )
            if not inputs:
                self._finish_job(conn, job, "done")
                return "nothing-to-do"
        version = thread["checkpoint_version"]
        cp = db.row(
            conn, "SELECT * FROM checkpoints WHERE thread_id = ? AND version = ?", job.thread_id, version
        )
        agent_id = cp["agent_id"] if cp else thread["active_agent"]
        chat_ctx = llm.ChatContext.from_dict(json.loads(cp["chat_ctx_json"])) if cp else None
        values = json.loads(cp["values_json"]) if cp else {}
        texts_in = [r["body"] for r in inputs]
        input_info: dict[str, Any] = (
            {"message_ids": [r["id"] for r in inputs], "texts": texts_in, "coalesced": len(inputs)}
            if wake is None
            else {"event": wake.model_dump()}
        )
        run.turn_id = self._open_turn(conn, run, agent_id, version, input_info)
        state = self._state(conn, job.thread_id, thread, values)
        agent = agent_class(agent_id)(state, chat_ctx=chat_ctx, resumed=True, wake=wake)
        out = await self._run_session(
            run, state, agent, user_input="\n".join(texts_in) if wake is None else None
        )
        with db.tx(conn):
            self._fence(conn, job)
            t = db.row(conn, "SELECT * FROM threads WHERE id = ?", job.thread_id)
            assert t is not None
            if t["checkpoint_version"] != version:
                raise LeaseLost()
            new_version = version + 1
            conn.execute(
                "INSERT INTO checkpoints (thread_id, version, agent_id, chat_ctx_json, values_json, "
                "turn_id, created_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    job.thread_id,
                    new_version,
                    out.agent_after,
                    json.dumps(out.chat_ctx),
                    json.dumps(out.values),
                    run.turn_id,
                    db.now_ms(),
                ),
            )
            conn.execute(
                "UPDATE threads SET active_agent = ?, checkpoint_version = ? WHERE id = ?",
                (out.agent_after, new_version, job.thread_id),
            )
            if inputs:
                conn.execute(
                    f"UPDATE messages SET consumed_by_turn = ? WHERE id IN ({','.join('?' * len(inputs))})",
                    (run.turn_id, *[r["id"] for r in inputs]),
                )
            sim_now = clock.fmt_iso(clock.get(conn))
            status = "suppressed" if t["opted_out"] else "queued"
            for text in out.texts:
                body = to_plain_text(text)
                seg = segment_info(body)
                conn.execute(
                    "INSERT INTO messages (thread_id, direction, body, kind, status, segments, encoding, "
                    "turn_id, sim_at, created_ms) VALUES (?, 'out', ?, 'agent', ?, ?, ?, ?, ?, ?)",
                    (
                        job.thread_id,
                        body,
                        status,
                        seg.segments,
                        seg.encoding,
                        run.turn_id,
                        sim_now,
                        db.now_ms(),
                    ),
                )
            trace = self._apply_effects(conn, job.thread_id, run.turn_id, out)
            self._close_turn_in_tx(conn, run, "committed", None, out.agent_after, new_version, trace)
            self._finish_job_in_tx(conn, job, "done")
            db.emit(conn, "turn.committed", turn_id=run.turn_id, thread_id=job.thread_id)
        return "committed"

    def _apply_effects(
        self, conn: sqlite3.Connection, thread_id: str, turn_id: int | None, out: TurnOutput
    ) -> list[dict[str, Any]]:
        trace = list(out.trace)
        eff = out.effects
        if eff.holds:
            conn.execute(
                "UPDATE slots SET status = 'open', thread_id = NULL WHERE thread_id = ? AND status = 'held'",
                (thread_id,),
            )
            for slot_id in eff.holds:
                conn.execute(
                    "UPDATE slots SET status = 'held', thread_id = ? WHERE id = ? AND status = 'open'",
                    (thread_id, slot_id),
                )
                trace.append({"t": "hold", "slot": slot_id})
        for d in eff.delegations:
            cur = conn.execute(
                "INSERT INTO jobs (kind, thread_id, payload_json, run_after_ms, created_ms) "
                "VALUES ('dispatch', ?, ?, ?, ?)",
                (thread_id, json.dumps({"delegation_id": d.id}), db.now_ms(), db.now_ms()),
            )
            conn.execute(
                "INSERT INTO delegations (id, thread_id, from_agent, to_agent, task_json, status, job_id, "
                "created_ms) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)",
                (d.id, thread_id, out.agent_after, d.to, json.dumps(d.task), cur.lastrowid, db.now_ms()),
            )
            trace.append({"t": "delegation", "id": d.id, "to": d.to})
        for timer in eff.timers:
            conn.execute(
                "INSERT INTO timers (thread_id, kind, due_sim, payload_json, created_by_turn) "
                "VALUES (?, ?, ?, ?, ?)",
                (thread_id, timer.kind, timer.due, json.dumps(timer.payload), turn_id),
            )
            trace.append({"t": "timer", "kind": timer.kind, "due": timer.due})
        return trace

    # ---------- headless delegation ----------

    async def _dispatch(self, conn: sqlite3.Connection, run: JobRun) -> str:
        job = run.job
        delegation_id = job.payload["delegation_id"]
        d = db.row(conn, "SELECT * FROM delegations WHERE id = ?", delegation_id)
        thread = db.row(conn, "SELECT * FROM threads WHERE id = ?", job.thread_id)
        assert d is not None and thread is not None
        task = json.loads(d["task_json"])
        run.turn_id = self._open_turn(
            conn,
            run,
            d["to_agent"],
            None,
            {"delegation_id": delegation_id, "from": d["from_agent"], "task": task},
        )
        with db.tx(conn):
            conn.execute("UPDATE delegations SET status = 'running' WHERE id = ?", (delegation_id,))
            db.emit(conn, "delegation.running", id=delegation_id, attempt=job.attempts)
        state = self._state(conn, job.thread_id, thread, {"task": task})
        agent = agent_class(d["to_agent"])(state, resumed=True)
        brief = (
            f"New task from {d['from_agent']}: pick a {task['skill']} technician in the {task['zone']} "
            f"zone for {task['window']} ({task['slot_id']}). Issue: {task.get('issue', '')}"
        )
        out = await self._run_session(run, state, agent, user_input=brief)
        proposal = out.effects.proposal
        with db.tx(conn):
            self._fence(conn, job)
            if proposal is None:
                conn.execute(
                    "UPDATE delegations SET status = 'failed', result_json = ? WHERE id = ?",
                    (json.dumps({"error": "no proposal"}), delegation_id),
                )
            else:
                conn.execute(
                    "UPDATE delegations SET status = 'needs_approval', result_json = ? WHERE id = ?",
                    (proposal.model_dump_json(), delegation_id),
                )
            trace = out.trace + ([{"t": "proposal", **proposal.model_dump()}] if proposal else [])
            self._close_turn_in_tx(conn, run, "committed", None, d["to_agent"], None, trace)
            self._finish_job_in_tx(conn, job, "done")
            db.emit(conn, "delegation.proposed", id=delegation_id)
        return "committed"

    # ---------- session plumbing ----------

    def _state(
        self, conn: sqlite3.Connection, thread_id: str, thread: sqlite3.Row, values: dict[str, Any]
    ) -> TurnState:
        seq = db.row(conn, "SELECT COUNT(*) AS n FROM delegations")
        return TurnState(
            thread_id=thread_id,
            conn=conn,
            now=clock.get(conn),
            customer=json.loads(thread["customer_json"]),
            values=dict(values),
            delegation_seq=int(seq["n"]) if seq else 0,
        )

    async def _run_session(
        self, run: JobRun, state: TurnState, agent: DurableAgent, *, user_input: str | None
    ) -> TurnOutput:
        model = self._llm_factory()
        run.llm = model
        conn = state.conn
        async with AgentSession[TurnState](llm=model, userdata=state) as session:

            def on_tools(ev: Any) -> None:
                for call, output in zip(ev.function_calls, ev.function_call_outputs, strict=False):
                    run.progress.append({"t": "call", "name": call.name, "args": _json(call.arguments)})
                    if output is not None:
                        run.progress.append(
                            {
                                "t": "output",
                                "name": output.name,
                                "output": output.output[:600],
                                "error": bool(output.is_error),
                            }
                        )
                self._write_progress(conn, run)

            def on_handoff(ev: Any) -> None:
                item = ev.item
                if getattr(item, "type", None) == "agent_handoff" and item.old_agent_id:
                    run.progress.append({"t": "handoff", "from": item.old_agent_id, "to": item.new_agent_id})
                    self._write_progress(conn, run)

            session.on("function_tools_executed", on_tools)
            session.on("conversation_item_added", on_handoff)
            if user_input is None:
                result = await session.start(agent, capture_run=True)
                await result
            else:
                await session.start(agent)
                result = await session.run(user_input=user_input)
            current = session.current_agent
            trace = trace_from_run(result)
            texts = [s["text"] for s in trace if s["t"] == "message"]
            if isinstance(current, DurableAgent) and current.headless:
                texts = []
            return TurnOutput(
                agent_after=current.id,
                chat_ctx=current.chat_ctx.to_dict(exclude_timestamp=False),
                texts=texts,
                trace=trace,
                effects=state.effects,
                values=state.values,
            )

    # ---------- bookkeeping ----------

    def _open_turn(
        self,
        conn: sqlite3.Connection,
        run: JobRun,
        agent_id: str,
        version: int | None,
        input_info: dict[str, Any],
    ) -> int:
        with db.tx(conn):
            conn.execute(
                "UPDATE turns SET status = 'interrupted', error = 'worker stopped before commit', "
                "ended_ms = ? WHERE job_id = ? AND status = 'running'",
                (db.now_ms(), run.job.id),
            )
            cur = conn.execute(
                "INSERT INTO turns (job_id, thread_id, kind, attempt, status, worker_pid, agent_before, "
                "checkpoint_from, input_json, sim_at, started_ms) "
                "VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?)",
                (
                    run.job.id,
                    run.job.thread_id,
                    run.job.kind,
                    run.job.attempts,
                    self._pid,
                    agent_id,
                    version,
                    json.dumps(input_info),
                    clock.fmt_iso(clock.get(conn)),
                    db.now_ms(),
                ),
            )
            turn_id = int(cur.lastrowid or 0)
            db.emit(conn, "turn.started", turn_id=turn_id, kind=run.job.kind, pid=self._pid)
        return turn_id

    def _write_progress(self, conn: sqlite3.Connection, run: JobRun) -> None:
        if run.turn_id is None:
            return
        try:
            with db.tx(conn):
                conn.execute(
                    "UPDATE turns SET trace_json = ?, llm_json = ? WHERE id = ? AND status = 'running'",
                    (json.dumps(run.progress), self._llm_json(run), run.turn_id),
                )
                db.emit(conn, "turn.progress", turn_id=run.turn_id)
        except sqlite3.Error:
            log.warning("progress write skipped", exc_info=True)

    def _llm_json(self, run: JobRun) -> str:
        calls = run.llm.calls if run.llm else []
        return json.dumps([c.model_dump() for c in calls])

    def _fence(self, conn: sqlite3.Connection, job: Job) -> None:
        r = db.row(conn, "SELECT status, lease_owner, cancel_requested FROM jobs WHERE id = ?", job.id)
        if r is None or r["status"] != "running" or r["lease_owner"] != self._pid:
            raise LeaseLost()
        if r["cancel_requested"]:
            raise Superseded()

    def _close_turn_in_tx(
        self,
        conn: sqlite3.Connection,
        run: JobRun,
        status: str,
        error: str | None,
        agent_after: str | None = None,
        checkpoint_to: int | None = None,
        trace: list[dict[str, Any]] | None = None,
    ) -> None:
        if run.turn_id is None:
            return
        conn.execute(
            "UPDATE turns SET status = ?, error = ?, agent_after = COALESCE(?, agent_after), "
            "checkpoint_to = ?, trace_json = ?, llm_json = ?, ended_ms = ? WHERE id = ?",
            (
                status,
                error,
                agent_after,
                checkpoint_to,
                json.dumps(trace if trace is not None else run.progress),
                self._llm_json(run),
                db.now_ms(),
                run.turn_id,
            ),
        )

    def _close_turn(self, conn: sqlite3.Connection, run: JobRun, status: str, error: str | None) -> None:
        with db.tx(conn):
            self._close_turn_in_tx(conn, run, status, error)
            db.emit(conn, f"turn.{status}", turn_id=run.turn_id)

    def _finish_job_in_tx(self, conn: sqlite3.Connection, job: Job, status: str) -> None:
        conn.execute(
            "UPDATE jobs SET status = ?, finished_ms = ?, lease_owner = NULL "
            "WHERE id = ? AND lease_owner = ?",
            (status, db.now_ms(), job.id, self._pid),
        )

    def _finish_job(self, conn: sqlite3.Connection, job: Job, status: str) -> None:
        with db.tx(conn):
            self._finish_job_in_tx(conn, job, status)
            db.emit(conn, "job.finished", job_id=job.id, status=status)

    def _finish_cancelled(self, conn: sqlite3.Connection, run: JobRun) -> None:
        reason = run.cancel_reason or "superseded"
        with db.tx(conn):
            if reason == "shutdown":
                self._close_turn_in_tx(conn, run, "interrupted", "worker shut down before commit")
                conn.execute(
                    "UPDATE jobs SET status = 'queued', lease_owner = NULL, lease_expires_ms = NULL "
                    "WHERE id = ? AND lease_owner = ?",
                    (run.job.id, self._pid),
                )
            else:
                self._close_turn_in_tx(conn, run, "superseded", "a newer text arrived; nothing was sent")
                self._finish_job_in_tx(conn, run.job, "superseded")
            db.emit(conn, "turn.cancelled", turn_id=run.turn_id, reason=reason)

    def _finish_failed(self, conn: sqlite3.Connection, run: JobRun, exc: Exception) -> None:
        with db.tx(conn):
            self._close_turn_in_tx(conn, run, "failed", f"{type(exc).__name__}: {exc}"[:500])
            if run.job.attempts < MAX_ATTEMPTS:
                conn.execute(
                    "UPDATE jobs SET status = 'queued', lease_owner = NULL, run_after_ms = ? "
                    "WHERE id = ? AND lease_owner = ?",
                    (db.now_ms() + 2000 * run.job.attempts, run.job.id, self._pid),
                )
            else:
                self._finish_job_in_tx(conn, run.job, "failed")
                if run.job.kind == "dispatch":
                    conn.execute(
                        "UPDATE delegations SET status = 'failed' WHERE id = ?",
                        (run.job.payload.get("delegation_id"),),
                    )
            db.emit(conn, "turn.failed", turn_id=run.turn_id)
