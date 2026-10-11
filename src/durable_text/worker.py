"""The agent worker: a separate process that claims jobs from SQLite and runs turns.

Nothing a turn needs lives only in this process. Jobs carry a lease (owner pid +
expiry, renewed by a heartbeat). When a worker dies, its jobs are reclaimed by
the next one at once (the dead pid is known) or when the lease expires.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sqlite3
import sys
from collections.abc import Callable

from . import channel, db
from .config import Settings
from .envsafe import scrub_process_env
from .llm.base import CompletionLLM
from .llm.factory import Mode, make_llm
from .runtime import Job, JobRun, Runtime

log = logging.getLogger("durable_text.worker")

CLAIM_SQL = """
SELECT j.* FROM jobs j
WHERE (
    (j.status = 'queued' AND j.run_after_ms <= :now)
    OR (j.status = 'running' AND j.lease_owner != :pid AND (
        j.lease_expires_ms < :now
        OR j.lease_owner IN (SELECT pid FROM workers WHERE status != 'running')))
)
AND (j.kind = 'dispatch' OR NOT EXISTS (
    SELECT 1 FROM jobs o
    WHERE o.thread_id = j.thread_id AND o.kind IN ('turn', 'wake') AND o.id != j.id
      AND o.status = 'running' AND o.lease_expires_ms >= :now
      AND o.lease_owner NOT IN (SELECT pid FROM workers WHERE status != 'running')))
AND j.id NOT IN (SELECT value FROM json_each(:mine))
ORDER BY j.id
LIMIT 1
"""


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Worker:
    def __init__(
        self, settings: Settings, mode: Mode, llm_factory: Callable[[], CompletionLLM] | None = None
    ) -> None:
        self.settings = settings
        self.mode = mode
        self.pid = os.getpid()
        self.conn = db.connect(settings.db_path)
        factory = llm_factory or (lambda: make_llm(mode, settings))
        self.runtime = Runtime(settings.db_path, self.pid, factory)
        self.running: dict[int, tuple[asyncio.Task[str], JobRun]] = {}
        self.stopping = False

    def register(self) -> None:
        with db.tx(self.conn):
            for r in db.rows(self.conn, "SELECT pid FROM workers WHERE status = 'running'"):
                if r["pid"] != self.pid and not pid_alive(r["pid"]):
                    self.conn.execute(
                        "UPDATE workers SET status = 'dead', ended_ms = ? WHERE pid = ?",
                        (db.now_ms(), r["pid"]),
                    )
            self.conn.execute(
                "INSERT INTO workers (pid, status, started_ms, last_seen_ms) VALUES (?, 'running', ?, ?) "
                "ON CONFLICT(pid) DO UPDATE SET status = 'running', last_seen_ms = excluded.last_seen_ms",
                (self.pid, db.now_ms(), db.now_ms()),
            )
            db.emit(self.conn, "worker.started", pid=self.pid, mode=self.mode)

    def claim(self) -> JobRun | None:
        now = db.now_ms()
        params = {"now": now, "pid": self.pid, "mine": json.dumps(list(self.running))}
        with db.tx(self.conn):
            r = self.conn.execute(CLAIM_SQL, params).fetchone()
            if r is None:
                return None
            reclaimed = r["status"] == "running"
            self.conn.execute(
                "UPDATE jobs SET status = 'running', lease_owner = ?, lease_expires_ms = ?, "
                "attempts = attempts + 1 WHERE id = ?",
                (self.pid, now + int(self.settings.lease_s * 1000), r["id"]),
            )
            if reclaimed:
                self.conn.execute(
                    "UPDATE turns SET status = 'interrupted', error = ?, ended_ms = ? "
                    "WHERE job_id = ? AND status = 'running'",
                    (f"worker pid {r['lease_owner']} died before commit", now, r["id"]),
                )
            db.emit(self.conn, "job.claimed", job_id=r["id"], pid=self.pid, reclaimed=reclaimed)
            fresh = db.row(self.conn, "SELECT * FROM jobs WHERE id = ?", r["id"])
        assert fresh is not None
        return JobRun(job=Job.from_row(fresh))

    async def tick(self) -> None:
        channel.fire_due_timers(self.conn)
        while len(self.running) < self.settings.max_concurrent_jobs and not self.stopping:
            run = self.claim()
            if run is None:
                break
            task = asyncio.create_task(self.runtime.run_job(run), name=f"job-{run.job.id}")
            self.running[run.job.id] = (task, run)
            task.add_done_callback(lambda _t, jid=run.job.id: self.running.pop(jid, None))

    def watch_cancels_and_renew(self) -> None:
        if not self.running:
            self._heartbeat()
            return
        ids = list(self.running)
        marks = ",".join("?" * len(ids))
        with db.tx(self.conn):
            self.conn.execute(
                f"UPDATE jobs SET lease_expires_ms = ? WHERE id IN ({marks}) AND lease_owner = ?",
                (db.now_ms() + int(self.settings.lease_s * 1000), *ids, self.pid),
            )
            self.conn.execute("UPDATE workers SET last_seen_ms = ? WHERE pid = ?", (db.now_ms(), self.pid))
            cancels = db.rows(
                self.conn, f"SELECT id FROM jobs WHERE id IN ({marks}) AND cancel_requested = 1", *ids
            )
        for r in cancels:
            entry = self.running.get(r["id"])
            if entry and not entry[0].done() and entry[1].cancel_reason is None:
                entry[1].cancel_reason = "superseded"
                entry[0].cancel()

    def _heartbeat(self) -> None:
        with contextlib.suppress(sqlite3.Error), db.tx(self.conn):
            self.conn.execute("UPDATE workers SET last_seen_ms = ? WHERE pid = ?", (db.now_ms(), self.pid))

    async def serve(self, poll_s: float = 0.25) -> None:
        self.register()
        while not self.stopping:
            try:
                self.watch_cancels_and_renew()
                await self.tick()
            except sqlite3.OperationalError:
                log.warning("db busy, retrying", exc_info=True)
            await asyncio.sleep(poll_s)
        await self.shutdown()

    async def shutdown(self) -> None:
        for task, run in list(self.running.values()):
            run.cancel_reason = run.cancel_reason or "shutdown"
            task.cancel()
        if self.running:
            await asyncio.gather(*(t for t, _ in self.running.values()), return_exceptions=True)
        with contextlib.suppress(sqlite3.Error), db.tx(self.conn):
            self.conn.execute(
                "UPDATE workers SET status = 'exited', ended_ms = ? WHERE pid = ?", (db.now_ms(), self.pid)
            )
            db.emit(self.conn, "worker.exited", pid=self.pid)
        self.conn.close()


def main() -> None:
    settings = Settings.from_env()
    mode: Mode = os.environ.get("DTA_MODE", "replay")  # type: ignore[assignment]
    removed = scrub_process_env()  # the CLI child inherits only the allowlist
    logging.basicConfig(level=logging.INFO, format="%(asctime)s worker %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)
    log.info("worker pid %s mode %s model %s (env scrubbed: %d vars)", os.getpid(), mode,
             settings.model, len(removed))
    worker = Worker(settings, mode)
    loop = asyncio.new_event_loop()

    def stop(*_: object) -> None:
        worker.stopping = True

    loop.add_signal_handler(signal.SIGTERM, stop)
    loop.add_signal_handler(signal.SIGINT, stop)
    loop.run_until_complete(worker.serve())


if __name__ == "__main__":
    main()
