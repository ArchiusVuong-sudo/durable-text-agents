"""Starts the worker as its own process group and restarts it when it dies."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
from pathlib import Path

from . import db
from .config import Settings
from .envsafe import child_env
from .llm.factory import Mode

log = logging.getLogger(__name__)


class Supervisor:
    def __init__(self, settings: Settings, mode: Mode, log_path: Path | None = None) -> None:
        self.settings = settings
        self.mode = mode
        self.log_path = log_path or settings.db_path.with_name("worker.log")
        self.proc: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._killed: set[int] = set()

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc and self.proc.returncode is None else None

    async def start(self) -> None:
        self._stopping = False
        self._task = asyncio.create_task(self._loop(), name="worker-supervisor")
        for _ in range(100):  # wait until the first worker is up
            if self.pid:
                return
            await asyncio.sleep(0.05)

    async def _spawn(self) -> asyncio.subprocess.Process:
        env = child_env() | self.settings.worker_env(self.mode)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = self.log_path.open("ab")
        try:
            return await asyncio.create_subprocess_exec(
                sys.executable, "-m", "durable_text.worker",
                env=env, stdin=asyncio.subprocess.DEVNULL, stdout=logf, stderr=logf,
                start_new_session=True,  # own process group: Kill reaches its CLI children too
            )
        finally:
            logf.close()

    async def _loop(self) -> None:
        conn = db.connect(self.settings.db_path)
        try:
            while not self._stopping:
                proc = await self._spawn()
                self.proc = proc
                with db.tx(conn):
                    conn.execute(
                        "INSERT INTO workers (pid, status, started_ms, last_seen_ms) "
                        "VALUES (?, 'running', ?, ?) ON CONFLICT(pid) DO UPDATE SET status = 'running'",
                        (proc.pid, db.now_ms(), db.now_ms()),
                    )
                    db.emit(conn, "worker.spawned", pid=proc.pid)
                rc = await proc.wait()
                if proc.pid in self._killed:
                    status = "killed"
                elif self._stopping:
                    status = "stopped"
                else:
                    status = "crashed" if rc != 0 else "exited"
                with db.tx(conn):
                    conn.execute(
                        "UPDATE workers SET status = ?, ended_ms = ? WHERE pid = ?",
                        (status, db.now_ms(), proc.pid),
                    )
                    db.emit(conn, "worker.ended", pid=proc.pid, status=status, rc=rc)
                log.info("worker %s %s (rc=%s)", proc.pid, status, rc)
                if not self._stopping:
                    await asyncio.sleep(0.4)
        finally:
            conn.close()

    def kill(self) -> int | None:
        """SIGKILL the worker's whole process group, as a crash would."""
        pid = self.pid
        if pid is None:
            return None
        self._killed.add(pid)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)
        return pid

    async def stop(self, timeout: float = 15) -> None:
        self._stopping = True
        pid = self.pid
        if pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGTERM)
            assert self.proc is not None
            try:
                await asyncio.wait_for(self.proc.wait(), timeout)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGKILL)
                await self.proc.wait()
        if self._task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
