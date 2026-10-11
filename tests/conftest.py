from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from durable_text import channel, db, seed
from durable_text.config import Settings
from durable_text.framework import TurnState
from durable_text.llm.base import CompletionLLM, LLMCall, LLMRequest
from durable_text.llm.scripted import ScriptedLLM
from durable_text.worker import Worker


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    base = Settings.from_env()
    return replace(
        base,
        db_path=tmp_path / "test.db",
        coalesce_s=0.0,
        record_path=None,
        replay_forced=True,
        replay_scale=0.0,
    )


@pytest.fixture
def conn(settings: Settings) -> Iterator[sqlite3.Connection]:
    c = db.connect(settings.db_path)
    seed.seed(c)
    yield c
    c.close()


@pytest.fixture
def state(conn: sqlite3.Connection) -> TurnState:
    thread = db.row(conn, "SELECT * FROM threads")
    assert thread is not None
    import json

    from durable_text import clock

    return TurnState(
        thread_id=thread["id"], conn=conn, now=clock.get(conn), customer=json.loads(thread["customer_json"])
    )


class SlowScripted(CompletionLLM):
    """Scripted policy with a delay per call, so a turn can be cancelled mid-flight."""

    def __init__(self, delay_s: float) -> None:
        super().__init__("scripted-slow")
        self._inner = ScriptedLLM()
        self._delay = delay_s

    async def complete(self, req: LLMRequest) -> LLMCall:
        await asyncio.sleep(self._delay)
        return await self._inner.complete(req)


class Harness:
    """An in-process worker on the test database with a chosen LLM."""

    def __init__(
        self, settings: Settings, conn: sqlite3.Connection, llm: Callable[[], CompletionLLM] = ScriptedLLM
    ) -> None:
        self.settings = settings
        self.conn = conn
        self.worker = Worker(settings, "scripted", llm)
        self.worker.register()

    def text(self, body: str, sid: str | None = None) -> channel.InboundResult:
        fields = {"From": seed.CUSTOMER["phone"], "To": seed.BUSINESS_NUMBER, "Body": body}
        if sid:
            fields["MessageSid"] = sid
        return channel.receive_inbound(self.conn, channel.InboundSMS(**fields), self.settings.coalesce_s)

    async def step(self) -> None:
        self.worker.watch_cancels_and_renew()
        await self.worker.tick()
        await asyncio.sleep(0.02)

    async def drain(self, limit_s: float = 15) -> None:
        loop = asyncio.get_running_loop()
        end = loop.time() + limit_s
        while loop.time() < end:
            await self.step()
            busy = db.row(self.conn, "SELECT 1 FROM jobs WHERE status IN ('queued', 'running')")
            if not self.worker.running and busy is None:
                return
        raise AssertionError("jobs did not drain")

    def outbound(self) -> list[sqlite3.Row]:
        return db.rows(self.conn, "SELECT * FROM messages WHERE direction = 'out' ORDER BY id")

    def turns(self, kind: str | None = None) -> list[sqlite3.Row]:
        if kind:
            return db.rows(self.conn, "SELECT * FROM turns WHERE kind = ? ORDER BY id", kind)
        return db.rows(self.conn, "SELECT * FROM turns ORDER BY id")


@pytest.fixture
def harness(settings: Settings, conn: sqlite3.Connection) -> Harness:
    return Harness(settings, conn)
