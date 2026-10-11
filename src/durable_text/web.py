"""FastAPI app: carrier-style webhook, operator API, SSE change feed and the UI."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from . import channel, db, seed
from .config import Settings
from .llm.factory import ModeInfo, detect_mode
from .llm.replay import load_fixtures
from .state import snapshot
from .supervisor import Supervisor

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class AdvanceBody(BaseModel):
    hours: float = Field(gt=0, le=72)


class App:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.conn = db.connect(settings.db_path)
        db.init(self.conn)
        if db.row(self.conn, "SELECT 1 FROM threads") is None:
            seed.seed(self.conn)
        self.mode: ModeInfo = detect_mode(settings)
        self.fixture_count = len(load_fixtures(settings.fixtures)) if self.mode.mode == "replay" else None
        self.supervisor = Supervisor(settings, self.mode.mode)
        self._carrier: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self.supervisor.start()
        self._carrier = asyncio.create_task(self._carrier_loop(), name="fake-carrier")

    async def stop(self) -> None:
        if self._carrier:
            self._carrier.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._carrier
        await self.supervisor.stop()
        self.conn.close()

    async def _carrier_loop(self) -> None:
        while True:
            try:
                channel.deliver_outbox(self.conn)
            except Exception:  # keep delivering; log and retry next tick
                log.exception("carrier tick failed")
            await asyncio.sleep(0.3)

    def state(self) -> dict[str, Any]:
        data = snapshot(
            self.conn,
            mode=self.mode.mode,
            mode_reason=self.mode.reason,
            model=self.settings.model,
            fixtures=self.fixture_count,
        )
        data["worker_pid"] = self.supervisor.pid
        data["coalesce_s"] = self.settings.coalesce_s
        return data


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    holder: dict[str, App] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        app_state = App(settings)
        holder["app"] = app_state
        await app_state.start()
        log.info("mode %s (%s)", app_state.mode.mode, app_state.mode.reason)
        try:
            yield
        finally:
            await app_state.stop()

    api = FastAPI(title="Durable Text Agents", lifespan=lifespan)
    api.mount("/static", StaticFiles(directory=STATIC), name="static")

    def a() -> App:
        return holder["app"]

    @api.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @api.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "mode": a().mode.mode, "worker_pid": a().supervisor.pid}

    @api.post("/sms/inbound")
    async def inbound(
        From: Annotated[str, Form()],
        To: Annotated[str, Form()],
        Body: Annotated[str, Form()],
        MessageSid: Annotated[str | None, Form()] = None,
    ) -> Response:
        """Carrier-style webhook. Answers 200 at once; the reply goes out from the outbox."""
        fields: dict[str, Any] = {"From": From, "To": To, "Body": Body}
        if MessageSid:
            fields["MessageSid"] = MessageSid
        try:
            sms = channel.InboundSMS(**fields)
        except ValidationError as exc:
            raise HTTPException(422, exc.errors()) from exc
        result = channel.receive_inbound(a().conn, sms, settings.coalesce_s)
        return Response(
            content="<Response></Response>",
            media_type="application/xml",
            headers={"X-Outcome": result.outcome},
        )

    @api.get("/api/state")
    async def get_state() -> JSONResponse:
        return JSONResponse(a().state())

    @api.get("/api/stream")
    async def stream(request: Request) -> StreamingResponse:
        async def gen() -> AsyncIterator[str]:
            conn = db.connect(settings.db_path)
            try:
                last = request.headers.get("last-event-id")
                cursor = int(last) if last and last.isdigit() else _max_event(conn)
                yield f"id: {cursor}\nevent: hello\ndata: {{}}\n\n"
                idle = 0
                while not await request.is_disconnected():
                    top = _max_event(conn)
                    if top < cursor:
                        # The feed went backwards (database recreated, or a stale
                        # Last-Event-ID): follow it and tell the page to refresh.
                        cursor = top
                        yield f"id: {cursor}\nevent: change\ndata: {json.dumps({'types': ['reset']})}\n\n"
                        idle = 0
                    new = db.rows(
                        conn, "SELECT id, type FROM events WHERE id > ? ORDER BY id LIMIT 200", cursor
                    )
                    if new:
                        cursor = new[-1]["id"]
                        types = sorted({r["type"] for r in new})
                        yield f"id: {cursor}\nevent: change\ndata: {json.dumps({'types': types})}\n\n"
                        idle = 0
                    else:
                        idle += 1
                        if idle % 60 == 0:
                            yield ": keep-alive\n\n"
                    await asyncio.sleep(0.25)
            finally:
                conn.close()

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @api.post("/api/worker/kill")
    async def kill_worker() -> dict[str, Any]:
        pid = a().supervisor.kill()
        if pid is None:
            raise HTTPException(409, "no worker running")
        return {"killed": pid}

    @api.post("/api/clock/advance")
    async def advance(body: AdvanceBody) -> dict[str, Any]:
        return {"sim_now": channel.advance_clock(a().conn, body.hours)}

    @api.post("/api/clock/next-timer")
    async def next_timer() -> dict[str, Any]:
        moved = channel.jump_to_next_timer(a().conn)
        if moved is None:
            raise HTTPException(409, "no pending timers")
        return {"sim_now": moved}

    @api.post("/api/delegations/{delegation_id}/approve")
    async def approve(delegation_id: str) -> dict[str, Any]:
        try:
            event = channel.approve(a().conn, delegation_id)
        except channel.ApprovalError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"event": event}

    @api.post("/api/dev/replay-webhook")
    async def replay_webhook() -> dict[str, Any]:
        result = channel.replay_last_webhook(a().conn, settings.coalesce_s)
        if result is None:
            raise HTTPException(409, "no webhook to replay yet")
        return result.model_dump()

    @api.post("/api/reset")
    async def reset() -> dict[str, Any]:
        app_state = a()
        await app_state.supervisor.stop()
        seed.seed(app_state.conn)
        app_state.conn.execute("DELETE FROM workers")
        await app_state.supervisor.start()
        return {"ok": True, "worker_pid": app_state.supervisor.pid}

    return api


def _max_event(conn: Any) -> int:
    r = db.row(conn, "SELECT MAX(id) AS id FROM events")
    return int(r["id"]) if r and r["id"] else 0
