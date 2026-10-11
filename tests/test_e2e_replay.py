"""End to end: real server + real worker process, replayed LLM, flow driven over HTTP."""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from durable_text.config import DEFAULT_FIXTURES, ROOT
from durable_text.demo_flow import Flow


@pytest.fixture
def server(tmp_path: Path) -> Iterator[httpx.Client]:
    # AF_UNIX paths are capped near 104 bytes on macOS, and pytest's tmp_path under the
    # default TMPDIR (/var/folders/...) is longer than that, so the socket gets a short folder.
    sockdir = Path(tempfile.mkdtemp(prefix="dta-", dir="/tmp"))
    sock = sockdir / "s"
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env.update(
        {
            "DEMO_REPLAY": "1",
            "DTA_DB": str(tmp_path / "e2e.db"),
            "DTA_UDS": str(sock),
            "DTA_FIXTURES": str(DEFAULT_FIXTURES),
            "DTA_REPLAY_SCALE": "0.35",
            "DTA_COALESCE_S": "1.5",
        }
    )
    log = (tmp_path / "server.log").open("wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "durable_text"],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=log,
        start_new_session=True,
    )
    client = httpx.Client(transport=httpx.HTTPTransport(uds=str(sock)), base_url="http://app", timeout=20)
    try:
        for _ in range(300):
            try:
                if client.get("/healthz").json().get("worker_pid"):
                    break
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.05)
        else:
            pytest.fail("server did not start: " + (tmp_path / "server.log").read_text()[-2000:])
        yield client
    finally:
        client.close()
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        shutil.rmtree(sockdir, ignore_errors=True)


def test_main_flow_in_replay_mode(server: httpx.Client) -> None:
    assert server.get("/healthz").json()["mode"] == "replay"
    final = Flow(server, timeout_s=90, gap_s=0.3, log=lambda _: None).run()

    turns = {t["id"]: t for t in final["turns"]}
    statuses = [t["status"] for t in sorted(turns.values(), key=lambda t: t["id"])]
    assert "superseded" in statuses
    assert "interrupted" in statuses  # the dispatch job cut off by SIGKILL
    dispatch = [t for t in turns.values() if t["kind"] == "dispatch" and t["status"] == "committed"]
    assert dispatch and dispatch[0]["attempt"] == 2

    # every LLM answer came from the recorded live run, none from the fallback policy
    sources = {c["source"] for t in turns.values() for c in t["llm"]}
    assert sources == {"replay"}, sources

    out = [m for m in final["messages"] if m["direction"] == "out"]
    assert sum(1 for m in out if m["kind"] == "template") == 1  # STOP confirmation only
    assert all(m["status"] == "delivered" for m in out)
    assert any("Priya" in m["body"] for m in out)
    assert final["thread"]["opted_out"] is True
    assert [t["status"] for t in final["timers"]] == ["fired", "suppressed"]
    assert any(w["outcome"] == "duplicate" for w in final["webhooks"])
    workers = {w["status"] for w in final["workers"]}
    assert "killed" in workers and "running" in workers


def _sse_events(server: httpx.Client, headers: dict[str, str] | None = None) -> queue.Queue[tuple[str, str]]:
    """Read (event, id) pairs from /api/stream on a background thread."""
    out: queue.Queue[tuple[str, str]] = queue.Queue()

    def run() -> None:
        event, eid = "", ""
        try:
            with server.stream("GET", "/api/stream", headers=headers, timeout=None) as resp:
                for line in resp.iter_lines():
                    if line.startswith("event: "):
                        event = line[7:]
                    elif line.startswith("id: "):
                        eid = line[4:]
                    elif line == "" and event:
                        out.put((event, eid))
                        event, eid = "", ""
        except httpx.HTTPError:
            pass  # server shut down at the end of the test

    threading.Thread(target=run, daemon=True).start()
    return out


def _next_change(events: queue.Queue[tuple[str, str]], timeout_s: float = 5.0) -> tuple[str, str]:
    deadline = time.monotonic() + timeout_s
    while (left := deadline - time.monotonic()) > 0:
        try:
            event, eid = events.get(timeout=left)
        except queue.Empty:
            break
        if event == "change":
            return event, eid
    pytest.fail("no change event on the stream")


def test_stream_keeps_flowing_after_reset(server: httpx.Client) -> None:
    for _ in range(3):  # push event ids up so a restart at 1 would fall below the cursor
        server.post("/api/clock/advance", json={"hours": 1}).raise_for_status()
    events = _sse_events(server)
    assert events.get(timeout=5)[0] == "hello"

    server.post("/api/reset").raise_for_status()
    _, reset_id = _next_change(events)
    server.post("/api/clock/advance", json={"hours": 1}).raise_for_status()
    _, after_id = _next_change(events)
    assert int(after_id) > int(reset_id)


def test_stream_follows_a_cursor_past_the_end(server: httpx.Client) -> None:
    """A Last-Event-ID beyond the feed (old tab, recreated database) resyncs instead of stalling."""
    events = _sse_events(server, headers={"Last-Event-ID": "999999"})
    assert events.get(timeout=5)[0] == "hello"
    _next_change(events)  # the resync itself
    server.post("/api/clock/advance", json={"hours": 1}).raise_for_status()
    _next_change(events)
