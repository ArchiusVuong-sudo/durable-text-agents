"""End to end: real server + real worker process, replayed LLM, flow driven over HTTP."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from durable_text.config import DEFAULT_FIXTURES, ROOT
from durable_text.demo_flow import Flow


@pytest.fixture
def server(tmp_path: Path) -> Iterator[httpx.Client]:
    sock = tmp_path / "app.sock"
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
