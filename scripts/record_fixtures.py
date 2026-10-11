"""Record replay fixtures from a real live run of the main flow.

Starts the server in live mode on a Unix socket with a fresh database, drives
the scripted flow over HTTP, and writes every LLM call the worker made to
fixtures/replay.json. Needs a Claude Code login or ANTHROPIC_API_KEY.

    uv run python scripts/record_fixtures.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from durable_text.config import DEFAULT_FIXTURES  # noqa: E402
from durable_text.demo_flow import Flow  # noqa: E402
from durable_text.llm.replay import merge_recording  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_FIXTURES)
    p.add_argument("--keep", action="store_true", help="keep the temp folder for inspection")
    args = p.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="dta-record-"))
    sock = tmp / "app.sock"
    jsonl = tmp / "calls.jsonl"
    env = {k: v for k, v in os.environ.items() if not (k == "CLAUDECODE" or k.startswith("CLAUDE_CODE_"))}
    env.update(
        {"DTA_DB": str(tmp / "rec.db"), "DTA_RECORD": str(jsonl), "DTA_UDS": str(sock), "DEMO_REPLAY": "0"}
    )
    log = (tmp / "server.log").open("wb")
    server = subprocess.Popen(
        [sys.executable, "-m", "durable_text"],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=log,
        start_new_session=True,
    )
    try:
        transport = httpx.HTTPTransport(uds=str(sock))
        with httpx.Client(transport=transport, base_url="http://app", timeout=30) as client:
            for _ in range(200):
                try:
                    if client.get("/healthz").json().get("worker_pid"):
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            mode = client.get("/healthz").json()["mode"]
            if mode != "live":
                print(f"server is in {mode} mode: no Claude login or key, nothing to record")
                return 2
            started = time.monotonic()
            Flow(client, timeout_s=240, gap_s=1.0).run()
            print(f"flow took {time.monotonic() - started:.0f} s")
    finally:
        server.terminate()
        try:
            server.wait(20)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
    count = merge_recording(jsonl, args.out)
    print(f"wrote {count} recorded calls to {args.out}")
    if args.keep:
        print(f"kept {tmp}")
    else:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
