"""The scripted main flow, driven over HTTP exactly as the UI drives it.

Used three ways: ``scripts/record_fixtures.py`` runs it live to record replay
fixtures, the end-to-end test runs it in replay mode, and ``python -m
durable_text.demo_flow`` runs it against a running server.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from .seed import BUSINESS_NUMBER, CUSTOMER

State = dict[str, Any]


class FlowError(AssertionError):
    pass


@dataclass
class Flow:
    client: httpx.Client
    timeout_s: float = 180.0
    gap_s: float = 0.5
    log: Callable[[str], None] = print
    sids: list[str] = field(default_factory=list)

    def state(self) -> State:
        r = self.client.get("/api/state")
        r.raise_for_status()
        data: State = r.json()
        return data

    def wait(self, what: str, cond: Callable[[State], bool], timeout_s: float | None = None) -> State:
        deadline = time.monotonic() + (timeout_s or self.timeout_s)
        while time.monotonic() < deadline:
            s = self.state()
            if cond(s):
                return s
            time.sleep(0.25)
        raise FlowError(f"timed out waiting for: {what}")

    def text(self, body: str) -> str:
        sid = f"msg_flow_{len(self.sids) + 1:03d}"
        r = self.client.post(
            "/sms/inbound",
            data={"From": CUSTOMER["phone"], "To": BUSINESS_NUMBER, "Body": body, "MessageSid": sid},
        )
        r.raise_for_status()
        self.sids.append(sid)
        self.log(f"> {body!r} -> {r.headers.get('x-outcome')}")
        return r.headers.get("x-outcome", "")

    def post(self, path: str, **json: Any) -> State:
        r = self.client.post(path, json=json or None)
        r.raise_for_status()
        out: State = r.json()
        return out

    def run(self) -> State:
        s = self.post("/api/reset")
        self.log(f"reset, worker pid {s['worker_pid']}")

        # 1. Two texts in a row become one turn: front desk hands off to the scheduler.
        self.text("hi, my dishwasher is leaking")
        time.sleep(self.gap_s)
        self.text("water all over the kitchen floor")
        s = self.wait(
            "first reply delivered", lambda s: _committed(s, "turn") >= 1 and _delivered_all(s)
        )
        first = _turns(s, "turn")[-1]
        if first["input"].get("coalesced") != 2:
            raise FlowError(f"expected 2 coalesced texts, got {first['input']}")
        if first["agent_after"] != "scheduler":
            raise FlowError("front desk did not hand off to the scheduler")

        # 2. A second text lands while the turn is running: supersede and rerun.
        self.text("tomorrow morning works")
        self.wait("turn running", lambda s: _running(s, "turn"), timeout_s=30)
        self.text("actually afternoon is better")
        s = self.wait("rerun committed", lambda s: _committed(s, "turn") >= 2)
        if not any(t["status"] == "superseded" for t in s["turns"]):
            raise FlowError("no superseded turn")
        self.wait("delegation created", lambda s: bool(s["delegations"]))

        # 3. Kill the worker while dispatch is running; a new worker picks the job up.
        s = self.wait("dispatch running", lambda s: s["delegations"][0]["status"] == "running", 60)
        killed = self.post("/api/worker/kill")["killed"]
        self.log(f"killed worker {killed}")
        s = self.wait("dispatch proposal", lambda s: s["delegations"][0]["status"] == "needs_approval")
        if s["worker_pid"] == killed:
            raise FlowError("worker was not replaced")

        # 4. Three simulated hours later the customer checks in.
        self.post("/api/clock/advance", hours=3)
        self.text("any update on my repair?")
        s = self.wait("status turn committed", lambda s: _committed(s, "turn") >= 3)

        # 5. A dispatcher approves; the thread wakes with no inbound text.
        delegation = s["delegations"][0]["id"]
        self.post(f"/api/delegations/{delegation}/approve")
        s = self.wait("approval wake committed", lambda s: _committed(s, "wake") >= 1)
        if len([t for t in s["timers"] if t["status"] == "pending"]) != 2:
            raise FlowError(f"expected 2 pending timers, got {s['timers']}")

        # 6. A durable timer fires on the simulated clock.
        self.post("/api/clock/next-timer")
        s = self.wait(
            "reminder delivered", lambda s: _committed(s, "wake") >= 2 and _delivered_all(s)
        )

        # 7. A carrier retry with the same MessageSid gets no second reply.
        dup = self.post("/api/dev/replay-webhook")
        if dup["outcome"] != "duplicate":
            raise FlowError(f"replayed webhook was not a duplicate: {dup}")

        # 8. STOP is answered from a template; the follow-up timer is suppressed.
        self.text("STOP")
        self.post("/api/clock/next-timer")
        s = self.wait(
            "follow-up suppressed",
            lambda s: any(t["status"] == "suppressed" for t in s["timers"]) and _delivered_all(s),
        )
        self.log("flow complete")
        return s


def _turns(s: State, kind: str) -> list[dict[str, Any]]:
    return sorted((t for t in s["turns"] if t["kind"] == kind), key=lambda t: t["id"])


def _committed(s: State, kind: str) -> int:
    return sum(1 for t in s["turns"] if t["kind"] == kind and t["status"] == "committed")


def _running(s: State, kind: str) -> bool:
    return any(t["kind"] == kind and t["status"] == "running" for t in s["turns"])


def _delivered_all(s: State) -> bool:
    return all(m["status"] != "queued" for m in s["messages"])


def main() -> None:
    p = argparse.ArgumentParser(description="Run the scripted flow against a running server")
    p.add_argument("--url", default="http://127.0.0.1:4106")
    args = p.parse_args()
    with httpx.Client(base_url=args.url, timeout=30) as client:
        Flow(client).run()


if __name__ == "__main__":
    main()
