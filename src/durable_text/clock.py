"""The simulated business clock.

Two clocks run side by side: wall time (leases, coalescing, retries) and the
simulated clock (appointments, durable timers). Only the simulated clock moves
when the user clicks "+3 h" or "Next timer", so hours-long waits are visible in
seconds without faking any wall-time behaviour.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

SIM_START = datetime(2026, 10, 12, 16, 5)  # Mon 12 Oct 2026 16:05


def parse(value: str) -> datetime:
    return datetime.fromisoformat(value)


def fmt_iso(value: datetime) -> str:
    return value.replace(second=0, microsecond=0).isoformat(timespec="minutes")


def human(value: datetime) -> str:
    """'Tue 13 Oct 14:00' - short, unambiguous, used in prompts and UI."""
    return value.strftime("%a %-d %b %H:%M")


def window(start: datetime, end: datetime) -> str:
    return f"{start.strftime('%a %-d %b')} {start.strftime('%H:%M')}-{end.strftime('%H:%M')}"


def get(conn: sqlite3.Connection) -> datetime:
    r = conn.execute("SELECT sim_now FROM clock WHERE id = 1").fetchone()
    return parse(r["sim_now"]) if r else SIM_START


def set_(conn: sqlite3.Connection, value: datetime) -> None:
    conn.execute(
        "INSERT INTO clock (id, sim_now) VALUES (1, ?) "
        "ON CONFLICT(id) DO UPDATE SET sim_now = excluded.sim_now",
        (fmt_iso(value),),
    )


def advance(conn: sqlite3.Connection, delta: timedelta) -> datetime:
    new = get(conn) + delta
    set_(conn, new)
    return new
