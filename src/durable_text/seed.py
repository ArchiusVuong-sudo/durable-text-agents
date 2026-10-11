"""Synthetic data: one fictional repair desk, one customer, six technicians.

Everything here is invented. Phone numbers use the fictional 555-01xx range.
Only one technician fits the scripted request (dishwasher, East zone, Tuesday
afternoon), so the dispatch pick stays stable across live runs.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

from . import clock, db

BUSINESS_NAME = "Kettle & Coil Repair"
BUSINESS_NUMBER = "+13125550100"
THREAD_ID = "t1"

CUSTOMER = {
    "name": "Maya Chen",
    "phone": "+13125550142",
    "zone": "East",
    "address": "41 Larkspur Ave (synthetic)",
    "appliance": "dishwasher",
    "appliance_detail": "2023 dishwasher, model KC-DW45",
}

TECHNICIANS = [
    # id, name, skills, zone, rating, availability windows (weekday, start hour, end hour)
    ("tech-priya", "Priya Nair", ["dishwasher", "washer"], "East", 4.9, [("Tue", 12, 18), ("Thu", 8, 12)]),
    ("tech-tomas", "Tomas Reyes", ["oven", "dryer"], "East", 4.7, [("Tue", 8, 18)]),
    ("tech-grace", "Grace Okafor", ["dishwasher", "fridge"], "West", 4.8, [("Tue", 12, 18)]),
    ("tech-daniel", "Daniel Moreau", ["fridge", "dishwasher"], "East", 4.6, [("Tue", 8, 12), ("Wed", 8, 18)]),
    ("tech-lena", "Lena Fischer", ["dishwasher", "oven"], "North", 4.5, [("Tue", 8, 18)]),
    ("tech-omar", "Omar Haddad", ["washer", "dryer"], "South", 4.8, [("Tue", 8, 18), ("Wed", 8, 18)]),
]

SLOT_HOURS = [8, 10, 12, 14, 16]
BOOKED = {("Tue", 12), ("Tue", 16)}  # keep Tuesday narrow: two mornings and one afternoon


def slot_id(start: datetime) -> str:
    return f"{start.strftime('%a').lower()}-{start.strftime('%H%M')}"


def build_slots(start: datetime = clock.SIM_START, days: int = 5) -> list[tuple[str, str, str, str, str]]:
    out = []
    first = (start + timedelta(days=1)).replace(hour=0, minute=0)
    for d in range(days):
        day = first + timedelta(days=d)
        for hour in SLOT_HOURS:
            s = day.replace(hour=hour)
            e = s + timedelta(hours=2)
            status = "booked" if (s.strftime("%a"), hour) in BOOKED else "open"
            out.append((slot_id(s), s.strftime("%a"), clock.fmt_iso(s), clock.fmt_iso(e), status))
    return out


def seed(conn: sqlite3.Connection) -> None:
    db.init(conn)
    with db.tx(conn):
        for table in (
            "clock",
            "threads",
            "checkpoints",
            "messages",
            "webhooks",
            "jobs",
            "turns",
            "delegations",
            "timers",
            "technicians",
            "slots",
            "events",
        ):
            conn.execute(f"DELETE FROM {table}")  # noqa: S608 - fixed table names
        # Event ids must keep rising across resets: open SSE streams and reconnects
        # (Last-Event-ID) tail "id > cursor", so restarting at 1 would silence them.
        conn.execute("DELETE FROM sqlite_sequence WHERE name != 'events'")
        clock.set_(conn, clock.SIM_START)
        conn.execute(
            "INSERT INTO threads (id, phone, business_number, customer_json, active_agent, "
            "checkpoint_version) VALUES (?, ?, ?, ?, 'front_desk', 0)",
            (THREAD_ID, CUSTOMER["phone"], BUSINESS_NUMBER, json.dumps(CUSTOMER)),
        )
        for tid, name, skills, zone, rating, windows in TECHNICIANS:
            conn.execute(
                "INSERT INTO technicians VALUES (?, ?, ?, ?, ?, ?)",
                (tid, name, json.dumps(skills), zone, rating, json.dumps(windows)),
            )
        for sid, day, s, e, status in build_slots():
            conn.execute(
                "INSERT INTO slots (id, day, start_sim, end_sim, status) VALUES (?, ?, ?, ?, ?)",
                (sid, day, s, e, status),
            )
        db.emit(conn, "reset")


def technician_free(windows: list[list[object]], start: datetime, end: datetime) -> bool:
    day = start.strftime("%a")
    for w_day, w_start, w_end in windows:
        if w_day == day and int(str(w_start)) <= start.hour and end.hour <= int(str(w_end)):
            return True
    return False
