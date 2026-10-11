"""SQLite store shared by the web server and the worker process (WAL mode).

Every write that matters runs in one ``BEGIN IMMEDIATE`` transaction, so a turn's
checkpoint, outbound texts, timers and delegations land together or not at all.
The ``events`` table is an append-only change feed the web server tails for SSE.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS clock (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    sim_now TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    phone TEXT NOT NULL UNIQUE,
    business_number TEXT NOT NULL,
    customer_json TEXT NOT NULL,
    opted_out INTEGER NOT NULL DEFAULT 0,
    active_agent TEXT NOT NULL,
    checkpoint_version INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    agent_id TEXT NOT NULL,
    chat_ctx_json TEXT NOT NULL,
    values_json TEXT NOT NULL,
    turn_id INTEGER,
    created_ms INTEGER NOT NULL,
    PRIMARY KEY (thread_id, version)
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('in', 'out')),
    sid TEXT UNIQUE,
    body TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'agent',
    status TEXT NOT NULL,
    segments INTEGER NOT NULL DEFAULT 1,
    encoding TEXT NOT NULL DEFAULT 'GSM-7',
    turn_id INTEGER,
    consumed_by_turn INTEGER,
    sim_at TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS webhooks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sid TEXT NOT NULL,
    form_json TEXT NOT NULL,
    outcome TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('turn', 'wake', 'dispatch')),
    thread_id TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'queued',
    run_after_ms INTEGER NOT NULL,
    lease_owner INTEGER,
    lease_expires_ms INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_ms INTEGER NOT NULL,
    finished_ms INTEGER
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, run_after_ms);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    thread_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    worker_pid INTEGER NOT NULL,
    agent_before TEXT,
    agent_after TEXT,
    checkpoint_from INTEGER,
    checkpoint_to INTEGER,
    input_json TEXT NOT NULL DEFAULT '{}',
    trace_json TEXT NOT NULL DEFAULT '[]',
    llm_json TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    sim_at TEXT NOT NULL,
    started_ms INTEGER NOT NULL,
    ended_ms INTEGER
);
CREATE TABLE IF NOT EXISTS delegations (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    from_agent TEXT NOT NULL,
    to_agent TEXT NOT NULL,
    task_json TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT,
    job_id INTEGER,
    created_ms INTEGER NOT NULL,
    decided_ms INTEGER
);
CREATE TABLE IF NOT EXISTS timers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    due_sim TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    created_by_turn INTEGER,
    fired_job INTEGER,
    note TEXT
);
CREATE TABLE IF NOT EXISTS workers (
    pid INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    started_ms INTEGER NOT NULL,
    ended_ms INTEGER,
    last_seen_ms INTEGER
);
CREATE TABLE IF NOT EXISTS technicians (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    skills_json TEXT NOT NULL,
    zone TEXT NOT NULL,
    rating REAL NOT NULL,
    windows_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS slots (
    id TEXT PRIMARY KEY,
    day TEXT NOT NULL,
    start_sim TEXT NOT NULL,
    end_sim TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    thread_id TEXT,
    technician_id TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);
"""


def now_ms() -> int:
    return int(time.time() * 1000)


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


@contextmanager
def tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """One write transaction. Takes the write lock up front to avoid upgrade deadlocks."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def emit(conn: sqlite3.Connection, type_: str, **payload: Any) -> None:
    conn.execute(
        "INSERT INTO events (type, payload_json, created_ms) VALUES (?, ?, ?)",
        (type_, json.dumps(payload), now_ms()),
    )


def row(conn: sqlite3.Connection, sql: str, *args: Any) -> sqlite3.Row | None:
    cur: sqlite3.Row | None = conn.execute(sql, args).fetchone()
    return cur


def rows(conn: sqlite3.Connection, sql: str, *args: Any) -> list[sqlite3.Row]:
    return list(conn.execute(sql, args).fetchall())


def loads(text: str | None, default: Any = None) -> Any:
    if text is None:
        return default
    return json.loads(text)
