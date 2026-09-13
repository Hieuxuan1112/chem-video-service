import sqlite3
import threading
from pathlib import Path

from app.jobs.clock import FakeClock, SystemClock
from app.jobs.db import JOBS_SCHEMA_SQL, apply_schema, connect


def test_pragmas(tmp_path: Path) -> None:
    conn = connect(tmp_path / "nested" / "app.sqlite")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.isolation_level is None
    assert conn.row_factory is sqlite3.Row


def test_apply_schema_idempotent(tmp_path: Path) -> None:
    conn = connect(tmp_path / "app.sqlite")
    apply_schema(conn, JOBS_SCHEMA_SQL)
    apply_schema(conn, JOBS_SCHEMA_SQL)
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"jobs", "job_events", "stage_checkpoints", "worker_heartbeats"} <= tables
    indexes = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    }
    assert {"jobs_claim", "job_events_job"} <= indexes


def test_fake_clock_advance() -> None:
    clock = FakeClock(start=100.0)
    assert clock.now() == 100.0
    clock.advance(2.5)
    assert clock.now() == 102.5

    def advance_many() -> None:
        for _ in range(100):
            clock.advance(1.0)

    threads = [threading.Thread(target=advance_many) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert clock.now() == 902.5
    assert SystemClock().now() > 1_700_000_000


def test_returning_supported() -> None:
    assert sqlite3.sqlite_version_info >= (3, 35)
