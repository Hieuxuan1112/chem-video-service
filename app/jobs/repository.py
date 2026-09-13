"""JobRepository: all job SQL lives here (spec §4)."""

import base64
import binascii
import builtins
import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.domain.errors import AppError, IdempotencyConflict, NotRetryable, StaleLeaseError
from app.domain.job import Job, JobStatus, Lease, Stage, new_job_id
from app.jobs.clock import Clock
from app.jobs.db import JOBS_SCHEMA_SQL, apply_schema, connect


@dataclass(frozen=True)
class Event:
    id: int
    job_id: str
    attempt: int
    ts: float
    type: str
    stage: str | None
    data: dict[str, Any]


@dataclass(frozen=True)
class CheckpointRow:
    job_id: str
    stage_key: str
    attempt: int
    input_hash: str
    manifest_sha256: str
    versions: dict[str, Any]
    created_at: float


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        query=row["query"],
        topic_id=row["topic_id"],
        status=JobStatus(row["status"]),
        stage=Stage(row["stage"]) if row["stage"] is not None else None,
        attempt=row["attempt"],
        lease_claims=row["lease_claims"],
        scheduled_retries=row["scheduled_retries"],
        worker_id=row["worker_id"],
        lease_until=row["lease_until"],
        not_before=row["not_before"],
        generation_source=row["generation_source"],
        fallback_reason=row["fallback_reason"],
        error_code=row["error_code"],
        error_cause=row["error_cause"],
        error_message=row["error_message"],
        retryable=bool(row["retryable"]) if row["retryable"] is not None else None,
        idempotency_key=row["idempotency_key"],
        request_fingerprint=row["request_fingerprint"],
        pipeline_version=row["pipeline_version"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def _encode_cursor(created_at: float, job_id: str) -> str:
    return base64.urlsafe_b64encode(f"{created_at!r}|{job_id}".encode()).decode()


def _decode_cursor(cursor: str) -> tuple[float, str]:
    try:
        created_at, job_id = base64.urlsafe_b64decode(cursor.encode()).decode().split("|", 1)
        return float(created_at), job_id
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid cursor: {cursor!r}") from exc


class JobRepository:
    def __init__(self, db_path: Path, clock: Clock) -> None:
        self._db_path = db_path
        self._clock = clock
        self._local = threading.local()
        apply_schema(self._conn(), JOBS_SCHEMA_SQL)

    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self._db_path)
            self._local.conn = conn
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """One write transaction; BEGIN IMMEDIATE takes the writer lock up front."""
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    @staticmethod
    def _insert_event(
        conn: sqlite3.Connection,
        job_id: str,
        attempt: int,
        ts: float,
        type: str,
        stage: str | None,
        data: dict[str, Any] | None,
    ) -> None:
        conn.execute(
            "INSERT INTO job_events(job_id, attempt, ts, type, stage, data_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, attempt, ts, type, stage, json.dumps(data or {}, sort_keys=True)),
        )

    # --- create / read -------------------------------------------------------------------------

    def create(
        self,
        *,
        query: str,
        topic_id: str,
        pipeline_version: str,
        idempotency_key: str | None,
        request_fingerprint: str | None,
    ) -> tuple[Job, bool]:
        job_id = new_job_id()
        now = self._clock.now()
        try:
            with self._tx() as conn:
                conn.execute(
                    "INSERT INTO jobs(id, query, topic_id, status, idempotency_key,"
                    " request_fingerprint, pipeline_version, created_at, updated_at)"
                    " VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?)",
                    (
                        job_id,
                        query,
                        topic_id,
                        idempotency_key,
                        request_fingerprint,
                        pipeline_version,
                        now,
                        now,
                    ),
                )
                self._insert_event(
                    conn, job_id, 0, now, "job_created", None, {"topic_id": topic_id}
                )
        except sqlite3.IntegrityError:
            if idempotency_key is None:
                raise
            row = (
                self._conn()
                .execute("SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,))
                .fetchone()
            )
            if row is None:
                raise
            existing = _row_to_job(row)
            if existing.request_fingerprint != request_fingerprint:
                raise IdempotencyConflict(idempotency_key) from None
            return existing, True
        job = self.get(job_id)
        assert job is not None
        return job, False

    def get(self, job_id: str) -> Job | None:
        row = self._conn().execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row is not None else None

    def list(
        self,
        *,
        status: JobStatus | None,
        topic_id: str | None,
        limit: int,
        cursor: str | None,
    ) -> tuple[list[Job], str | None]:
        where: list[str] = []
        params: list[Any] = []
        if status is not None:
            where.append("status = ?")
            params.append(str(status))
        if topic_id is not None:
            where.append("topic_id = ?")
            params.append(topic_id)
        if cursor is not None:
            created_at, last_id = _decode_cursor(cursor)
            where.append("(created_at < ? OR (created_at = ? AND id < ?))")
            params.extend([created_at, created_at, last_id])
        sql = "SELECT * FROM jobs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit + 1)
        rows = self._conn().execute(sql, params).fetchall()
        jobs = [_row_to_job(r) for r in rows[:limit]]
        next_cursor = None
        if len(rows) > limit:
            next_cursor = _encode_cursor(jobs[-1].created_at, jobs[-1].id)
        return jobs, next_cursor

    # --- events --------------------------------------------------------------------------------

    def add_event(
        self,
        job_id: str,
        attempt: int,
        type: str,
        stage: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Unfenced event for API-side actions; worker writes use add_event_fenced."""
        self._insert_event(self._conn(), job_id, attempt, self._clock.now(), type, stage, data)

    def events(self, job_id: str) -> builtins.list[Event]:
        rows = (
            self._conn()
            .execute("SELECT * FROM job_events WHERE job_id = ? ORDER BY id", (job_id,))
            .fetchall()
        )
        return [
            Event(
                id=r["id"],
                job_id=r["job_id"],
                attempt=r["attempt"],
                ts=r["ts"],
                type=r["type"],
                stage=r["stage"],
                data=json.loads(r["data_json"]),
            )
            for r in rows
        ]

    # --- claim, sweep, heartbeat, fencing (spec §4.3-4.4) ---------------------------------------

    _FENCE = "id = ? AND worker_id = ? AND attempt = ? AND status = 'running'"

    def _fenced_update(
        self, conn: sqlite3.Connection, lease: Lease, set_sql: str, params: tuple[Any, ...]
    ) -> None:
        cur = conn.execute(
            f"UPDATE jobs SET {set_sql} WHERE {self._FENCE}",
            (*params, lease.job_id, lease.worker_id, lease.attempt),
        )
        if cur.rowcount != 1:
            raise StaleLeaseError(f"lease lost: {lease}")

    def sweep_expired(self, *, max_claims: int) -> int:
        now = self._clock.now()
        with self._tx() as conn:
            rows = conn.execute(
                "UPDATE jobs SET status='failed', error_code='WORKER_LOST',"
                " error_cause='lease_expired_max_claims', error_message=?, retryable=1,"
                " worker_id=NULL, lease_until=NULL, finished_at=?, updated_at=?"
                " WHERE status='running' AND lease_until < ? AND lease_claims >= ?"
                " RETURNING id, attempt, stage",
                (f"lease expired after {max_claims} claims", now, now, now, max_claims),
            ).fetchall()
            for r in rows:
                self._insert_event(
                    conn,
                    r["id"],
                    r["attempt"],
                    now,
                    "job_failed",
                    r["stage"],
                    {
                        "error_code": "WORKER_LOST",
                        "error_cause": "lease_expired_max_claims",
                        "retryable": True,
                    },
                )
        return len(rows)

    def claim(self, *, worker_id: str, lease_s: float, max_claims: int) -> tuple[Job, Lease] | None:
        now = self._clock.now()
        claimable = (
            "(status='queued' AND (not_before IS NULL OR not_before <= :now))"
            " OR (status='running' AND lease_until < :now AND lease_claims < :max_claims)"
        )
        params = {"now": now, "max_claims": max_claims, "worker": worker_id, "lease_s": lease_s}
        with self._tx() as conn:
            # The writer lock is held, so this read sees exactly the row the UPDATE will take;
            # it only supplies the previous owner for the lease_reclaimed event.
            previous = conn.execute(
                f"SELECT id, status, worker_id, attempt FROM jobs WHERE {claimable}"
                " ORDER BY created_at LIMIT 1",
                params,
            ).fetchone()
            row = conn.execute(
                "UPDATE jobs SET status='running', worker_id=:worker, lease_until=:now + :lease_s,"
                " attempt=attempt+1, lease_claims=lease_claims+1,"
                " started_at=COALESCE(started_at, :now), updated_at=:now"
                f" WHERE id = (SELECT id FROM jobs WHERE {claimable} ORDER BY created_at LIMIT 1)"
                f" AND ({claimable})"
                " RETURNING *",
                params,
            ).fetchone()
            if row is None:
                return None
            job = _row_to_job(row)
            data: dict[str, Any] = {"worker_id": worker_id}
            event_type = "job_claimed"
            if (
                previous is not None
                and previous["id"] == job.id
                and previous["status"] == "running"
            ):
                event_type = "lease_reclaimed"
                data["previous_worker"] = previous["worker_id"]
                data["previous_attempt"] = previous["attempt"]
            self._insert_event(conn, job.id, job.attempt, now, event_type, None, data)
        return job, Lease(job_id=job.id, worker_id=worker_id, attempt=job.attempt)

    def heartbeat(self, lease: Lease, *, lease_s: float) -> None:
        now = self._clock.now()
        with self._tx() as conn:
            self._fenced_update(
                conn, lease, "lease_until = ?, updated_at = ?", (now + lease_s, now)
            )

    def set_stage(self, lease: Lease, stage: Stage) -> None:
        now = self._clock.now()
        with self._tx() as conn:
            self._fenced_update(conn, lease, "stage = ?, updated_at = ?", (str(stage), now))
            self._insert_event(conn, lease.job_id, lease.attempt, now, "stage_started", stage, None)

    def add_event_fenced(
        self,
        lease: Lease,
        type: str,
        stage: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        now = self._clock.now()
        with self._tx() as conn:
            self._fenced_update(conn, lease, "updated_at = ?", (now,))
            self._insert_event(conn, lease.job_id, lease.attempt, now, type, stage, data)

    def touch_worker(self, worker_id: str) -> None:
        self._conn().execute(
            "INSERT INTO worker_heartbeats(worker_id, last_seen) VALUES (?, ?)"
            " ON CONFLICT(worker_id) DO UPDATE SET last_seen = excluded.last_seen",
            (worker_id, self._clock.now()),
        )

    def worker_heartbeat_age_s(self) -> float | None:
        last = self._conn().execute("SELECT MAX(last_seen) FROM worker_heartbeats").fetchone()[0]
        return None if last is None else self._clock.now() - last

    # --- terminal states, retries, checkpoint pointers (spec §4.6, §6) --------------------------

    def succeed(self, lease: Lease) -> None:
        now = self._clock.now()
        with self._tx() as conn:
            self._fenced_update(
                conn,
                lease,
                "status='succeeded', worker_id=NULL, lease_until=NULL, error_code=NULL,"
                " error_cause=NULL, error_message=NULL, retryable=NULL,"
                " finished_at=?, updated_at=?",
                (now, now),
            )
            self._insert_event(conn, lease.job_id, lease.attempt, now, "job_succeeded", None, None)

    def _fail_in_tx(self, conn: sqlite3.Connection, lease: Lease, error: AppError) -> None:
        now = self._clock.now()
        self._fenced_update(
            conn,
            lease,
            "status='failed', worker_id=NULL, lease_until=NULL, error_code=?, error_cause=?,"
            " error_message=?, retryable=?, finished_at=?, updated_at=?",
            (str(error.code), error.cause, error.message, int(error.retryable), now, now),
        )
        self._insert_event(
            conn,
            lease.job_id,
            lease.attempt,
            now,
            "job_failed",
            None,
            {
                "error_code": str(error.code),
                "error_cause": error.cause,
                "retryable": error.retryable,
            },
        )

    def fail(self, lease: Lease, error: AppError) -> None:
        with self._tx() as conn:
            self._fail_in_tx(conn, lease, error)

    def schedule_retry_or_fail(
        self, lease: Lease, error: AppError, *, delays_s: Sequence[float]
    ) -> JobStatus:
        now = self._clock.now()
        with self._tx() as conn:
            row = conn.execute(
                f"SELECT scheduled_retries FROM jobs WHERE {self._FENCE}",
                (lease.job_id, lease.worker_id, lease.attempt),
            ).fetchone()
            if row is None:
                raise StaleLeaseError(f"lease lost: {lease}")
            used = row["scheduled_retries"]
            if not error.retryable or used >= len(delays_s):
                self._fail_in_tx(conn, lease, error)
                return JobStatus.FAILED
            delay = delays_s[used]
            self._fenced_update(
                conn,
                lease,
                "status='queued', stage=NULL, worker_id=NULL, lease_until=NULL,"
                " lease_claims=0, scheduled_retries=scheduled_retries+1, not_before=?,"
                " error_code=?, error_cause=?, error_message=?, retryable=1, updated_at=?",
                (now + delay, str(error.code), error.cause, error.message, now),
            )
            self._insert_event(
                conn,
                lease.job_id,
                lease.attempt,
                now,
                "retry_scheduled",
                None,
                {
                    "error_code": str(error.code),
                    "error_cause": error.cause,
                    "delay_s": delay,
                    "scheduled_retries": used + 1,
                },
            )
        return JobStatus.QUEUED

    def manual_retry(self, job_id: str) -> Job:
        now = self._clock.now()
        with self._tx() as conn:
            row = conn.execute(
                "UPDATE jobs SET status='queued', stage=NULL, worker_id=NULL, lease_until=NULL,"
                " not_before=NULL, lease_claims=0, scheduled_retries=0, error_code=NULL,"
                " error_cause=NULL, error_message=NULL, retryable=NULL, updated_at=?"
                " WHERE id=? AND status='failed' AND retryable=1 RETURNING *",
                (now, job_id),
            ).fetchone()
            if row is None:
                raise NotRetryable(job_id)
            job = _row_to_job(row)
            self._insert_event(conn, job.id, job.attempt, now, "retry_requested", None, None)
        return job

    def set_generation(
        self, lease: Lease, *, generation_source: str, fallback_reason: str | None
    ) -> None:
        now = self._clock.now()
        with self._tx() as conn:
            self._fenced_update(
                conn,
                lease,
                "generation_source=?, fallback_reason=?, updated_at=?",
                (generation_source, fallback_reason, now),
            )

    def promote_checkpoint(
        self,
        lease: Lease,
        *,
        stage_key: str,
        input_hash: str,
        manifest_sha256: str,
        versions: dict[str, Any],
    ) -> None:
        now = self._clock.now()
        with self._tx() as conn:
            self._fenced_update(conn, lease, "updated_at=?", (now,))
            conn.execute(
                "INSERT OR REPLACE INTO stage_checkpoints(job_id, stage_key, attempt, input_hash,"
                " manifest_sha256, versions_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    lease.job_id,
                    stage_key,
                    lease.attempt,
                    input_hash,
                    manifest_sha256,
                    json.dumps(versions, sort_keys=True),
                    now,
                ),
            )

    @staticmethod
    def _row_to_checkpoint(r: sqlite3.Row) -> CheckpointRow:
        return CheckpointRow(
            job_id=r["job_id"],
            stage_key=r["stage_key"],
            attempt=r["attempt"],
            input_hash=r["input_hash"],
            manifest_sha256=r["manifest_sha256"],
            versions=json.loads(r["versions_json"]),
            created_at=r["created_at"],
        )

    def get_checkpoint(self, job_id: str, stage_key: str) -> CheckpointRow | None:
        r = (
            self._conn()
            .execute(
                "SELECT * FROM stage_checkpoints WHERE job_id = ? AND stage_key = ?",
                (job_id, stage_key),
            )
            .fetchone()
        )
        return self._row_to_checkpoint(r) if r is not None else None

    def list_checkpoints(self, job_id: str) -> builtins.list[CheckpointRow]:
        rows = (
            self._conn()
            .execute(
                "SELECT * FROM stage_checkpoints WHERE job_id = ? ORDER BY stage_key", (job_id,)
            )
            .fetchall()
        )
        return [self._row_to_checkpoint(r) for r in rows]
