"""Cost ledger: all ledger SQL lives here (spec §4.2, §9.3)."""

import builtins
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.cost.pricing import PRICING_VERSION
from app.domain.errors import BudgetRefused
from app.jobs.clock import Clock
from app.jobs.db import apply_schema, connect

LEDGER_SCHEMA_SQL = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")

# Tolerance for float sums of many small amounts (e.g. five 0.01 reservations under a 0.05 cap).
_EPS = 1e-9

# Settled rows count at usd, outstanding reservations at their projection.
_COMMITTED_SQL = (
    "SELECT COALESCE(SUM(CASE WHEN status='reserved' THEN projected_usd ELSE usd END), 0)"
    " FROM cost_entries WHERE billable=1"
)


@dataclass(frozen=True)
class ReservationRequest:
    operation: str
    provider: str
    model: str | None
    projected_usd: float
    job_id: str | None
    attempt: int | None
    stage: str | None


@dataclass(frozen=True)
class Limits:
    hard_project_cap_usd: float
    job_budget_usd: float
    daily_budget_usd: float | None
    day_start_ts: float


@dataclass(frozen=True)
class SpendTotals:
    billable_settled_usd: float
    reserved_usd: float

    @property
    def committed_usd(self) -> float:
        return self.billable_settled_usd + self.reserved_usd


class Ledger:
    def __init__(self, db_path: Path, clock: Clock) -> None:
        self._db_path = db_path
        self._clock = clock
        self._local = threading.local()
        conn = self._conn()
        apply_schema(conn, LEDGER_SCHEMA_SQL)
        # Ledgers created before error_message existed gain the column; rows are never rewritten.
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(cost_entries)")}
        if "error_message" not in columns:
            conn.execute("ALTER TABLE cost_entries ADD COLUMN error_message TEXT")

    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self._db_path)
            self._local.conn = conn
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """One write transaction; BEGIN IMMEDIATE serialises writers across processes."""
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def _insert(
        self,
        conn: sqlite3.Connection,
        *,
        operation: str,
        provider: str,
        model: str | None,
        status: str,
        billable: bool,
        projected_usd: float,
        usd: float | None,
        cost_source: str | None,
        units: dict[str, Any],
        job_id: str | None,
        attempt: int | None,
        stage: str | None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> int:
        now = self._clock.now()
        cur = conn.execute(
            "INSERT INTO cost_entries(job_id, attempt, ts, settled_at, stage, operation, provider,"
            " model, status, billable, projected_usd, usd, cost_source, units_json, error_code,"
            " error_message, pricing_version)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                attempt,
                now,
                None if status == "reserved" else now,
                stage,
                operation,
                provider,
                model,
                status,
                int(billable),
                projected_usd,
                usd,
                cost_source,
                json.dumps(units, sort_keys=True),
                error_code,
                error_message,
                PRICING_VERSION,
            ),
        )
        assert cur.lastrowid is not None
        return cur.lastrowid

    # --- reserve / settle ------------------------------------------------------------------------

    def try_reserve(self, req: ReservationRequest, limits: Limits) -> int:
        with self._tx() as conn:
            project = conn.execute(_COMMITTED_SQL).fetchone()[0]
            if project + req.projected_usd > limits.hard_project_cap_usd + _EPS:
                raise BudgetRefused(
                    "project", req.projected_usd, limits.hard_project_cap_usd, project
                )
            if req.job_id is not None:
                job = conn.execute(_COMMITTED_SQL + " AND job_id=?", (req.job_id,)).fetchone()[0]
                if job + req.projected_usd > limits.job_budget_usd + _EPS:
                    raise BudgetRefused("job", req.projected_usd, limits.job_budget_usd, job)
            if limits.daily_budget_usd is not None:
                day = conn.execute(
                    _COMMITTED_SQL + " AND ts>=?", (limits.day_start_ts,)
                ).fetchone()[0]
                if day + req.projected_usd > limits.daily_budget_usd + _EPS:
                    raise BudgetRefused("daily", req.projected_usd, limits.daily_budget_usd, day)
            return self._insert(
                conn,
                operation=req.operation,
                provider=req.provider,
                model=req.model,
                status="reserved",
                billable=True,
                projected_usd=req.projected_usd,
                usd=None,
                cost_source=None,
                units={},
                job_id=req.job_id,
                attempt=req.attempt,
                stage=req.stage,
            )

    def settle(
        self,
        entry_id: int,
        *,
        status: Literal["ok", "error"],
        usd: float,
        cost_source: Literal["actual", "estimated", "free"],
        units: dict[str, Any],
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE cost_entries SET status=?, usd=?, cost_source=?, units_json=?,"
                " error_code=?, error_message=?, settled_at=? WHERE id=? AND status='reserved'",
                (
                    status,
                    usd,
                    cost_source,
                    json.dumps(units, sort_keys=True),
                    error_code,
                    error_message,
                    self._clock.now(),
                    entry_id,
                ),
            )

    def record_free(
        self,
        *,
        operation: str,
        provider: str,
        model: str | None,
        units: dict[str, Any],
        job_id: str | None,
        attempt: int | None,
        stage: str | None,
        status: str = "ok",
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._tx() as conn:
            self._insert(
                conn,
                operation=operation,
                provider=provider,
                model=model,
                status=status,
                billable=True,
                projected_usd=0.0,
                usd=0.0,
                cost_source="free",
                units=units,
                job_id=job_id,
                attempt=attempt,
                stage=stage,
                error_code=error_code,
                error_message=error_message,
            )

    def record_production_estimate(
        self,
        *,
        operation: str,
        usd: float,
        units: dict[str, Any],
        job_id: str,
        attempt: int,
        stage: str,
    ) -> None:
        """billable=0: reported per video, never counted toward any budget."""
        with self._tx() as conn:
            self._insert(
                conn,
                operation=operation,
                provider="local",
                model=None,
                status="ok",
                billable=False,
                projected_usd=usd,
                usd=usd,
                cost_source="estimated",
                units=units,
                job_id=job_id,
                attempt=attempt,
                stage=stage,
            )

    # --- totals ----------------------------------------------------------------------------------

    def _totals(self, where: str, params: tuple[Any, ...]) -> SpendTotals:
        row = (
            self._conn()
            .execute(
                "SELECT COALESCE(SUM(CASE WHEN status!='reserved' THEN usd END), 0),"
                " COALESCE(SUM(CASE WHEN status='reserved' THEN projected_usd END), 0)"
                " FROM cost_entries WHERE billable=1" + where,
                params,
            )
            .fetchone()
        )
        return SpendTotals(billable_settled_usd=row[0], reserved_usd=row[1])

    def project_totals(self) -> SpendTotals:
        return self._totals("", ())

    def job_totals(self, job_id: str) -> SpendTotals:
        return self._totals(" AND job_id=?", (job_id,))

    def day_totals(self, day_start_ts: float) -> SpendTotals:
        return self._totals(" AND ts>=?", (day_start_ts,))

    # --- once-only marks, reports ----------------------------------------------------------------

    def mark_once(self, key: str, value: str) -> bool:
        with self._tx() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO budget_state(key, value, ts) VALUES (?, ?, ?)",
                (key, value, self._clock.now()),
            )
            return cur.rowcount == 1

    def has_mark(self, key: str) -> bool:
        return (
            self._conn().execute("SELECT 1 FROM budget_state WHERE key=?", (key,)).fetchone()
            is not None
        )

    def unsettled(self) -> builtins.list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM cost_entries WHERE status='reserved' ORDER BY id"
        )
        return [dict(r) for r in rows]

    def call_counts(self) -> dict[str, int]:
        """Rows per status over the whole ledger: every provider call, failed ones included."""
        counts = {"ok": 0, "error": 0, "reserved": 0}
        for row in self._conn().execute(
            "SELECT status, COUNT(*) AS n FROM cost_entries WHERE billable=1 GROUP BY status"
        ):
            counts[row["status"]] = row["n"]
        return counts

    def job_cost_summary(self, job_id: str) -> dict[str, Any]:
        """Shape of the API's CostOut."""
        return self._cost_summary(" AND job_id=?", (job_id,))

    def project_cost_summary(self) -> dict[str, Any]:
        """Same shape as job_cost_summary, over every row in the ledger."""
        return self._cost_summary("", ())

    def _cost_summary(self, where: str, params: tuple[Any, ...]) -> dict[str, Any]:
        rows = (
            self._conn()
            .execute(
                "SELECT * FROM cost_entries WHERE status!='reserved'" + where + " ORDER BY id",
                params,
            )
            .fetchall()
        )
        summary: dict[str, Any] = {
            "billable_usd": 0.0,
            "billable_actual_usd": 0.0,
            "billable_estimated_usd": 0.0,
            "failed_call_usd": 0.0,
            "free_calls": 0,
            "estimated_production_compute_usd": 0.0,
            "by_operation": {},
            "pricing_version": rows[-1]["pricing_version"] if rows else None,
        }
        for r in rows:
            usd = r["usd"] or 0.0
            if not r["billable"]:
                summary["estimated_production_compute_usd"] += usd
                continue
            summary["billable_usd"] += usd
            if r["cost_source"] == "actual":
                summary["billable_actual_usd"] += usd
            elif r["cost_source"] == "estimated":
                summary["billable_estimated_usd"] += usd
            else:
                summary["free_calls"] += 1
            if r["status"] == "error":
                summary["failed_call_usd"] += usd
            summary["by_operation"][r["operation"]] = (
                summary["by_operation"].get(r["operation"], 0.0) + usd
            )
        return summary
