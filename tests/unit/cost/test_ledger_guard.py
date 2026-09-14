import threading
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.cost.budget import Settlement, SpendGuard, assert_project_ledger, settle_text
from app.cost.ledger import Ledger
from app.cost.pricing import PRICING_VERSION, estimate_text_tokens, text_call_cost
from app.domain.errors import BudgetRefused
from app.jobs.clock import FakeClock
from app.observability.context import job_context, stage_context
from app.providers.errors import ProviderError
from app.providers.retry import RetryPolicy, call_with_retry

MODEL = "gemini-2.5-flash"


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


class Usage:
    def __init__(self, input_tokens: int, output_tokens: int, thinking_tokens: int = 0) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.thinking_tokens = thinking_tokens


class Response:
    def __init__(self, text: str, usage: Usage | None) -> None:
        self.text = text
        self.usage = usage


def _fixed(usd: float) -> Any:
    def settle(result: object, exc: BaseException | None) -> Settlement:
        if exc is not None:
            return Settlement("error", 0.0, "estimated", {}, error_code=type(exc).__name__)
        return Settlement("ok", usd, "actual", {"input_tokens": 1})

    return settle


def _rows(ledger: Ledger) -> list[dict[str, Any]]:
    conn = ledger._conn()
    return [dict(r) for r in conn.execute("SELECT * FROM cost_entries ORDER BY id")]


@pytest.fixture
def ledger(db_path: Path, clock: FakeClock) -> Ledger:
    return Ledger(db_path, clock)


def test_reserve_row_exists_when_provider_called(ledger: Ledger, clock: FakeClock) -> None:
    guard = SpendGuard(ledger, _settings(), clock)
    seen: list[dict[str, Any]] = []

    def call() -> str:
        seen.extend(_rows(ledger))
        return "ok"

    with job_context(job_id="job1", attempt=2, worker_id="w"), stage_context("storyboard"):
        guard.run(
            operation="writer",
            provider="gemini",
            model=MODEL,
            projected_usd=0.01,
            call=call,
            settle=_fixed(0.004),
        )

    assert len(seen) == 1
    assert seen[0]["status"] == "reserved"
    assert seen[0]["usd"] is None
    assert (seen[0]["job_id"], seen[0]["attempt"], seen[0]["stage"]) == ("job1", 2, "storyboard")
    row = _rows(ledger)[0]
    assert row["status"] == "ok"
    assert row["pricing_version"] == PRICING_VERSION
    assert row["billable"] == 1
    assert row["settled_at"] is not None


def test_settle_success_actual(ledger: Ledger, clock: FakeClock) -> None:
    guard = SpendGuard(ledger, _settings(), clock)
    resp = Response('{"a": 1}', Usage(1200, 300, 50))
    guard.run(
        operation="writer",
        provider="gemini",
        model=MODEL,
        projected_usd=0.02,
        call=lambda: resp,
        settle=settle_text(MODEL, prompt_chars=4800),
    )
    row = _rows(ledger)[0]
    assert row["status"] == "ok"
    assert row["cost_source"] == "actual"
    assert row["usd"] == pytest.approx(text_call_cost(MODEL, 1200, 300, 50))
    assert '"input_tokens": 1200' in row["units_json"]


def test_settle_missing_usage_estimated(ledger: Ledger, clock: FakeClock) -> None:
    guard = SpendGuard(ledger, _settings(), clock)
    resp = Response("x" * 400, None)
    guard.run(
        operation="writer",
        provider="gemini",
        model=MODEL,
        projected_usd=0.02,
        call=lambda: resp,
        settle=settle_text(MODEL, prompt_chars=4800),
    )
    row = _rows(ledger)[0]
    assert row["cost_source"] == "estimated"
    assert row["usd"] == pytest.approx(text_call_cost(MODEL, 1200, 100, 0))


def test_failed_call_settled_as_error_with_rules(ledger: Ledger, clock: FakeClock) -> None:
    guard = SpendGuard(ledger, _settings(), clock)
    for err in (
        ProviderError("rate_limited", "429", status=429),
        ProviderError("timeout", "timed out"),
        ProviderError("server", "503", status=503),
    ):
        with pytest.raises(ProviderError):
            guard.run(
                operation="writer",
                provider="gemini",
                model=MODEL,
                projected_usd=0.02,
                call=lambda err=err: (_ for _ in ()).throw(err),  # type: ignore[misc]
                settle=settle_text(MODEL, prompt_chars=4000),
            )
    rate, timeout, server = _rows(ledger)
    assert rate["error_message"] == "429"
    assert timeout["error_message"] == "timed out"
    input_estimate = text_call_cost(MODEL, estimate_text_tokens(4000), 0, 0)
    assert (rate["status"], rate["usd"], rate["cost_source"]) == ("error", 0.0, "estimated")
    assert rate["error_code"] == "rate_limited"
    assert timeout["status"] == "error"
    assert timeout["usd"] == pytest.approx(input_estimate)
    assert timeout["error_code"] == "timeout"
    assert server["usd"] == pytest.approx(input_estimate)


def test_failed_call_logs_provider_message(
    ledger: Ledger, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    guard = SpendGuard(ledger, _settings(), clock)
    message = "400 INVALID_ARGUMENT: schema has too many states " + "x" * 3000
    err = ProviderError("invalid_request", message, status=400, provider="gemini")
    with caplog.at_level("WARNING", logger="app.cost.budget"), pytest.raises(ProviderError):
        guard.run(
            operation="writer",
            provider="gemini",
            model=MODEL,
            projected_usd=0.02,
            call=lambda: (_ for _ in ()).throw(err),
            settle=settle_text(MODEL, prompt_chars=40),
        )
    [record] = [r for r in caplog.records if r.getMessage() == "provider_call_failed"]
    data = record.data  # type: ignore[attr-defined]
    assert data["error_code"] == "invalid_request"
    assert data["status"] == 400
    assert data["error_message"].startswith("400 INVALID_ARGUMENT: schema has too many states")
    row = _rows(ledger)[0]
    assert row["error_message"].startswith("400 INVALID_ARGUMENT")
    assert len(row["error_message"]) <= 1000


def test_each_retry_is_its_own_row(ledger: Ledger, clock: FakeClock) -> None:
    guard = SpendGuard(ledger, _settings(), clock)
    outcomes: list[object] = [ProviderError("rate_limited", "429", status=429), "ok"]

    def call() -> Response:
        outcome = outcomes.pop(0)
        if isinstance(outcome, ProviderError):
            raise outcome
        return Response("{}", Usage(10, 10))

    call_with_retry(
        lambda: guard.run(
            operation="writer",
            provider="gemini",
            model=MODEL,
            projected_usd=0.01,
            call=call,
            settle=settle_text(MODEL, prompt_chars=40),
        ),
        policy=RetryPolicy(),
        sleep=lambda _: None,
    )
    assert [r["status"] for r in _rows(ledger)] == ["error", "ok"]


def test_job_budget_refusal_at_020(ledger: Ledger, clock: FakeClock) -> None:
    guard = SpendGuard(ledger, _settings(), clock)
    calls: list[int] = []
    with job_context(job_id="job1", attempt=1, worker_id="w"):
        for _ in range(3):
            guard.run(
                operation="tts",
                provider="gemini",
                model=MODEL,
                projected_usd=0.06,
                call=lambda: calls.append(1),
                settle=_fixed(0.06),
            )
        assert guard.headroom(job_id="job1")["job_usd"] == pytest.approx(0.02)
        with pytest.raises(BudgetRefused) as info:
            guard.run(
                operation="tts",
                provider="gemini",
                model=MODEL,
                projected_usd=0.03,
                call=lambda: calls.append(1),
                settle=_fixed(0.03),
            )
    assert info.value.scope == "job"
    assert info.value.limit_usd == 0.20
    assert len(calls) == 3
    assert len(_rows(ledger)) == 3
    # another job still has its own budget
    with job_context(job_id="job2", attempt=1, worker_id="w"):
        guard.run(
            operation="tts",
            provider="gemini",
            model=MODEL,
            projected_usd=0.03,
            call=lambda: calls.append(1),
            settle=_fixed(0.03),
        )
    assert len(calls) == 4


def test_hard_cap_refusal_counts_reservations(ledger: Ledger, clock: FakeClock) -> None:
    guard = SpendGuard(ledger, _settings(hard_project_cap_usd=0.10), clock)
    inside: list[bool] = []

    def call() -> None:
        # while this call holds a 0.08 reservation, a nested 0.03 call must be refused
        with pytest.raises(BudgetRefused) as info:
            guard.run(
                operation="critic",
                provider="gemini",
                model=MODEL,
                projected_usd=0.03,
                call=lambda: inside.append(True),
                settle=_fixed(0.0),
            )
        assert info.value.scope == "project"
        assert info.value.current_usd == pytest.approx(0.08)

    guard.run(
        operation="writer",
        provider="gemini",
        model=MODEL,
        projected_usd=0.08,
        call=call,
        settle=_fixed(0.01),
    )
    assert inside == []
    guard.run(
        operation="critic",
        provider="gemini",
        model=MODEL,
        projected_usd=0.03,
        call=lambda: inside.append(True),
        settle=_fixed(0.0),
    )
    assert inside == [True]


def test_parallel_reservations_never_exceed_cap(db_path: Path, clock: FakeClock) -> None:
    ledger = Ledger(db_path, clock)
    guard = SpendGuard(ledger, _settings(hard_project_cap_usd=0.05), clock)
    successes: list[int] = []
    errors: list[BaseException] = []
    lock = threading.Lock()
    start = threading.Barrier(8)

    def worker() -> None:
        start.wait()
        for _ in range(10):
            try:
                guard.run(
                    operation="writer",
                    provider="gemini",
                    model=MODEL,
                    projected_usd=0.01,
                    call=lambda: None,
                    settle=_fixed(0.01),
                )
            except BudgetRefused:
                continue
            except Exception as exc:  # e.g. "database is locked" without BEGIN IMMEDIATE
                with lock:
                    errors.append(exc)
                continue
            with lock:
                successes.append(1)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    totals = ledger.project_totals()
    assert totals.committed_usd <= 0.05 + 1e-9
    assert len(successes) == 5


def test_warning_emitted_once_across_two_ledgers_same_file(db_path: Path, clock: FakeClock) -> None:
    settings = _settings(warning_budget_usd=0.05)
    events: list[tuple[str, dict[str, Any]]] = []
    guards = [
        SpendGuard(
            Ledger(db_path, clock), settings, clock, events=lambda t, d: events.append((t, d))
        )
        for _ in range(2)
    ]
    for i in range(6):
        guards[i % 2].run(
            operation="writer",
            provider="gemini",
            model=MODEL,
            projected_usd=0.02,
            call=lambda: None,
            settle=_fixed(0.02),
        )
    assert [t for t, _ in events] == ["project_budget_warning"]
    assert guards[0].state()["state"] == "warning"
    assert guards[1].state()["state"] == "warning"


def test_unsettled_reservation_counts_forever(ledger: Ledger, clock: FakeClock) -> None:
    from app.cost.ledger import Limits, ReservationRequest

    limits = Limits(
        hard_project_cap_usd=0.05, job_budget_usd=0.20, daily_budget_usd=None, day_start_ts=0.0
    )
    ledger.try_reserve(
        ReservationRequest("writer", "gemini", MODEL, 0.04, job_id=None, attempt=None, stage=None),
        limits,
    )  # the process "crashes" here: never settled
    clock.advance(365 * 86400)
    assert ledger.project_totals().reserved_usd == pytest.approx(0.04)
    assert len(ledger.unsettled()) == 1
    with pytest.raises(BudgetRefused):
        ledger.try_reserve(
            ReservationRequest("writer", "gemini", MODEL, 0.02, None, None, None), limits
        )
    reopened = Ledger(ledger._db_path, clock)
    assert reopened.project_totals().committed_usd == pytest.approx(0.04)


def test_production_estimate_not_billable_and_excluded_from_budgets(
    ledger: Ledger, clock: FakeClock
) -> None:
    guard = SpendGuard(ledger, _settings(hard_project_cap_usd=0.05), clock)
    ledger.record_production_estimate(
        operation="render_compute",
        usd=10.0,
        units={"cpu_seconds": 1e6},
        job_id="job1",
        attempt=1,
        stage="render",
    )
    ledger.record_free(
        operation="tts",
        provider="edge",
        model=None,
        units={"characters": 20},
        job_id="job1",
        attempt=1,
        stage="voicing",
    )
    assert ledger.project_totals().committed_usd == 0.0
    assert ledger.job_totals("job1").committed_usd == 0.0
    with job_context(job_id="job1", attempt=1, worker_id="w"):
        guard.run(
            operation="writer",
            provider="gemini",
            model=MODEL,
            projected_usd=0.05,
            call=lambda: None,
            settle=_fixed(0.03),
        )
    rows = _rows(ledger)
    assert rows[0]["billable"] == 0
    assert rows[0]["cost_source"] == "estimated"
    summary = ledger.job_cost_summary("job1")
    assert summary["billable_usd"] == pytest.approx(0.03)
    assert summary["billable_actual_usd"] == pytest.approx(0.03)
    assert summary["estimated_production_compute_usd"] == pytest.approx(10.0)
    assert summary["free_calls"] == 1
    assert summary["by_operation"] == {"writer": pytest.approx(0.03), "tts": 0.0}
    assert summary["pricing_version"] == PRICING_VERSION


def test_daily_guard_disabled_by_default_and_enforced_when_set(
    db_path: Path, clock: FakeClock
) -> None:
    def spend(guard: SpendGuard) -> None:
        guard.run(
            operation="writer",
            provider="gemini",
            model=MODEL,
            projected_usd=0.02,
            call=lambda: None,
            settle=_fixed(0.02),
        )

    ledger = Ledger(db_path, clock)
    default_guard = SpendGuard(ledger, _settings(), clock)
    assert _settings().cost_daily_budget_usd is None
    for _ in range(3):
        spend(default_guard)
    assert default_guard.headroom(job_id=None)["daily_usd"] is None

    daily_guard = SpendGuard(ledger, _settings(cost_daily_budget_usd=0.07), clock)
    with pytest.raises(BudgetRefused) as info:
        spend(daily_guard)  # 0.06 already spent today
    assert info.value.scope == "daily"
    clock.advance(86400)  # next UTC day
    spend(daily_guard)


def test_assert_project_ledger_rejects_tmp(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        assert_project_ledger(_settings(database_path=tmp_path / "scratch.sqlite"))
    assert_project_ledger(_settings())  # defaults resolve to data/app.sqlite


def test_existing_ledger_gains_error_message_column(db_path: Path, clock: FakeClock) -> None:
    import sqlite3

    from app.cost.ledger import LEDGER_SCHEMA_SQL

    old_schema = LEDGER_SCHEMA_SQL.replace("  error_message TEXT,\n", "")
    assert "error_message" not in old_schema
    conn = sqlite3.connect(db_path)
    conn.executescript(old_schema)
    conn.execute(
        "INSERT INTO cost_entries(ts, operation, provider, status, billable, projected_usd, usd,"
        " cost_source, error_code, pricing_version)"
        " VALUES (1, 'writer', 'gemini', 'error', 1, 0.013, 0.0, 'estimated', 'invalid_request',"
        " '2026-09-13')"
    )
    conn.commit()
    conn.close()

    ledger = Ledger(db_path, clock)
    Ledger(db_path, clock)  # idempotent on a second open
    [row] = _rows(ledger)
    assert row["error_code"] == "invalid_request"
    assert row["error_message"] is None
    assert ledger.project_totals().billable_settled_usd == 0.0
