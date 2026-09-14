import contextlib
import json
from pathlib import Path

import pytest

from app.config import Settings
from app.cost import report
from app.cost.budget import Settlement, SpendGuard
from app.cost.ledger import Ledger
from app.jobs.clock import SystemClock


def _spend(guard: SpendGuard, operation: str, usd: float, status: str = "ok") -> None:
    def settle(result: object, exc: BaseException | None) -> Settlement:
        if exc is not None:
            return Settlement("error", usd, "estimated", {}, "invalid_request", "400")
        return Settlement("ok", usd, "actual", {})

    def call() -> None:
        if status == "error":
            raise RuntimeError("rejected")

    with contextlib.suppress(RuntimeError):
        guard.run(
            operation=operation,
            provider="gemini",
            model="gemini-2.5-flash",
            projected_usd=0.013,
            call=call,
            settle=settle,
        )


def test_report_billable_usd_is_project_cumulative_and_labelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "ledger.sqlite"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.ledger_path() == db
    # two separate "runs" against the same ledger, like two live tests on different days
    for run in (["writer_error", "writer"], ["writer", "critic"]):
        guard = SpendGuard(Ledger(db, SystemClock()), settings, SystemClock())
        for op in run:
            if op == "writer_error":
                _spend(guard, "writer", 0.0, status="error")
            else:
                _spend(guard, op, {"writer": 0.004, "critic": 0.0001}[op])

    report.main()
    out = json.loads(capsys.readouterr().out)

    assert out["scope"].startswith("project")
    assert out["billable_usd"] == pytest.approx(0.004 + 0.004 + 0.0001)  # both runs, not the last
    breakdown = out["breakdown"]
    assert breakdown["billable_usd"] == pytest.approx(out["billable_usd"])
    assert breakdown["billable_actual_usd"] + breakdown["billable_estimated_usd"] == pytest.approx(
        out["billable_usd"]
    )
    assert breakdown["by_operation"] == {
        "writer": pytest.approx(0.008),
        "critic": pytest.approx(0.0001),
    }
    assert out["calls"] == {"ok": 3, "error": 1, "reserved": 0}
    assert out["reserved_usd"] == 0
    assert out["unsettled"] == []
