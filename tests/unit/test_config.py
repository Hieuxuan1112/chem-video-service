from pathlib import Path

import pytest

from app.config import Settings


def test_defaults_match_spec() -> None:
    s = Settings(_env_file=None)
    assert s.warning_budget_usd == 3.00
    assert s.hard_project_cap_usd == 5.00
    assert s.cost_job_budget_usd == 0.20
    assert s.cost_daily_budget_usd is None
    assert s.a5_wer_threshold is None
    assert s.max_claims == 3


def test_ledger_path_defaults_to_database_path(tmp_path: Path) -> None:
    db = tmp_path / "jobs.sqlite"
    s = Settings(_env_file=None, database_path=db)
    assert s.ledger_path() == db
    ledger = tmp_path / "ledger.sqlite"
    assert Settings(_env_file=None, database_path=db, ledger_db_path=ledger).ledger_path() == ledger


def test_is_project_ledger_false_for_tmp_path(tmp_path: Path) -> None:
    s = Settings(_env_file=None, database_path=tmp_path / "app.sqlite")
    assert s.is_project_ledger() is False


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARD_PROJECT_CAP_USD", "1.5")
    assert Settings(_env_file=None).hard_project_cap_usd == 1.5
