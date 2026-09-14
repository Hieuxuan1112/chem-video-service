"""Paid live tests run only with RUN_LIVE=1, a key, and the persistent project ledger."""

import os

import pytest

from app.config import Settings
from app.cost.budget import SpendGuard, assert_project_ledger
from app.cost.ledger import Ledger
from app.jobs.clock import SystemClock


@pytest.fixture
def live_settings() -> Settings:
    if os.environ.get("RUN_LIVE") != "1":
        pytest.skip("paid live test: set RUN_LIVE=1")
    settings = Settings()
    if settings.google_api_key is None:
        pytest.skip("GOOGLE_API_KEY is not set")
    if not settings.is_project_ledger():
        pytest.skip(f"ledger {settings.ledger_path()} is not the project ledger")
    assert_project_ledger(settings)
    return settings


@pytest.fixture
def live_ledger(live_settings: Settings) -> Ledger:
    return Ledger(live_settings.ledger_path(), SystemClock())


@pytest.fixture
def live_guard(live_settings: Settings, live_ledger: Ledger) -> SpendGuard:
    return SpendGuard(live_ledger, live_settings, SystemClock())
