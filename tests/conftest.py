"""Shared pytest fixtures."""

from pathlib import Path

import pytest

from app.jobs.clock import FakeClock


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "app.sqlite"
