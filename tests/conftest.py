"""Shared pytest fixtures."""

from pathlib import Path

import pytest

from app.jobs.clock import FakeClock
from app.render.layout import FontSet

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "app.sqlite"


@pytest.fixture(scope="session")
def fonts() -> FontSet:
    return FontSet.from_dir(REPO_ROOT / "assets" / "fonts")
