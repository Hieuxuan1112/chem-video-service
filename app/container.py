"""Dependency container: plain constructor wiring, no registry."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings
from app.jobs.clock import Clock
from app.jobs.repository import JobRepository
from app.jobs.worker import Worker
from app.topics.resolver import TopicResolver


class ArtifactStoreLike(Protocol):
    def open(self, job_id: str, name: str) -> Path | None: ...

    def list_refs(self, job_id: str) -> list[dict[str, Any]]: ...

    def manifest(self, job_id: str) -> dict[str, Any] | None: ...


@dataclass
class Container:
    settings: Settings
    clock: Clock
    repo: JobRepository
    resolver: TopicResolver
    artifacts: ArtifactStoreLike | None = None
    spend_state: Callable[[], dict[str, Any]] | None = None
    job_cost: Callable[[str], dict[str, Any]] | None = None
    worker: Worker | None = None
