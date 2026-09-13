"""Context variables that tie log lines to a job, attempt, stage and worker (spec §9.4)."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

job_id_var: ContextVar[str | None] = ContextVar("job_id", default=None)
attempt_var: ContextVar[int | None] = ContextVar("attempt", default=None)
stage_var: ContextVar[str | None] = ContextVar("stage", default=None)
worker_id_var: ContextVar[str | None] = ContextVar("worker_id", default=None)


@contextmanager
def job_context(*, job_id: str, attempt: int, worker_id: str) -> Iterator[None]:
    tokens = (job_id_var.set(job_id), attempt_var.set(attempt), worker_id_var.set(worker_id))
    try:
        yield
    finally:
        job_id_var.reset(tokens[0])
        attempt_var.reset(tokens[1])
        worker_id_var.reset(tokens[2])


@contextmanager
def stage_context(stage: str) -> Iterator[None]:
    token = stage_var.set(stage)
    try:
        yield
    finally:
        stage_var.reset(token)
