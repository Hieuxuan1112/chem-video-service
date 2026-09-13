import socket
import threading
import time
from pathlib import Path

import pytest

from app.config import Settings
from app.domain.errors import AppError, ErrorCode, StaleLeaseError
from app.domain.job import Job, JobStatus, Lease, Stage
from app.jobs.clock import FakeClock, SystemClock
from app.jobs.repository import JobRepository
from app.jobs.worker import JobContext, LeaseKeeper, Worker
from app.observability.context import job_id_var


def _settings(db_path: Path, **kw: object) -> Settings:
    return Settings(_env_file=None, database_path=db_path, **kw)  # type: ignore[arg-type]


def _new_job(repo: JobRepository) -> Job:
    job, _ = repo.create(
        query="How does the pH scale work?",
        topic_id="ph_scale",
        pipeline_version="1.0.0",
        idempotency_key=None,
        request_fingerprint=None,
    )
    return job


class RecordingHandler:
    def __init__(self, action=None) -> None:  # type: ignore[no-untyped-def]
        self.action = action
        self.contexts: list[JobContext] = []

    def handle(self, ctx: JobContext) -> None:
        self.contexts.append(ctx)
        if self.action is not None:
            self.action(ctx)


def _get(repo: JobRepository, job_id: str) -> Job:
    job = repo.get(job_id)
    assert job is not None
    return job


def test_success_path_marks_succeeded(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    job = _new_job(repo)
    seen_job_ids: list[str | None] = []

    def action(ctx: JobContext) -> None:
        seen_job_ids.append(job_id_var.get())
        ctx.set_stage(Stage.SCRIPTING)
        ctx.emit("writer_attempt", stage="scripting", data={"attempt": 1})
        ctx.renew_lease()

    handler = RecordingHandler(action)
    worker = Worker(repo, handler, _settings(db_path), worker_id="w-1")
    assert worker.run_once() is True
    done = _get(repo, job.id)
    assert done.status is JobStatus.SUCCEEDED
    assert seen_job_ids == [job.id]
    assert handler.contexts[0].lease == Lease(job.id, "w-1", 1)
    types = [e.type for e in repo.events(job.id)]
    assert types == [
        "job_created",
        "job_claimed",
        "stage_started",
        "writer_attempt",
        "job_succeeded",
    ]
    assert repo.worker_heartbeat_age_s() == 0.0
    assert worker.run_once() is False


def test_retryable_app_error_schedules_retry(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    job = _new_job(repo)

    def action(ctx: JobContext) -> None:
        raise AppError(ErrorCode.RENDER_FAILED, "subprocess_crash", "crashed")

    Worker(repo, RecordingHandler(action), _settings(db_path)).run_once()
    queued = _get(repo, job.id)
    assert queued.status is JobStatus.QUEUED
    assert (queued.scheduled_retries, queued.error_code) == (1, "RENDER_FAILED")


def test_non_retryable_fails(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    job = _new_job(repo)

    def action(ctx: JobContext) -> None:
        raise AppError(ErrorCode.RENDER_LAYOUT_ERROR, "deterministic_bug", "overflow")

    Worker(repo, RecordingHandler(action), _settings(db_path)).run_once()
    failed = _get(repo, job.id)
    assert (failed.status, failed.error_code, failed.retryable) == (
        JobStatus.FAILED,
        "RENDER_LAYOUT_ERROR",
        False,
    )


def test_unexpected_exception_is_internal_error(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    job = _new_job(repo)

    def action(ctx: JobContext) -> None:
        raise ZeroDivisionError("oops")

    Worker(repo, RecordingHandler(action), _settings(db_path)).run_once()
    failed = _get(repo, job.id)
    assert (failed.status, failed.error_code, failed.error_cause, failed.retryable) == (
        JobStatus.FAILED,
        "INTERNAL_ERROR",
        "unclassified",
        False,
    )
    assert "ZeroDivisionError" in (failed.error_message or "")


def test_lease_keeper_extends_lease(db_path: Path) -> None:
    clock = SystemClock()
    repo = JobRepository(db_path, clock)
    job = _new_job(repo)
    rival = JobRepository(db_path, clock)
    rival_claims: list[object] = []

    def action(ctx: JobContext) -> None:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            rival_claims.append(rival.claim(worker_id="rival", lease_s=0.6, max_claims=3))
            time.sleep(0.1)
        ctx.check()

    settings = _settings(db_path, lease_s=0.6, heartbeat_interval_s=0.1)
    Worker(repo, RecordingHandler(action), settings, worker_id="holder").run_once()
    assert rival_claims and all(c is None for c in rival_claims)
    done = _get(repo, job.id)
    assert done.status is JobStatus.SUCCEEDED
    assert [e.type for e in repo.events(job.id)].count("lease_reclaimed") == 0


def test_lease_keeper_sets_lost_on_stale_lease(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    _new_job(repo)
    stale = Lease("vid_missing", "nobody", 1)
    lost = threading.Event()
    with LeaseKeeper(repo, stale, lease_s=60.0, interval_s=0.02, lost=lost):
        assert lost.wait(timeout=2.0)


def _reclaim_by_rival(db_path: Path, clock: FakeClock) -> None:
    clock.advance(61)
    rival = JobRepository(db_path, clock)
    assert rival.claim(worker_id="rival", lease_s=60.0, max_claims=3) is not None


@pytest.mark.parametrize("ending", ["stale_write", "return_normally", "raise_app_error"])
def test_lost_lease_abandons_without_writes(db_path: Path, clock: FakeClock, ending: str) -> None:
    repo = JobRepository(db_path, clock)
    job = _new_job(repo)
    snapshot: dict[str, object] = {}

    def action(ctx: JobContext) -> None:
        _reclaim_by_rival(db_path, clock)
        snapshot["job"] = repo.get(job.id)
        snapshot["events"] = repo.events(job.id)
        if ending == "stale_write":
            ctx.set_stage(Stage.VOICING)  # raises StaleLeaseError and marks the lease lost
        elif ending == "return_normally":
            with pytest.raises(StaleLeaseError):
                ctx.emit("stage_completed")
        else:
            ctx.lease_lost.set()
            raise AppError(ErrorCode.RENDER_FAILED, "subprocess_crash", "late failure")

    Worker(repo, RecordingHandler(action), _settings(db_path), worker_id="holder").run_once()
    after = _get(repo, job.id)
    assert after == snapshot["job"]
    assert after.status is JobStatus.RUNNING
    assert after.worker_id == "rival"
    assert repo.events(job.id) == snapshot["events"]


def test_run_forever_stops_on_event(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    worker = Worker(repo, RecordingHandler(), _settings(db_path, poll_interval_s=0.05))
    stop = threading.Event()
    thread = threading.Thread(target=worker.run_forever, args=(stop,))
    thread.start()
    time.sleep(0.2)
    job = _new_job(repo)
    deadline = time.monotonic() + 3.0
    while _get(repo, job.id).status is not JobStatus.SUCCEEDED and time.monotonic() < deadline:
        time.sleep(0.02)
    stop.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert _get(repo, job.id).status is JobStatus.SUCCEEDED


def test_worker_id_format(db_path: Path, clock: FakeClock) -> None:
    worker = Worker(JobRepository(db_path, clock), RecordingHandler(), _settings(db_path))
    host, pid, suffix = worker.worker_id.rsplit(":", 2)
    assert host == socket.gethostname()
    assert pid.isdigit()
    assert len(suffix) == 8
