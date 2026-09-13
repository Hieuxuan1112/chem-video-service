import threading
from pathlib import Path

import pytest

from app.domain.errors import AppError, ErrorCode, NotRetryable, StaleLeaseError
from app.domain.job import Job, JobStatus, Lease
from app.jobs.clock import FakeClock
from app.jobs.repository import JobRepository

LEASE_S = 60.0
MAX_CLAIMS = 3
DELAYS = (60.0, 300.0)
RETRYABLE = AppError(ErrorCode.RENDER_FAILED, "subprocess_crash", "renderer crashed")
NON_RETRYABLE = AppError(ErrorCode.RENDER_TEMPLATE_ERROR, "deterministic_bug", "template bug")


def _new_job(repo: JobRepository) -> Job:
    job, _ = repo.create(
        query="Why do atoms form covalent bonds?",
        topic_id="covalent_bonds",
        pipeline_version="1.0.0",
        idempotency_key=None,
        request_fingerprint=None,
    )
    return job


def _claim(repo: JobRepository, worker: str = "w") -> Lease:
    res = repo.claim(worker_id=worker, lease_s=LEASE_S, max_claims=MAX_CLAIMS)
    assert res is not None
    return res[1]


def _get(repo: JobRepository, job_id: str) -> Job:
    job = repo.get(job_id)
    assert job is not None
    return job


def _fail_retryable_to_terminal(repo: JobRepository, clock: FakeClock) -> Job:
    job = _new_job(repo)
    assert repo.schedule_retry_or_fail(_claim(repo), RETRYABLE, delays_s=DELAYS) is JobStatus.QUEUED
    clock.advance(DELAYS[0])
    assert repo.schedule_retry_or_fail(_claim(repo), RETRYABLE, delays_s=DELAYS) is JobStatus.QUEUED
    clock.advance(DELAYS[1])
    assert repo.schedule_retry_or_fail(_claim(repo), RETRYABLE, delays_s=DELAYS) is JobStatus.FAILED
    return _get(repo, job.id)


def test_succeed_and_fail_are_fenced(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    ok_job = _new_job(repo)
    lease = _claim(repo, "A")
    stale = Lease(ok_job.id, "someone-else", lease.attempt)
    with pytest.raises(StaleLeaseError):
        repo.succeed(stale)
    clock.advance(5)
    repo.succeed(lease)
    done = _get(repo, ok_job.id)
    assert done.status is JobStatus.SUCCEEDED
    assert (done.finished_at, done.lease_until, done.worker_id) == (clock.now(), None, None)
    assert repo.events(ok_job.id)[-1].type == "job_succeeded"
    with pytest.raises(StaleLeaseError):
        repo.succeed(lease)  # not running any more

    bad_job = _new_job(repo)
    lease2 = _claim(repo, "A")
    with pytest.raises(StaleLeaseError):
        repo.fail(Lease(bad_job.id, "A", lease2.attempt + 1), NON_RETRYABLE)
    repo.fail(lease2, NON_RETRYABLE)
    failed = _get(repo, bad_job.id)
    assert failed.status is JobStatus.FAILED
    assert (failed.error_code, failed.error_cause, failed.error_message, failed.retryable) == (
        "RENDER_TEMPLATE_ERROR",
        "deterministic_bug",
        "template bug",
        False,
    )
    ev = repo.events(bad_job.id)[-1]
    assert ev.type == "job_failed"
    assert ev.data["error_code"] == "RENDER_TEMPLATE_ERROR"


def test_scheduled_retry_resets_lease_claims_keeps_attempt(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    job = _new_job(repo)
    _claim(repo, "A")
    clock.advance(LEASE_S + 1)
    lease = _claim(repo, "B")  # reclaim: attempt 2, lease_claims 2
    assert repo.schedule_retry_or_fail(lease, RETRYABLE, delays_s=DELAYS) is JobStatus.QUEUED
    queued = _get(repo, job.id)
    assert queued.status is JobStatus.QUEUED
    assert (queued.stage, queued.worker_id, queued.lease_until) == (None, None, None)
    assert (queued.attempt, queued.lease_claims, queued.scheduled_retries) == (2, 0, 1)
    assert queued.not_before == clock.now() + DELAYS[0]
    assert (queued.error_code, queued.error_cause, queued.retryable) == (
        "RENDER_FAILED",
        "subprocess_crash",
        True,
    )
    ev = repo.events(job.id)[-1]
    assert (ev.type, ev.data["delay_s"], ev.data["scheduled_retries"]) == (
        "retry_scheduled",
        DELAYS[0],
        1,
    )

    clock.advance(DELAYS[0] - 1)
    assert repo.claim(worker_id="C", lease_s=LEASE_S, max_claims=MAX_CLAIMS) is None
    clock.advance(1)
    lease3 = _claim(repo, "C")
    resumed = _get(repo, job.id)
    assert (lease3.attempt, resumed.lease_claims) == (3, 1)


def test_second_scheduled_retry_uses_300s(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    job = _new_job(repo)
    repo.schedule_retry_or_fail(_claim(repo), RETRYABLE, delays_s=DELAYS)
    clock.advance(DELAYS[0])
    assert repo.schedule_retry_or_fail(_claim(repo), RETRYABLE, delays_s=DELAYS) is JobStatus.QUEUED
    queued = _get(repo, job.id)
    assert queued.scheduled_retries == 2
    assert queued.not_before == clock.now() + 300.0


def test_third_retryable_failure_fails_job(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    failed = _fail_retryable_to_terminal(repo, clock)
    assert failed.status is JobStatus.FAILED
    assert (failed.error_code, failed.retryable, failed.scheduled_retries) == (
        "RENDER_FAILED",
        True,
        2,
    )
    assert failed.finished_at == clock.now()
    assert repo.events(failed.id)[-1].type == "job_failed"


def test_non_retryable_error_fails_immediately(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    job = _new_job(repo)
    status = repo.schedule_retry_or_fail(_claim(repo), NON_RETRYABLE, delays_s=DELAYS)
    assert status is JobStatus.FAILED
    failed = _get(repo, job.id)
    assert (failed.status, failed.scheduled_retries, failed.retryable) == (
        JobStatus.FAILED,
        0,
        False,
    )


def test_schedule_retry_with_stale_lease_rejected(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    job = _new_job(repo)
    lease_a = _claim(repo, "A")
    clock.advance(LEASE_S + 1)
    _claim(repo, "B")
    before = _get(repo, job.id)
    with pytest.raises(StaleLeaseError):
        repo.schedule_retry_or_fail(lease_a, RETRYABLE, delays_s=DELAYS)
    with pytest.raises(StaleLeaseError):
        repo.schedule_retry_or_fail(lease_a, NON_RETRYABLE, delays_s=DELAYS)
    assert _get(repo, job.id) == before


def test_manual_retry_resets_counters_and_error(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    failed = _fail_retryable_to_terminal(repo, clock)
    clock.advance(10)
    retried = repo.manual_retry(failed.id)
    assert retried.status is JobStatus.QUEUED
    assert (retried.lease_claims, retried.scheduled_retries, retried.not_before) == (0, 0, None)
    assert (retried.error_code, retried.error_cause, retried.error_message) == (None, None, None)
    assert retried.retryable is None
    assert retried.attempt == failed.attempt
    assert retried.updated_at == clock.now()
    assert repo.events(failed.id)[-1].type == "retry_requested"
    assert _claim(repo).attempt == failed.attempt + 1


def test_manual_retry_rejects_non_retryable_and_non_failed(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    queued = _new_job(repo)
    with pytest.raises(NotRetryable):
        repo.manual_retry(queued.id)

    running = _new_job(repo)
    # the oldest queued job is claimed first, so claim twice to reach `running`
    _claim(repo)
    lease = _claim(repo)
    assert lease.job_id == running.id
    with pytest.raises(NotRetryable):
        repo.manual_retry(running.id)

    repo.fail(lease, NON_RETRYABLE)
    with pytest.raises(NotRetryable):
        repo.manual_retry(running.id)

    with pytest.raises(NotRetryable):
        repo.manual_retry("vid_missing")


def test_concurrent_manual_retry_one_winner(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    failed = _fail_retryable_to_terminal(repo, clock)
    barrier = threading.Barrier(8)
    outcomes: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        local = JobRepository(db_path, clock)
        barrier.wait()
        try:
            local.manual_retry(failed.id)
            result = "ok"
        except NotRetryable:
            result = "rejected"
        except BaseException as exc:
            result = f"error: {exc!r}"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["ok"] + ["rejected"] * 7
    assert [e.type for e in repo.events(failed.id)].count("retry_requested") == 1


def test_set_generation_is_fenced(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    job = _new_job(repo)
    lease = _claim(repo)
    repo.set_generation(lease, generation_source="curated_fallback", fallback_reason="LLM_X")
    got = _get(repo, job.id)
    assert (got.generation_source, got.fallback_reason) == ("curated_fallback", "LLM_X")
    with pytest.raises(StaleLeaseError):
        repo.set_generation(
            Lease(job.id, "other", lease.attempt), generation_source="llm", fallback_reason=None
        )


def test_promote_checkpoint_rejects_stale_lease(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    job = _new_job(repo)
    lease_a = _claim(repo, "A")
    repo.promote_checkpoint(
        lease_a, stage_key="reviewing", input_hash="in-1", manifest_sha256="m-1", versions={"v": 1}
    )
    row = repo.get_checkpoint(job.id, "reviewing")
    assert row is not None
    assert (row.attempt, row.input_hash, row.manifest_sha256, row.versions) == (
        1,
        "in-1",
        "m-1",
        {"v": 1},
    )

    clock.advance(LEASE_S + 1)
    lease_b = _claim(repo, "B")
    with pytest.raises(StaleLeaseError):
        repo.promote_checkpoint(
            lease_a, stage_key="reviewing", input_hash="in-x", manifest_sha256="m-x", versions={}
        )
    assert repo.get_checkpoint(job.id, "reviewing") == row

    repo.promote_checkpoint(
        lease_b, stage_key="reviewing", input_hash="in-2", manifest_sha256="m-2", versions={"v": 2}
    )
    repo.promote_checkpoint(
        lease_b, stage_key="voicing", input_hash="in-3", manifest_sha256="m-3", versions={"v": 2}
    )
    rows = repo.list_checkpoints(job.id)
    assert [(r.stage_key, r.attempt, r.manifest_sha256) for r in rows] == [
        ("reviewing", 2, "m-2"),
        ("voicing", 2, "m-3"),
    ]
    assert repo.get_checkpoint(job.id, "rendering") is None


def test_attempt_monotonic_across_retry_cycles(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    job = _new_job(repo)
    attempts: list[int] = []
    for _cycle in range(2):
        for _ in range(3):
            lease = _claim(repo)
            attempts.append(lease.attempt)
            if repo.schedule_retry_or_fail(lease, RETRYABLE, delays_s=DELAYS) is JobStatus.FAILED:
                break
            clock.advance(max(DELAYS))
        repo.manual_retry(job.id)
    lease = _claim(repo)
    attempts.append(lease.attempt)
    clock.advance(LEASE_S + 1)
    attempts.append(_claim(repo).attempt)  # lease reclaim
    assert attempts == list(range(1, len(attempts) + 1))
