import threading
from pathlib import Path

import pytest

from app.domain.errors import StaleLeaseError
from app.domain.job import Job, JobStatus, Lease, Stage
from app.jobs.clock import FakeClock
from app.jobs.repository import JobRepository

LEASE_S = 60.0
MAX_CLAIMS = 3


def _new_job(repo: JobRepository, query: str = "How does the pH scale work?") -> Job:
    job, _ = repo.create(
        query=query,
        topic_id="ph_scale",
        pipeline_version="1.0.0",
        idempotency_key=None,
        request_fingerprint=None,
    )
    return job


def _claim(repo: JobRepository, worker: str) -> tuple[Job, Lease] | None:
    return repo.claim(worker_id=worker, lease_s=LEASE_S, max_claims=MAX_CLAIMS)


def test_claim_returns_oldest_and_sets_lease(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    first = _new_job(repo, "first")
    clock.advance(1)
    _new_job(repo, "second")
    clock.advance(1)
    claimed = _claim(repo, "w1")
    assert claimed is not None
    job, lease = claimed
    assert job.id == first.id
    assert job.status is JobStatus.RUNNING
    assert job.worker_id == "w1"
    assert job.lease_until == clock.now() + LEASE_S
    assert (job.attempt, job.lease_claims) == (1, 1)
    assert job.started_at == clock.now()
    assert lease == Lease(job_id=first.id, worker_id="w1", attempt=1)
    last = repo.events(job.id)[-1]
    assert (last.type, last.attempt, last.data["worker_id"]) == ("job_claimed", 1, "w1")


def test_claim_none_when_empty(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    assert _claim(repo, "w1") is None


def test_parallel_claims_never_duplicate(db_path: Path, clock: FakeClock) -> None:
    setup = JobRepository(db_path, clock)
    for _ in range(50):
        _new_job(setup)
    barrier = threading.Barrier(8)
    claimed: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(n: int) -> None:
        repo = JobRepository(db_path, clock)
        barrier.wait()
        try:
            while (res := _claim(repo, f"w{n}")) is not None:
                with lock:
                    claimed.append(res[0].id)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(claimed) == 50
    assert len(set(claimed)) == 50


def test_expired_lease_reclaimed_with_attempt_increment(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    job = _new_job(repo)
    assert _claim(repo, "A") is not None
    clock.advance(LEASE_S - 1)
    assert _claim(repo, "B") is None  # lease still valid
    clock.advance(2)
    reclaimed = _claim(repo, "B")
    assert reclaimed is not None
    again, lease = reclaimed
    assert again.id == job.id
    assert again.status is JobStatus.RUNNING
    assert (again.worker_id, again.attempt, again.lease_claims) == ("B", 2, 2)
    assert lease == Lease(job.id, "B", 2)
    ev = repo.events(job.id)[-1]
    assert ev.type == "lease_reclaimed"
    assert ev.data["previous_worker"] == "A"
    assert ev.data["previous_attempt"] == 1


def test_heartbeat_prevents_reclaim(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    _new_job(repo)
    claimed = _claim(repo, "A")
    assert claimed is not None
    _, lease = claimed
    for _ in range(5):
        clock.advance(50)
        repo.heartbeat(lease, lease_s=LEASE_S)
        assert _claim(repo, "B") is None
    job = repo.get(lease.job_id)
    assert job is not None
    assert job.lease_until == clock.now() + LEASE_S


def test_stale_worker_write_rejected(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    _new_job(repo)
    claimed_a = _claim(repo, "A")
    assert claimed_a is not None
    _, lease_a = claimed_a
    clock.advance(LEASE_S + 1)
    claimed_b = _claim(repo, "B")
    assert claimed_b is not None
    _, lease_b = claimed_b
    before_job = repo.get(lease_a.job_id)
    before_events = repo.events(lease_a.job_id)

    with pytest.raises(StaleLeaseError):
        repo.heartbeat(lease_a, lease_s=LEASE_S)
    with pytest.raises(StaleLeaseError):
        repo.set_stage(lease_a, Stage.VOICING)
    with pytest.raises(StaleLeaseError):
        repo.add_event_fenced(lease_a, "stage_completed", stage="voicing")

    assert repo.get(lease_a.job_id) == before_job
    assert repo.events(lease_a.job_id) == before_events
    repo.set_stage(lease_b, Stage.SCRIPTING)  # the live holder can still write


def test_set_stage_and_fenced_event(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    _new_job(repo)
    claimed = _claim(repo, "A")
    assert claimed is not None
    _, lease = claimed
    clock.advance(3)
    repo.set_stage(lease, Stage.REVIEWING)
    repo.add_event_fenced(lease, "gate_result", stage="reviewing", data={"gate": "G1"})
    job = repo.get(lease.job_id)
    assert job is not None
    assert job.stage is Stage.REVIEWING
    assert job.updated_at == clock.now()
    events = repo.events(lease.job_id)
    assert [(e.type, e.stage, e.attempt) for e in events[-2:]] == [
        ("stage_started", "reviewing", 1),
        ("gate_result", "reviewing", 1),
    ]
    assert events[-1].data == {"gate": "G1"}


def test_sweep_fails_after_max_claims(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    job = _new_job(repo)
    for worker in ("A", "B", "C"):
        assert _claim(repo, worker) is not None
        clock.advance(LEASE_S + 1)
    assert repo.sweep_expired(max_claims=MAX_CLAIMS) == 1
    failed = repo.get(job.id)
    assert failed is not None
    assert failed.status is JobStatus.FAILED
    assert (failed.error_code, failed.error_cause, failed.retryable) == (
        "WORKER_LOST",
        "lease_expired_max_claims",
        True,
    )
    assert (failed.worker_id, failed.lease_until) == (None, None)
    assert failed.finished_at == clock.now()
    assert failed.scheduled_retries == 0  # the sweep never schedules a retry
    assert repo.events(job.id)[-1].type == "job_failed"
    assert _claim(repo, "D") is None
    assert repo.sweep_expired(max_claims=MAX_CLAIMS) == 0


def test_sweep_ignores_live_and_reclaimable_leases(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    _new_job(repo)
    assert _claim(repo, "A") is not None
    assert repo.sweep_expired(max_claims=MAX_CLAIMS) == 0  # live lease
    clock.advance(LEASE_S + 1)
    assert repo.sweep_expired(max_claims=MAX_CLAIMS) == 0  # expired but below max_claims


def test_not_before_blocks_claim(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    job = _new_job(repo)
    repo._conn().execute("UPDATE jobs SET not_before = ? WHERE id = ?", (clock.now() + 60, job.id))
    assert _claim(repo, "A") is None
    clock.advance(59)
    assert _claim(repo, "A") is None
    clock.advance(1)
    assert _claim(repo, "A") is not None


def test_worker_heartbeat_age(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    assert repo.worker_heartbeat_age_s() is None
    repo.touch_worker("A")
    clock.advance(4)
    repo.touch_worker("B")
    clock.advance(2)
    assert repo.worker_heartbeat_age_s() == 2.0
