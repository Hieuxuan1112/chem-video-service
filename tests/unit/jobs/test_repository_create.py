import threading
from pathlib import Path

import pytest

from app.domain.errors import IdempotencyConflict
from app.domain.job import JobStatus
from app.jobs.clock import FakeClock
from app.jobs.repository import JobRepository


def _create(repo: JobRepository, **kw: object) -> tuple:
    args: dict = {
        "query": "How does the pH scale work?",
        "topic_id": "ph_scale",
        "pipeline_version": "1.0.0",
        "idempotency_key": None,
        "request_fingerprint": "fp-1",
    }
    args.update(kw)
    return repo.create(**args)


def test_create_and_get(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    job, replayed = _create(repo)
    assert replayed is False
    assert job.id.startswith("vid_")
    assert job.status is JobStatus.QUEUED
    assert job.stage is None
    assert (job.attempt, job.lease_claims, job.scheduled_retries) == (0, 0, 0)
    assert job.created_at == job.updated_at == clock.now()
    assert repo.get(job.id) == job
    assert repo.get("vid_missing") is None
    events = repo.events(job.id)
    assert [e.type for e in events] == ["job_created"]
    assert events[0].attempt == 0


def test_add_event_and_order(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    job, _ = _create(repo)
    repo.add_event(job.id, 0, "retry_requested", data={"by": "api"})
    repo.add_event(job.id, 1, "stage_started", stage="scripting")
    events = repo.events(job.id)
    assert [e.type for e in events] == ["job_created", "retry_requested", "stage_started"]
    assert events[1].data == {"by": "api"}
    assert events[2].stage == "scripting"
    assert [e.id for e in events] == sorted(e.id for e in events)


def test_idempotent_replay_same_fingerprint(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    first, replayed1 = _create(repo, idempotency_key="key-1", request_fingerprint="fp-a")
    clock.advance(5)
    second, replayed2 = _create(repo, idempotency_key="key-1", request_fingerprint="fp-a")
    assert (replayed1, replayed2) == (False, True)
    assert second == first
    count = repo._conn().execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    assert count == 1


def test_idempotency_conflict_different_fingerprint(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    _create(repo, idempotency_key="key-1", request_fingerprint="fp-a")
    with pytest.raises(IdempotencyConflict):
        _create(repo, idempotency_key="key-1", request_fingerprint="fp-b")


def test_concurrent_same_key_creates_one_job(db_path: Path, clock: FakeClock) -> None:
    JobRepository(db_path, clock)  # schema applied once up front
    barrier = threading.Barrier(8)
    results: list[tuple] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        repo = JobRepository(db_path, clock)
        barrier.wait()
        try:
            res = _create(repo, idempotency_key="same", request_fingerprint="fp")
        except BaseException as exc:
            with lock:
                errors.append(exc)
            return
        with lock:
            results.append(res)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len({job.id for job, _ in results}) == 1
    assert sorted(replayed for _, replayed in results) == [False] + [True] * 7
    repo = JobRepository(db_path, clock)
    assert repo._conn().execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_list_pagination_and_filters(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    created = []
    for i in range(25):
        if i % 3 == 0:
            clock.advance(1)  # leaves created_at ties so the id tie-breaker is exercised
        topic = "ph_scale" if i % 2 == 0 else "covalent_bonds"
        created.append(_create(repo, topic_id=topic)[0])

    pages = []
    cursor = None
    while True:
        items, cursor = repo.list(status=None, topic_id=None, limit=10, cursor=cursor)
        pages.append(items)
        if cursor is None:
            break
    assert [len(p) for p in pages] == [10, 10, 5]
    flat = [j.id for p in pages for j in p]
    expected = [j.id for j in sorted(created, key=lambda j: (j.created_at, j.id), reverse=True)]
    assert flat == expected

    ph, _ = repo.list(status=None, topic_id="ph_scale", limit=100, cursor=None)
    assert len(ph) == 13
    assert all(j.topic_id == "ph_scale" for j in ph)
    queued, _ = repo.list(status=JobStatus.QUEUED, topic_id=None, limit=100, cursor=None)
    assert len(queued) == 25
    running, nxt = repo.list(status=JobStatus.RUNNING, topic_id=None, limit=100, cursor=None)
    assert (running, nxt) == ([], None)


def test_list_invalid_cursor_raises_value_error(db_path: Path, clock: FakeClock) -> None:
    repo = JobRepository(db_path, clock)
    with pytest.raises(ValueError):
        repo.list(status=None, topic_id=None, limit=10, cursor="not-a-cursor!!")
