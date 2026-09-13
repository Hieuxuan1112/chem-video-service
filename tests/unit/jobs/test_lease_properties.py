"""Stateful property test of the lease protocol (spec §10.1)."""

import contextlib
import shutil
import tempfile
from pathlib import Path

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from app.domain.errors import AppError, ErrorCode, NotRetryable, StaleLeaseError
from app.domain.job import JobStatus, Lease
from app.jobs.clock import FakeClock
from app.jobs.repository import JobRepository

LEASE_S = 60.0
MAX_CLAIMS = 3
DELAYS = (60.0, 300.0)
MAX_JOBS = 6
WORKERS = ("w0", "w1", "w2")
RETRYABLE = AppError(ErrorCode.RENDER_FAILED, "subprocess_crash", "injected")


class LeaseMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.tmp = Path(tempfile.mkdtemp(prefix="lease-props-"))
        self.clock = FakeClock()
        self.repo = JobRepository(self.tmp / "app.sqlite", self.clock)
        self.job_ids: list[str] = []
        self.held: dict[str, Lease | None] = dict.fromkeys(WORKERS)
        self.zombies: list[
            Lease
        ] = []  # leases of crashed workers that still believe they hold them
        self.max_attempt_seen: dict[str, int] = {}

    # --- rules ---------------------------------------------------------------------------------

    @precondition(lambda self: len(self.job_ids) < MAX_JOBS)
    @rule()
    def create_job(self) -> None:
        job, _ = self.repo.create(
            query="How does the pH scale work?",
            topic_id="ph_scale",
            pipeline_version="1.0.0",
            idempotency_key=None,
            request_fingerprint=None,
        )
        self.job_ids.append(job.id)

    @rule(worker=st.sampled_from(WORKERS))
    def claim(self, worker: str) -> None:
        if self.held[worker] is not None:
            return
        res = self.repo.claim(worker_id=worker, lease_s=LEASE_S, max_claims=MAX_CLAIMS)
        if res is not None:
            self.held[worker] = res[1]

    @rule(worker=st.sampled_from(WORKERS))
    def heartbeat(self, worker: str) -> None:
        lease = self.held[worker]
        if lease is None:
            return
        try:
            self.repo.heartbeat(lease, lease_s=LEASE_S)
        except StaleLeaseError:
            self.held[worker] = None

    @rule(worker=st.sampled_from(WORKERS))
    def crash(self, worker: str) -> None:
        lease = self.held[worker]
        if lease is None:
            return
        self.zombies.append(lease)
        self.held[worker] = None

    @rule(worker=st.sampled_from(WORKERS))
    def claim_then_crash(self, worker: str) -> None:
        # claim + crash in one step, so sequences that exhaust max_claims (and exercise the
        # WORKER_LOST sweep) are generated often enough.
        self.claim(worker)
        self.crash(worker)

    @rule(index=st.integers(min_value=0, max_value=20))
    def zombie_write(self, index: int) -> None:
        if not self.zombies:
            return
        lease = self.zombies[index % len(self.zombies)]
        try:
            self.repo.add_event_fenced(lease, "zombie_write")
        except StaleLeaseError:
            self.zombies.remove(lease)

    @rule(seconds=st.floats(min_value=1.0, max_value=90.0))
    def advance(self, seconds: float) -> None:
        self.clock.advance(seconds)

    @rule()
    def sweep(self) -> None:
        self.repo.sweep_expired(max_claims=MAX_CLAIMS)

    @rule(worker=st.sampled_from(WORKERS))
    def stage_fail_retryable(self, worker: str) -> None:
        lease = self.held[worker]
        if lease is None:
            return
        self.held[worker] = None
        with contextlib.suppress(StaleLeaseError):
            self.repo.schedule_retry_or_fail(lease, RETRYABLE, delays_s=DELAYS)

    @rule(worker=st.sampled_from(WORKERS))
    def succeed(self, worker: str) -> None:
        lease = self.held[worker]
        if lease is None:
            return
        self.held[worker] = None
        with contextlib.suppress(StaleLeaseError):
            self.repo.succeed(lease)

    @rule(index=st.integers(min_value=0, max_value=20))
    def manual_retry(self, index: int) -> None:
        if not self.job_ids:
            return
        with contextlib.suppress(NotRetryable):
            self.repo.manual_retry(self.job_ids[index % len(self.job_ids)])

    # --- invariants ----------------------------------------------------------------------------

    def _passes_fence(self, lease: Lease) -> bool:
        row = (
            self.repo._conn()
            .execute(
                "SELECT 1 FROM jobs WHERE id=? AND worker_id=? AND attempt=? AND status='running'",
                (lease.job_id, lease.worker_id, lease.attempt),
            )
            .fetchone()
        )
        return row is not None

    @invariant()
    def at_most_one_believed_lease_passes_fence(self) -> None:
        believed = [lease for lease in self.held.values() if lease is not None] + self.zombies
        passing: dict[str, int] = {}
        for lease in believed:
            if self._passes_fence(lease):
                passing[lease.job_id] = passing.get(lease.job_id, 0) + 1
        assert all(count <= 1 for count in passing.values()), passing

    @invariant()
    def attempt_never_decreases(self) -> None:
        for job_id in self.job_ids:
            job = self.repo.get(job_id)
            assert job is not None
            assert job.attempt >= self.max_attempt_seen.get(job_id, 0)
            self.max_attempt_seen[job_id] = job.attempt

    @invariant()
    def lease_claims_bounded_for_running_jobs(self) -> None:
        for job_id in self.job_ids:
            job = self.repo.get(job_id)
            assert job is not None
            if job.status is JobStatus.RUNNING:
                assert job.lease_claims <= MAX_CLAIMS

    # --- eventual termination ------------------------------------------------------------------

    def teardown(self) -> None:
        try:
            self.clock.advance(1000.0)
            for _ in range(200):
                self.repo.sweep_expired(max_claims=MAX_CLAIMS)
                res = self.repo.claim(worker_id="live", lease_s=LEASE_S, max_claims=MAX_CLAIMS)
                if res is not None:
                    self.repo.succeed(res[1])
                    continue
                statuses = [self.repo.get(j).status for j in self.job_ids]  # type: ignore[union-attr]
                if all(s in (JobStatus.SUCCEEDED, JobStatus.FAILED) for s in statuses):
                    break
            statuses = [self.repo.get(j).status for j in self.job_ids]  # type: ignore[union-attr]
            assert all(s in (JobStatus.SUCCEEDED, JobStatus.FAILED) for s in statuses), statuses
        finally:
            self.repo._conn().close()
            shutil.rmtree(self.tmp, ignore_errors=True)


LeaseMachine.TestCase.settings = settings(max_examples=150, stateful_step_count=40, deadline=None)
TestLeaseProtocol = LeaseMachine.TestCase
