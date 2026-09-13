"""Worker loop: claim with a lease, keep it alive with a heartbeat thread, write fenced outcomes."""

import contextvars
import logging
import os
import socket
import threading
import uuid
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol

from app.config import Settings
from app.domain.errors import AppError, ErrorCode, StaleLeaseError, classify
from app.domain.job import Job, Lease, Stage
from app.jobs.repository import JobRepository
from app.observability.context import job_context

log = logging.getLogger(__name__)


@dataclass
class JobContext:
    """A handler's view of its job. Writes are fenced; a stale write marks the lease lost."""

    job: Job
    lease: Lease
    repo: JobRepository
    lease_lost: threading.Event
    lease_s: float = 60.0

    def check(self) -> None:
        if self.lease_lost.is_set():
            raise StaleLeaseError(f"lease lost: {self.lease}")

    def _fenced(self, write: Any, *args: Any, **kwargs: Any) -> None:
        self.check()
        try:
            write(self.lease, *args, **kwargs)
        except StaleLeaseError:
            self.lease_lost.set()
            raise

    def set_stage(self, stage: Stage) -> None:
        self._fenced(self.repo.set_stage, stage)

    def emit(self, type: str, stage: str | None = None, data: dict[str, Any] | None = None) -> None:
        self._fenced(self.repo.add_event_fenced, type, stage, data)

    def renew_lease(self) -> None:
        self._fenced(self.repo.heartbeat, lease_s=self.lease_s)


class JobHandler(Protocol):
    def handle(self, ctx: JobContext) -> None:
        """Runs the job; returns only after publication."""
        ...


class LeaseKeeper:
    """Heartbeat thread. Any heartbeat failure sets `lost` (spec §4.4) and stops the thread."""

    def __init__(
        self,
        repo: JobRepository,
        lease: Lease,
        *,
        lease_s: float,
        interval_s: float,
        lost: threading.Event,
    ) -> None:
        self._repo = repo
        self._lease = lease
        self._lease_s = lease_s
        self._interval_s = interval_s
        self._lost = lost
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self._repo.heartbeat(self._lease, lease_s=self._lease_s)
            except Exception as exc:
                self._lost.set()
                log.warning("lease_lost", extra={"data": {"reason": repr(exc)}})
                return

    def __enter__(self) -> "LeaseKeeper":
        ctx = contextvars.copy_context()  # heartbeat logs keep the job context
        self._thread = threading.Thread(
            target=ctx.run, args=(self._run,), name=f"lease-{self._lease.job_id}", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


class Worker:
    def __init__(
        self,
        repo: JobRepository,
        handler: JobHandler,
        settings: Settings,
        *,
        worker_id: str | None = None,
    ) -> None:
        self.repo = repo
        self.handler = handler
        self.settings = settings
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    def run_once(self) -> bool:
        """Poll once. Returns True if a job was claimed."""
        s = self.settings
        self.repo.touch_worker(self.worker_id)
        self.repo.sweep_expired(max_claims=s.max_claims)
        claimed = self.repo.claim(
            worker_id=self.worker_id, lease_s=s.lease_s, max_claims=s.max_claims
        )
        if claimed is None:
            return False
        job, lease = claimed
        lost = threading.Event()
        with job_context(job_id=job.id, attempt=lease.attempt, worker_id=self.worker_id):
            log.info("job_claimed")
            ctx = JobContext(
                job=job, lease=lease, repo=self.repo, lease_lost=lost, lease_s=s.lease_s
            )
            error: AppError | None = None
            try:
                with LeaseKeeper(
                    self.repo,
                    lease,
                    lease_s=s.lease_s,
                    interval_s=s.heartbeat_interval_s,
                    lost=lost,
                ):
                    self.handler.handle(ctx)
            except StaleLeaseError:
                lost.set()
            except Exception as exc:
                error = classify(exc)
                if error.code is ErrorCode.INTERNAL_ERROR:
                    log.exception("unclassified handler error")
            if lost.is_set():
                log.warning("lease_lost: abandoning job without further writes")
                return True
            try:
                if error is None:
                    self.repo.succeed(lease)
                    log.info("job_succeeded")
                else:
                    status = self.repo.schedule_retry_or_fail(
                        lease, error, delays_s=s.scheduled_retry_delays_s
                    )
                    log.warning(
                        "job_error",
                        extra={
                            "data": {
                                "code": str(error.code),
                                "cause": error.cause,
                                "status": status,
                            }
                        },
                    )
            except StaleLeaseError:
                log.warning("lease_lost: outcome not written")
        return True

    def run_forever(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                worked = self.run_once()
            except Exception:
                log.exception("worker poll failed")
                worked = False
            if not worked:
                stop.wait(self.settings.poll_interval_s)
