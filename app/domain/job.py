"""Job states, stages and the single transition table (spec §4.1)."""

import uuid
from dataclasses import dataclass
from enum import StrEnum

from app.domain.errors import IllegalTransition


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Stage(StrEnum):
    SCRIPTING = "scripting"
    REVIEWING = "reviewing"
    VOICING = "voicing"
    RENDERING = "rendering"
    QUALITY_CHECK = "quality_check"
    PUBLISHING = "publishing"


STAGE_ORDER: tuple[Stage, ...] = (
    Stage.SCRIPTING,
    Stage.REVIEWING,
    Stage.VOICING,
    Stage.RENDERING,
    Stage.QUALITY_CHECK,
    Stage.PUBLISHING,
)

ALLOWED_TRANSITIONS: frozenset[tuple[JobStatus, JobStatus]] = frozenset(
    {
        (JobStatus.QUEUED, JobStatus.RUNNING),  # claim
        (JobStatus.RUNNING, JobStatus.RUNNING),  # lease reclaim
        (JobStatus.RUNNING, JobStatus.SUCCEEDED),
        (JobStatus.RUNNING, JobStatus.FAILED),
        (JobStatus.RUNNING, JobStatus.QUEUED),  # scheduled retry
        (JobStatus.FAILED, JobStatus.QUEUED),  # manual retry
    }
)


def check_transition(src: JobStatus, dst: JobStatus) -> None:
    if (src, dst) not in ALLOWED_TRANSITIONS:
        raise IllegalTransition(f"{src} -> {dst}")


def stage_progress(status: JobStatus, stage: Stage | None) -> float:
    """Stage-based progress: 0.0 when queued, 1.0 when succeeded, else the stage's share."""
    if status is JobStatus.SUCCEEDED:
        return 1.0
    if status is JobStatus.QUEUED or stage is None:
        return 0.0
    return STAGE_ORDER.index(stage) / len(STAGE_ORDER)


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    query: str
    topic_id: str
    status: JobStatus
    stage: Stage | None
    attempt: int
    lease_claims: int
    scheduled_retries: int
    worker_id: str | None
    lease_until: float | None
    not_before: float | None
    generation_source: str | None
    fallback_reason: str | None
    error_code: str | None
    error_cause: str | None
    error_message: str | None
    retryable: bool | None
    idempotency_key: str | None
    request_fingerprint: str | None
    pipeline_version: str
    created_at: float
    updated_at: float
    started_at: float | None
    finished_at: float | None


@dataclass(frozen=True, slots=True)
class Lease:
    job_id: str
    worker_id: str
    attempt: int


def new_job_id() -> str:
    return "vid_" + uuid.uuid4().hex
