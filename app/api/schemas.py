"""Request/response DTOs for the /v1 API (spec §9.2)."""

import hashlib
import json
import re
import unicodedata
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.domain.job import Job, stage_progress
from app.topics.resolver import normalize

QUERY_MIN_CHARS = 3
QUERY_MAX_CHARS = 300
_WHITESPACE = re.compile(r"\s+")


def _clean_query(text: str) -> str:
    """Control and format characters become spaces; whitespace runs collapse; ends are trimmed."""
    spaced = "".join(" " if unicodedata.category(ch) in ("Cc", "Cf") else ch for ch in text)
    return _WHITESPACE.sub(" ", spaced).strip()


class CreateVideoRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        cleaned = _clean_query(value)
        if not QUERY_MIN_CHARS <= len(cleaned) <= QUERY_MAX_CHARS:
            raise ValueError(
                f"query must be {QUERY_MIN_CHARS}-{QUERY_MAX_CHARS} characters after cleaning"
            )
        return cleaned


class ErrorBody(BaseModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class Links(BaseModel):
    self: str
    content: str
    artifacts: str


class JobError(BaseModel):
    code: str
    cause: str | None
    message: str | None
    retryable: bool | None


class VideoJob(BaseModel):
    id: str
    query: str
    topic_id: str
    status: str
    stage: str | None
    progress: float
    attempt: int
    generation_source: str | None
    fallback_reason: str | None
    error: JobError | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    links: Links


class EventOut(BaseModel):
    ts: datetime
    attempt: int
    type: str
    stage: str | None
    data: dict[str, Any]


class ArtifactRefOut(BaseModel):
    name: str
    content_type: str
    bytes: int
    sha256: str
    url: str


class CostOut(BaseModel):
    billable_usd: float = 0
    billable_actual_usd: float = 0
    billable_estimated_usd: float = 0
    failed_call_usd: float = 0
    free_calls: int = 0
    estimated_production_compute_usd: float = 0
    by_operation: dict[str, float] = Field(default_factory=dict)
    pricing_version: str | None = None


class VideoJobDetail(VideoJob):
    events: list[EventOut]
    quality: dict[str, Any] | None
    cost: CostOut
    artifacts: list[ArtifactRefOut]
    stage_durations_s: dict[str, float]


class VideoJobList(BaseModel):
    items: list[VideoJob]
    next_cursor: str | None


class TopicOut(BaseModel):
    topic_id: str
    title: str
    canonical_query: str


def to_datetime(ts: float | None) -> datetime | None:
    return datetime.fromtimestamp(ts, tz=UTC) if ts is not None else None


def job_to_dto(job: Job) -> VideoJob:
    base = f"/v1/videos/{job.id}"
    error = None
    if job.error_code is not None:
        error = JobError(
            code=job.error_code,
            cause=job.error_cause,
            message=job.error_message,
            retryable=job.retryable,
        )
    return VideoJob(
        id=job.id,
        query=job.query,
        topic_id=job.topic_id,
        status=str(job.status),
        stage=str(job.stage) if job.stage is not None else None,
        progress=stage_progress(job.status, job.stage),
        attempt=job.attempt,
        generation_source=job.generation_source,
        fallback_reason=job.fallback_reason,
        error=error,
        created_at=datetime.fromtimestamp(job.created_at, tz=UTC),
        updated_at=datetime.fromtimestamp(job.updated_at, tz=UTC),
        started_at=to_datetime(job.started_at),
        finished_at=to_datetime(job.finished_at),
        links=Links(self=base, content=f"{base}/content", artifacts=f"{base}/artifacts"),
    )


def request_fingerprint(req: CreateVideoRequest) -> str:
    body = json.dumps({"query": normalize(req.query)}, sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()
