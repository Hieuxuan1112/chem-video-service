"""HTTP routes (spec §9.2)."""

import json
import mimetypes
import re
import shutil
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Header, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from app.api.errors import ApiError, error_response
from app.api.schemas import (
    ArtifactRefOut,
    CostOut,
    CreateVideoRequest,
    EventOut,
    TopicOut,
    VideoJob,
    VideoJobDetail,
    VideoJobList,
    job_to_dto,
    request_fingerprint,
)
from app.container import Container
from app.domain.errors import IdempotencyConflict, NotRetryable
from app.domain.job import Job, JobStatus

PUBLISHED_FILES: tuple[str, ...] = (
    "video.mp4",
    "thumbnail.jpg",
    "captions.vtt",
    "storyboard.json",
    "quality_report.json",
    "contact_sheet.jpg",
    "manifest.json",
)
_IDEMPOTENCY_KEY = re.compile(r"^[\x21-\x7E]{1,128}$")

router = APIRouter()


def _container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


def _job_or_404(container: Container, job_id: str) -> Job:
    job = container.repo.get(job_id)
    if job is None:
        raise ApiError(404, "NOT_FOUND", f"video job {job_id} not found")
    return job


def _require_ready(job: Job) -> None:
    if job.status is not JobStatus.SUCCEEDED:
        raise ApiError(
            409,
            "VIDEO_NOT_READY",
            f"video job {job.id} is {job.status}",
            retryable=True,
            details={"status": str(job.status), "stage": job.stage},
        )


def _artifact_path_or_404(container: Container, job_id: str, name: str) -> Path:
    path = container.artifacts.open(job_id, name) if container.artifacts is not None else None
    if path is None:
        raise ApiError(404, "NOT_FOUND", f"artifact {name} not found for {job_id}")
    return path


@router.post("/v1/videos", status_code=202, response_model=VideoJob)
def create_video(
    request: Request,
    body: CreateVideoRequest,
    response: Response,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> VideoJob:
    container = _container(request)
    if idempotency_key is not None and not _IDEMPOTENCY_KEY.match(idempotency_key):
        raise ApiError(
            422,
            "INVALID_REQUEST",
            "Idempotency-Key must be 1-128 visible ASCII characters",
            details={"header": "Idempotency-Key"},
        )
    topic_id = container.resolver.resolve(body.query)
    if topic_id is None:
        supported = [
            {"topic_id": t, "title": title, "canonical_query": q}
            for t, title, q in container.resolver.topics()
        ]
        raise ApiError(
            422,
            "UNSUPPORTED_TOPIC",
            "the query does not match a supported topic",
            details={"supported_topics": supported},
        )
    try:
        job, replayed = container.repo.create(
            query=body.query,
            topic_id=topic_id,
            pipeline_version=container.settings.pipeline_version,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint(body),
        )
    except IdempotencyConflict:
        raise ApiError(
            409,
            "IDEMPOTENCY_KEY_REUSED",
            "Idempotency-Key was already used with a different request body",
        ) from None
    response.headers["Location"] = f"/v1/videos/{job.id}"
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return job_to_dto(job)


@router.get("/v1/videos", response_model=VideoJobList)
def list_videos(
    request: Request,
    status: Annotated[JobStatus | None, Query()] = None,
    topic_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query()] = None,
) -> VideoJobList:
    container = _container(request)
    try:
        jobs, next_cursor = container.repo.list(
            status=status, topic_id=topic_id, limit=limit, cursor=cursor
        )
    except ValueError:
        raise ApiError(
            422, "INVALID_REQUEST", "invalid cursor", details={"param": "cursor"}
        ) from None
    return VideoJobList(items=[job_to_dto(j) for j in jobs], next_cursor=next_cursor)


@router.get("/v1/videos/{job_id}", response_model=VideoJobDetail)
def get_video(request: Request, job_id: str) -> VideoJobDetail:
    container = _container(request)
    job = _job_or_404(container, job_id)
    events = [
        EventOut.model_validate(
            {"ts": e.ts, "attempt": e.attempt, "type": e.type, "stage": e.stage, "data": e.data}
        )
        for e in container.repo.events(job_id)
    ]
    cost = CostOut(**container.job_cost(job_id)) if container.job_cost is not None else CostOut()
    artifacts: list[ArtifactRefOut] = []
    quality: dict[str, Any] | None = None
    stage_durations: dict[str, float] = {}
    if job.status is JobStatus.SUCCEEDED and container.artifacts is not None:
        artifacts = [ArtifactRefOut(**ref) for ref in container.artifacts.list_refs(job_id)]
        manifest = container.artifacts.manifest(job_id) or {}
        stage_durations = manifest.get("stage_durations_s", {})
        report = container.artifacts.open(job_id, "quality_report.json")
        if report is not None:
            quality = json.loads(report.read_text(encoding="utf-8")).get("summary")
    return VideoJobDetail(
        **job_to_dto(job).model_dump(),
        events=events,
        quality=quality,
        cost=cost,
        artifacts=artifacts,
        stage_durations_s=stage_durations,
    )


@router.get("/v1/videos/{job_id}/content")
def get_content(request: Request, job_id: str) -> FileResponse:
    container = _container(request)
    job = _job_or_404(container, job_id)
    _require_ready(job)
    path = _artifact_path_or_404(container, job_id, "video.mp4")
    return FileResponse(path, media_type="video/mp4")


@router.get("/v1/videos/{job_id}/artifacts", response_model=list[ArtifactRefOut])
def list_artifacts(request: Request, job_id: str) -> list[ArtifactRefOut]:
    container = _container(request)
    job = _job_or_404(container, job_id)
    _require_ready(job)
    if container.artifacts is None:
        return []
    return [ArtifactRefOut(**ref) for ref in container.artifacts.list_refs(job_id)]


@router.get("/v1/videos/{job_id}/artifacts/{name}")
def get_artifact(request: Request, job_id: str, name: str) -> FileResponse:
    container = _container(request)
    job = _job_or_404(container, job_id)
    if name not in PUBLISHED_FILES:
        raise ApiError(404, "NOT_FOUND", f"unknown artifact {name!r}")
    _require_ready(job)
    path = _artifact_path_or_404(container, job_id, name)
    media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


@router.post("/v1/videos/{job_id}/retry", status_code=202, response_model=VideoJob)
def retry_video(request: Request, job_id: str) -> VideoJob:
    container = _container(request)
    job = _job_or_404(container, job_id)
    try:
        retried = container.repo.manual_retry(job_id)
    except NotRetryable:
        raise ApiError(
            409,
            "NOT_RETRYABLE",
            f"video job {job_id} is {job.status} and cannot be retried",
            details={"status": str(job.status), "retryable": job.retryable},
        ) from None
    return job_to_dto(retried)


@router.get("/v1/topics", response_model=list[TopicOut])
def list_topics(request: Request) -> list[TopicOut]:
    return [
        TopicOut(topic_id=t, title=title, canonical_query=q)
        for t, title, q in _container(request).resolver.topics()
    ]


@router.get("/healthz")
def healthz(request: Request) -> JSONResponse:
    container = _container(request)
    try:
        container.repo.ping()
        db = "ok"
    except Exception:
        db = "error"
    ffmpeg = "ok" if shutil.which("ffmpeg") is not None else "missing"
    body: dict[str, Any] = {
        "db": db,
        "ffmpeg": ffmpeg,
        "worker_heartbeat_age_s": container.repo.worker_heartbeat_age_s() if db == "ok" else None,
        "project_spend": container.spend_state() if container.spend_state is not None else None,
    }
    if db != "ok" or ffmpeg != "ok":
        return error_response(
            503,
            "SERVICE_UNAVAILABLE",
            "a required dependency is unavailable",
            retryable=True,
            details=body,
        )
    return JSONResponse(body)
