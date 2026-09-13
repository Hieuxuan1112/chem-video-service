from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.errors import ApiError, install_error_handlers
from app.api.schemas import CreateVideoRequest, job_to_dto, request_fingerprint
from app.domain.job import Job, JobStatus, Stage


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("abc", "abc"),
        ("  How does the pH scale work?  ", "How does the pH scale work?"),
        ("pH\x00scale\x07?", "pH scale ?"),
        ("x" * 300, "x" * 300),
    ],
)
def test_query_validation_accepts_and_cleans(raw: str, cleaned: str) -> None:
    assert CreateVideoRequest(query=raw).query == cleaned


@pytest.mark.parametrize("raw", ["ab", "   ab   ", "\x00\x01ab\x02", "x" * 301, ""])
def test_query_validation_bounds(raw: str) -> None:
    with pytest.raises(ValidationError):
        CreateVideoRequest(query=raw)


def test_query_must_be_string() -> None:
    with pytest.raises(ValidationError):
        CreateVideoRequest.model_validate({"query": 123})
    with pytest.raises(ValidationError):
        CreateVideoRequest.model_validate({})


def test_fingerprint_stable_under_whitespace_and_case() -> None:
    a = request_fingerprint(CreateVideoRequest(query="How does the pH scale work?"))
    b = request_fingerprint(CreateVideoRequest(query="  how DOES the ph   scale work "))
    c = request_fingerprint(CreateVideoRequest(query="Why do atoms form covalent bonds?"))
    assert a == b
    assert a != c
    assert len(a) == 64


def _job(**kw: object) -> Job:
    base: dict = {
        "id": "vid_1",
        "query": "How does the pH scale work?",
        "topic_id": "ph_scale",
        "status": JobStatus.RUNNING,
        "stage": Stage.VOICING,
        "attempt": 2,
        "lease_claims": 1,
        "scheduled_retries": 1,
        "worker_id": "w",
        "lease_until": 1_000_060.0,
        "not_before": None,
        "generation_source": "llm",
        "fallback_reason": None,
        "error_code": "RENDER_FAILED",
        "error_cause": "subprocess_crash",
        "error_message": "crashed",
        "retryable": True,
        "idempotency_key": None,
        "request_fingerprint": None,
        "pipeline_version": "1.0.0",
        "created_at": 1_000_000.0,
        "updated_at": 1_000_010.0,
        "started_at": 1_000_001.0,
        "finished_at": None,
    }
    base.update(kw)
    return Job(**base)


def test_job_to_dto_error_and_progress() -> None:
    dto = job_to_dto(_job())
    assert dto.progress == pytest.approx(2 / 6)
    assert dto.error is not None
    assert (dto.error.code, dto.error.cause, dto.error.message, dto.error.retryable) == (
        "RENDER_FAILED",
        "subprocess_crash",
        "crashed",
        True,
    )
    assert dto.created_at == datetime.fromtimestamp(1_000_000.0, tz=UTC)
    assert dto.finished_at is None
    assert dto.stage == "voicing"
    assert dto.links.self == "/v1/videos/vid_1"
    assert dto.links.content == "/v1/videos/vid_1/content"
    assert dto.links.artifacts == "/v1/videos/vid_1/artifacts"
    body = dto.model_dump(mode="json")
    assert body["links"]["self"] == "/v1/videos/vid_1"
    assert body["status"] == "running"

    done = job_to_dto(_job(status=JobStatus.SUCCEEDED, error_code=None, retryable=None))
    assert (done.progress, done.error) == (1.0, None)


def _error_app() -> TestClient:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise ApiError(409, "VIDEO_NOT_READY", "not ready", details={"status": "queued"})

    @app.post("/echo")
    def echo(req: CreateVideoRequest) -> dict:
        return {"query": req.query}

    return TestClient(app)


def test_error_handlers_use_one_envelope() -> None:
    client = _error_app()
    r = client.get("/boom")
    assert r.status_code == 409
    assert r.json() == {
        "error": {
            "code": "VIDEO_NOT_READY",
            "message": "not ready",
            "retryable": False,
            "details": {"status": "queued"},
        }
    }

    r = client.post("/echo", json={"query": "no"})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "INVALID_REQUEST"
    assert err["retryable"] is False
    assert err["details"]["errors"][0]["loc"] == ["body", "query"]

    r = client.post("/echo", content=b"{not json", headers={"content-type": "application/json"})
    assert (r.status_code, r.json()["error"]["code"]) == (422, "INVALID_REQUEST")

    r = client.get("/nope")
    assert (r.status_code, r.json()["error"]["code"]) == (404, "NOT_FOUND")

    r = client.delete("/boom")
    assert (r.status_code, r.json()["error"]["code"]) == (405, "METHOD_NOT_ALLOWED")
