import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.config import Settings
from app.container import Container
from app.domain.job import JobStatus
from app.jobs.clock import FakeClock, SystemClock
from app.jobs.repository import JobRepository
from app.jobs.worker import JobContext, Worker
from app.topics.resolver import normalize

TOPICS = [
    ("ph_scale", "The pH scale", "How does the pH scale work?"),
    ("covalent_bonds", "Covalent bonds", "Why do atoms form covalent bonds?"),
]


class StubResolver:
    def resolve(self, query: str) -> str | None:
        return "ph_scale" if "ph" in normalize(query).split() else None

    def topics(self) -> list[tuple[str, str, str]]:
        return TOPICS


class StubArtifacts:
    def __init__(self, root: Path) -> None:
        self.root = root

    def open(self, job_id: str, name: str) -> Path | None:
        path = self.root / job_id / name
        return path if path.is_file() else None

    def list_refs(self, job_id: str) -> list[dict[str, Any]]:
        return [
            {
                "name": p.name,
                "content_type": "video/mp4",
                "bytes": p.stat().st_size,
                "sha256": "0" * 64,
                "url": f"/v1/videos/{job_id}/artifacts/{p.name}",
            }
            for p in sorted((self.root / job_id).iterdir())
        ]

    def manifest(self, job_id: str) -> dict[str, Any] | None:
        return None


@pytest.fixture
def container(tmp_path: Path, clock: FakeClock) -> Container:
    settings = Settings(
        _env_file=None, database_path=tmp_path / "app.sqlite", run_embedded_worker=False
    )
    return Container(
        settings=settings,
        clock=clock,
        repo=JobRepository(settings.database_path, clock),
        resolver=StubResolver(),
        artifacts=StubArtifacts(tmp_path / "artifacts"),
        spend_state=lambda: {"state": "ok", "billable_usd": 0.0},
    )


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    with TestClient(create_app(container)) as c:
        yield c


def _post(
    client: TestClient, query: str = "How does the pH scale work?", key: str | bytes | None = None
):  # type: ignore[no-untyped-def]
    headers = {"Idempotency-Key": key} if key is not None else {}
    return client.post("/v1/videos", json={"query": query}, headers=headers)


def _succeed(container: Container, job_id: str, video_bytes: bytes = b"") -> None:
    claimed = container.repo.claim(worker_id="w", lease_s=60, max_claims=3)
    assert claimed is not None and claimed[0].id == job_id
    container.repo.succeed(claimed[1])
    if video_bytes:
        assert isinstance(container.artifacts, StubArtifacts)
        d = container.artifacts.root / job_id
        d.mkdir(parents=True)
        (d / "video.mp4").write_bytes(video_bytes)


def _assert_envelope(body: dict, code: str) -> dict:
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "retryable", "details"}
    assert body["error"]["code"] == code
    return body["error"]


def test_create_202_location_and_body(client: TestClient) -> None:
    r = _post(client)
    assert r.status_code == 202
    body = r.json()
    assert body["id"].startswith("vid_")
    assert r.headers["location"] == f"/v1/videos/{body['id']}"
    assert (body["status"], body["topic_id"], body["progress"], body["attempt"]) == (
        "queued",
        "ph_scale",
        0.0,
        0,
    )
    assert body["links"]["content"] == f"/v1/videos/{body['id']}/content"
    assert "idempotent-replayed" not in r.headers


def test_unsupported_topic_422_lists_topics(client: TestClient) -> None:
    r = _post(client, "What is photosynthesis?")
    assert r.status_code == 422
    err = _assert_envelope(r.json(), "UNSUPPORTED_TOPIC")
    assert [t["topic_id"] for t in err["details"]["supported_topics"]] == [
        "ph_scale",
        "covalent_bonds",
    ]
    assert client.get("/v1/videos").json()["items"] == []


def test_invalid_request_envelope(client: TestClient) -> None:
    for payload in ({"query": "ab"}, {}, {"query": 5}):
        r = client.post("/v1/videos", json=payload)
        assert r.status_code == 422
        _assert_envelope(r.json(), "INVALID_REQUEST")


def test_idempotent_replay_header(client: TestClient) -> None:
    first = _post(client, key="req-123")
    second = _post(client, "  how does the PH scale work ", key="req-123")
    assert (first.status_code, second.status_code) == (202, 202)
    assert second.json()["id"] == first.json()["id"]
    assert second.headers["idempotent-replayed"] == "true"
    assert len(client.get("/v1/videos").json()["items"]) == 1


def test_idempotency_conflict_409(client: TestClient) -> None:
    _post(client, key="req-123")
    r = _post(client, "Explain the pH of lemon juice", key="req-123")
    assert r.status_code == 409
    _assert_envelope(r.json(), "IDEMPOTENCY_KEY_REUSED")


@pytest.mark.parametrize("key", ["", "has space", "x" * 129, "tab\tkey", "ключ".encode()])
def test_bad_idempotency_key_422(client: TestClient, key: str | bytes) -> None:
    r = _post(client, key=key)
    assert r.status_code == 422
    _assert_envelope(r.json(), "INVALID_REQUEST")


def test_list_and_cursor(client: TestClient, container: Container) -> None:
    clock = container.clock
    assert isinstance(clock, FakeClock)
    ids = []
    for _ in range(5):
        ids.append(_post(client).json()["id"])
        clock.advance(1)
    page1 = client.get("/v1/videos", params={"limit": 2}).json()
    page2 = client.get("/v1/videos", params={"limit": 2, "cursor": page1["next_cursor"]}).json()
    page3 = client.get("/v1/videos", params={"limit": 2, "cursor": page2["next_cursor"]}).json()
    got = [j["id"] for p in (page1, page2, page3) for j in p["items"]]
    assert got == list(reversed(ids))
    assert page3["next_cursor"] is None

    _succeed(container, ids[0])
    done = client.get("/v1/videos", params={"status": "succeeded"}).json()["items"]
    assert [j["id"] for j in done] == [ids[0]]
    assert client.get("/v1/videos", params={"topic_id": "covalent_bonds"}).json()["items"] == []

    for params in ({"cursor": "garbage!!"}, {"limit": 0}, {"limit": 101}, {"status": "done"}):
        r = client.get("/v1/videos", params=params)
        assert r.status_code == 422, params
        _assert_envelope(r.json(), "INVALID_REQUEST")


def test_get_detail_and_404_envelope(client: TestClient) -> None:
    job_id = _post(client).json()["id"]
    detail = client.get(f"/v1/videos/{job_id}").json()
    assert detail["id"] == job_id
    assert [e["type"] for e in detail["events"]] == ["job_created"]
    assert detail["cost"]["billable_usd"] == 0
    assert detail["artifacts"] == []
    r = client.get("/v1/videos/vid_missing")
    assert r.status_code == 404
    _assert_envelope(r.json(), "NOT_FOUND")


def test_content_409_when_not_ready(client: TestClient) -> None:
    job_id = _post(client).json()["id"]
    for path in ("content", "artifacts", "artifacts/video.mp4"):
        r = client.get(f"/v1/videos/{job_id}/{path}")
        assert r.status_code == 409, path
        err = _assert_envelope(r.json(), "VIDEO_NOT_READY")
        assert err["details"] == {"status": "queued", "stage": None}
    r = client.get("/v1/videos/vid_missing/content")
    assert r.status_code == 404


def test_content_served_with_range(client: TestClient, container: Container) -> None:
    job_id = _post(client).json()["id"]
    payload = bytes(range(256)) * 4
    _succeed(container, job_id, payload)
    full = client.get(f"/v1/videos/{job_id}/content")
    assert full.status_code == 200
    assert full.headers["content-type"] == "video/mp4"
    assert full.content == payload
    part = client.get(f"/v1/videos/{job_id}/content", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206
    assert part.content == payload[:100]
    refs = client.get(f"/v1/videos/{job_id}/artifacts").json()
    assert [ref["name"] for ref in refs] == ["video.mp4"]
    assert client.get(f"/v1/videos/{job_id}/artifacts/video.mp4").content == payload


def test_artifact_name_not_allowlisted_404(client: TestClient, container: Container) -> None:
    job_id = _post(client).json()["id"]
    _succeed(container, job_id, b"data")
    for name in ("secrets.env", "..%2F..%2Fapp.sqlite", "stage.json"):
        r = client.get(f"/v1/videos/{job_id}/artifacts/{name}")
        assert r.status_code == 404, name
        _assert_envelope(r.json(), "NOT_FOUND")
    r = client.get(f"/v1/videos/{job_id}/artifacts/thumbnail.jpg")  # allow-listed but absent
    assert r.status_code == 404


def test_retry_flow_202_then_409(client: TestClient, container: Container) -> None:
    from app.domain.errors import AppError, ErrorCode

    job_id = _post(client).json()["id"]
    r = client.post(f"/v1/videos/{job_id}/retry")
    assert r.status_code == 409
    _assert_envelope(r.json(), "NOT_RETRYABLE")

    claimed = container.repo.claim(worker_id="w", lease_s=60, max_claims=3)
    assert claimed is not None
    err = AppError(ErrorCode.ARTIFACT_STORE_ERROR, "io_error", "disk")
    container.repo.fail(claimed[1], err)

    r = client.post(f"/v1/videos/{job_id}/retry")
    assert r.status_code == 202
    assert (r.json()["status"], r.json()["error"]) == ("queued", None)
    r = client.post(f"/v1/videos/{job_id}/retry")
    assert r.status_code == 409
    _assert_envelope(r.json(), "NOT_RETRYABLE")
    assert client.post("/v1/videos/vid_missing/retry").status_code == 404


def test_topics(client: TestClient) -> None:
    assert client.get("/v1/topics").json() == [
        {"topic_id": t, "title": title, "canonical_query": q} for t, title, q in TOPICS
    ]


def test_healthz_ok_and_503_without_ffmpeg(
    client: TestClient, container: Container, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert (body["db"], body["ffmpeg"], body["worker_heartbeat_age_s"]) == ("ok", "ok", None)
    assert body["project_spend"] == {"state": "ok", "billable_usd": 0.0}

    container.repo.touch_worker("w")
    assert isinstance(container.clock, FakeClock)
    container.clock.advance(3)
    assert client.get("/healthz").json()["worker_heartbeat_age_s"] == 3.0

    monkeypatch.setattr("app.api.routes.shutil.which", lambda name: None)
    r = client.get("/healthz")
    assert r.status_code == 503
    err = _assert_envelope(r.json(), "SERVICE_UNAVAILABLE")
    assert err["retryable"] is True
    assert err["details"]["ffmpeg"] == "missing"


def test_shutdown_does_not_wait_for_a_running_job(tmp_path: Path) -> None:
    import threading

    clock = SystemClock()
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "app.sqlite",
        poll_interval_s=0.05,
        worker_shutdown_grace_s=0.2,
    )
    repo = JobRepository(settings.database_path, clock)
    started = threading.Event()
    release = threading.Event()

    class SlowHandler:
        def handle(self, ctx: JobContext) -> None:
            started.set()
            release.wait(timeout=10)

    container = Container(
        settings=settings,
        clock=clock,
        repo=repo,
        resolver=StubResolver(),
        worker=Worker(repo, SlowHandler(), settings),
    )
    try:
        with TestClient(create_app(container)) as c:
            _post(c)
            assert started.wait(timeout=5)
            t0 = time.monotonic()
        assert time.monotonic() - t0 < 2.0  # bounded by the grace period, not by the job
    finally:
        release.set()


def test_embedded_worker_runs_and_stops(tmp_path: Path) -> None:
    clock = SystemClock()
    settings = Settings(_env_file=None, database_path=tmp_path / "app.sqlite", poll_interval_s=0.05)
    repo = JobRepository(settings.database_path, clock)
    handled: list[str] = []

    class Handler:
        def handle(self, ctx: JobContext) -> None:
            handled.append(ctx.job.id)

    worker = Worker(repo, Handler(), settings)
    container = Container(
        settings=settings, clock=clock, repo=repo, resolver=StubResolver(), worker=worker
    )
    with TestClient(create_app(container)) as c:
        job_id = _post(c).json()["id"]
        deadline = time.monotonic() + 5
        while c.get(f"/v1/videos/{job_id}").json()["status"] != "succeeded":
            assert time.monotonic() < deadline
            time.sleep(0.05)
    assert handled == [job_id]
    job = repo.get(job_id)
    assert job is not None and job.status is JobStatus.SUCCEEDED
