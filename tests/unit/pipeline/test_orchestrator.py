"""Orchestrator tests over build_container(fake_providers=True): real repository, checkpoints,
assembler and QA (ffmpeg), fake LLM/TTS/renderer, everything in tmp directories."""

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.artifacts.store import CONTENT_TYPES
from app.config import Settings
from app.container import Container, CuratedFakeLLM, build_container
from app.domain.errors import AppError, ErrorCode
from app.domain.job import JobStatus
from app.jobs.worker import JobContext
from app.pipeline.orchestrator import PipelineDeps, VideoPipeline, render_error_to_app_error
from app.render.fake import FakeRenderer
from app.render.protocol import SceneRenderResult

pytestmark = pytest.mark.integration  # fake renders, assembly and QA run ffmpeg

QUERY = "How does the pH scale work?"


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "database_path": tmp_path / "app.sqlite",
        "artifacts_dir": tmp_path / "artifacts",
        "work_dir": tmp_path / "work",
        "cache_dir": tmp_path / "cache",
        "width": 320,
        "height": 180,
        "fps": 10,
        "qa_min_total_s": 10.0,
        "run_embedded_worker": False,
        "scheduled_retry_delays_s": (0.0, 0.0),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
def container(tmp_path: Path) -> Container:
    return build_container(_settings(tmp_path), fake_providers=True)


def _deps(container: Container) -> PipelineDeps:
    assert container.worker is not None
    handler = container.worker.handler
    assert isinstance(handler, VideoPipeline)
    return handler._d


def _llm(container: Container) -> CuratedFakeLLM:
    llm = _deps(container).writer._llm
    assert isinstance(llm, CuratedFakeLLM)
    return llm


def _tts_calls(container: Container) -> int:
    return len(_deps(container).voicer._primary.calls)  # type: ignore[attr-defined]


def _submit(container: Container) -> str:
    job, _ = container.repo.create(
        query=QUERY,
        topic_id="ph_scale",
        pipeline_version=container.settings.pipeline_version,
        idempotency_key=None,
        request_fingerprint=None,
    )
    return job.id


def _run(container: Container) -> None:
    assert container.worker is not None
    assert container.worker.run_once()


def _events(container: Container, job_id: str) -> list[str]:
    return [e.type for e in container.repo.events(job_id)]


def test_happy_path_publishes_all_files_and_manifest(container: Container) -> None:
    job_id = _submit(container)
    _run(container)

    job = container.repo.get(job_id)
    assert job is not None and job.status is JobStatus.SUCCEEDED, job
    assert job.generation_source == "llm"
    refs = {r["name"] for r in container.artifacts.list_refs(job_id)}  # type: ignore[union-attr]
    assert refs == set(CONTENT_TYPES)
    manifest = container.artifacts.manifest(job_id)  # type: ignore[union-attr]
    assert manifest is not None
    assert manifest["checkpoint_attempts"] == {
        "reviewing": 1,
        "voicing": 1,
        "rendering": 1,
        "quality_check": 1,
        **{f"rendering/scene-{i:02d}": 1 for i in range(6)},
    }
    assert manifest["renderer_version"] == "fake-1" and manifest["voice"]["provider"] == "fake_tts"
    assert set(manifest["stage_durations_s"]) >= {"scripting", "voicing", "rendering"}
    events = _events(container, job_id)
    assert "artifact_published" in events and "media_qa" in events
    storyboard = container.artifacts.open(job_id, "storyboard.json")  # type: ignore[union-attr]
    assert (
        storyboard is not None
        and json.loads(storyboard.read_text("utf-8"))["topic_id"] == "ph_scale"
    )


def test_resume_skips_reviewing_and_voicing_when_checkpoints_valid(container: Container) -> None:
    deps = _deps(container)
    deps.renderer = FakeRenderer(script={3: "crash"})
    job_id = _submit(container)
    _run(container)
    job = container.repo.get(job_id)
    assert job is not None and job.status is JobStatus.QUEUED and job.error_code is not None
    llm_calls, tts_calls = len(_llm(container).requests), _tts_calls(container)

    renderer = FakeRenderer()
    deps.renderer = renderer
    _run(container)

    job = container.repo.get(job_id)
    assert job is not None and job.status is JobStatus.SUCCEEDED
    assert len(_llm(container).requests) == llm_calls  # FakeLLM not called again
    assert _tts_calls(container) == tts_calls
    assert renderer.calls == [[3]]  # only the crashed scene is rendered again
    skipped = [
        e.data for e in container.repo.events(job_id) if e.type == "stage_skipped_checkpoint"
    ]
    assert {"stage_key": "reviewing"} in skipped and {"stage_key": "voicing"} in skipped


def test_version_change_invalidates_checkpoint(container: Container) -> None:
    deps = _deps(container)
    deps.renderer = FakeRenderer(script={0: "crash"})
    job_id = _submit(container)
    _run(container)
    llm_calls = len(_llm(container).requests)

    deps.renderer = FakeRenderer()
    deps.settings = deps.settings.model_copy(update={"tts_voice": "Puck"})  # voice config changed
    _run(container)

    job = container.repo.get(job_id)
    assert job is not None and job.status is JobStatus.SUCCEEDED
    assert len(_llm(container).requests) > llm_calls  # every version-bound checkpoint is stale
    voicing = container.repo.get_checkpoint(job_id, "voicing")
    assert voicing is not None and voicing.attempt == 2  # recomputed (the TTS cache may still hit)


def test_render_template_error_non_retryable(container: Container) -> None:
    _deps(container).renderer = FakeRenderer(script={2: "template_error"})
    job_id = _submit(container)
    _run(container)

    job = container.repo.get(job_id)
    assert job is not None and job.status is JobStatus.FAILED
    assert (job.error_code, job.error_cause) == ("RENDER_TEMPLATE_ERROR", "deterministic_bug")
    assert container.artifacts.verify(job_id) is None  # type: ignore[union-attr]


def test_render_crash_retryable(container: Container) -> None:
    _deps(container).renderer = FakeRenderer(script={1: "crash"})
    job_id = _submit(container)
    _run(container)

    job = container.repo.get(job_id)
    assert job is not None and job.status is JobStatus.QUEUED
    assert job.error_code == "RENDER_FAILED"


def test_render_error_mapping_prefers_code_bugs() -> None:
    def failed(i: int, kind: Any) -> SceneRenderResult:
        return SceneRenderResult(i, False, None, None, kind, "m")

    cases = [
        ([failed(0, "crash"), failed(1, "layout_error")], ErrorCode.RENDER_LAYOUT_ERROR, False),
        (
            [failed(0, "timeout"), failed(1, "template_error")],
            ErrorCode.RENDER_TEMPLATE_ERROR,
            False,
        ),
        ([failed(0, "crash"), failed(1, "timeout")], ErrorCode.RENDER_TIMEOUT, True),
        ([failed(0, "crash")], ErrorCode.RENDER_FAILED, True),
    ]
    for results, code, retryable in cases:
        error = render_error_to_app_error(results)
        assert (error.code, error.retryable) == (code, retryable)


def test_qa_failure_cause_propagates(container: Container) -> None:
    _deps(container).renderer = FakeRenderer(script={5: "black"})
    job_id = _submit(container)
    _run(container)

    job = container.repo.get(job_id)
    assert job is not None and job.status is JobStatus.FAILED
    assert (job.error_code, job.error_cause) == ("QUALITY_CHECK_FAILED", "deterministic_output_bug")
    assert container.artifacts.verify(job_id) is None  # type: ignore[union-attr]


def _publish_without_success(container: Container) -> tuple[str, Any]:
    """Runs the pipeline to publication, then fails the job retryably as if the worker died."""
    job_id = _submit(container)
    s = container.settings
    claimed = container.repo.claim(worker_id="w1", lease_s=s.lease_s, max_claims=s.max_claims)
    assert claimed is not None
    job, lease = claimed
    ctx = JobContext(
        job=job, lease=lease, repo=container.repo, lease_lost=threading.Event(), lease_s=s.lease_s
    )
    assert container.worker is not None
    container.worker.handler.handle(ctx)
    lost = AppError(ErrorCode.RENDER_FAILED, "subprocess_crash", "worker died after publishing")
    container.repo.schedule_retry_or_fail(lease, lost, delays_s=(0.0, 0.0))
    return job_id, lease


def test_publish_recovery_requires_checkpoint_attempts_match(container: Container) -> None:
    job_id, _ = _publish_without_success(container)
    calls = (len(_llm(container).requests), _tts_calls(container))

    _run(container)  # attempt 2: manifest matches the checkpoint rows

    job = container.repo.get(job_id)
    assert job is not None and job.status is JobStatus.SUCCEEDED
    assert "publish_recovered" in _events(container, job_id)
    assert (len(_llm(container).requests), _tts_calls(container)) == calls


def test_publish_recovery_refused_when_checkpoints_superseded(container: Container) -> None:
    job_id, _ = _publish_without_success(container)
    s = container.settings
    claimed = container.repo.claim(worker_id="w2", lease_s=s.lease_s, max_claims=s.max_claims)
    assert claimed is not None
    _, lease2 = claimed
    row = container.repo.get_checkpoint(job_id, "quality_check")
    assert row is not None
    container.repo.promote_checkpoint(
        lease2, stage_key="quality_check", input_hash="other", manifest_sha256="x", versions={}
    )
    container.repo.schedule_retry_or_fail(
        lease2, AppError(ErrorCode.RENDER_FAILED, "subprocess_crash", "again"), delays_s=(0.0, 0.0)
    )

    _run(container)  # attempt 3: manifest attempts differ from the DB rows

    job = container.repo.get(job_id)
    assert job is not None and job.status is JobStatus.SUCCEEDED
    events = _events(container, job_id)
    assert "publish_recovered" not in events
    assert events.count("artifact_published") == 2  # published again from valid checkpoints
    manifest = container.artifacts.manifest(job_id)  # type: ignore[union-attr]
    assert manifest is not None and manifest["checkpoint_attempts"]["quality_check"] == 3


def test_stale_lease_mid_pipeline_stops_without_publish(container: Container) -> None:
    deps = _deps(container)
    inner = FakeRenderer()

    class LeaseStealingRenderer:
        name, version = inner.name, inner.version

        def render(
            self, plan: Any, out_dir: Path, *, scenes: Any = None, cancel: Any = None
        ) -> Any:
            result = inner.render(plan, out_dir, scenes=scenes)
            cancel.set()  # the heartbeat thread lost the lease while rendering
            return result

    deps.renderer = LeaseStealingRenderer()  # type: ignore[assignment]
    job_id = _submit(container)
    _run(container)

    job = container.repo.get(job_id)
    assert job is not None and job.status is JobStatus.RUNNING  # abandoned to lease expiry
    assert container.artifacts.verify(job_id) is None  # type: ignore[union-attr]
    assert container.repo.get_checkpoint(job_id, "rendering") is None


def test_render_compute_row_not_billable(container: Container) -> None:
    job_id = _submit(container)
    _run(container)

    ledger = _deps(container).ledger
    rows = [
        dict(r)
        for r in ledger._conn().execute(
            "SELECT * FROM cost_entries WHERE job_id=? AND operation='render_compute'", (job_id,)
        )
    ]
    assert len(rows) == 1 and rows[0]["billable"] == 0 and rows[0]["stage"] == "rendering"
    assert ledger.job_totals(job_id).committed_usd == 0.0
    assert ledger.project_totals().committed_usd == 0.0


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    with TestClient(create_app(container)) as c:
        yield c


def test_api_detail_shows_cost_quality_artifacts(client: TestClient, container: Container) -> None:
    created = client.post("/v1/videos", json={"query": QUERY})
    assert created.status_code == 202, created.text
    job_id = created.json()["id"]
    _run(container)

    detail = client.get(f"/v1/videos/{job_id}").json()
    assert detail["status"] == "succeeded"
    assert {a["name"] for a in detail["artifacts"]} == set(CONTENT_TYPES)
    assert detail["quality"]["voice_provider"] == "fake_tts"
    assert detail["quality"]["asr_skipped"] is True and detail["quality"]["media_qa_passed"] is True
    assert detail["quality"]["critic_scores"] == {
        k: 5 for k in ("accuracy", "relevance", "clarity", "alignment")
    }
    assert detail["stage_durations_s"]["rendering"] > 0
    assert detail["cost"]["estimated_production_compute_usd"] == 0.0
    assert "render_compute" not in detail["cost"]["by_operation"]
    content = client.get(f"/v1/videos/{job_id}/content")
    assert content.status_code == 200 and content.headers["content-type"] == "video/mp4"
