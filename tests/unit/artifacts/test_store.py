import json
import os
from pathlib import Path

import pytest

from app.artifacts import store as store_module
from app.artifacts.store import CONTENT_TYPES, LocalArtifactStore, Manifest
from app.domain.errors import AppError, ErrorCode

JOB = "vid_0123456789abcdef"


def _manifest(job_id: str = JOB) -> Manifest:
    return Manifest(
        job_id=job_id,
        pipeline_version="1.0.0",
        schema_version="1",
        prompt_versions={"writer": "w1"},
        models={"writer": "gemini-2.5-flash"},
        voice={"provider": "fake", "voice": "fake"},
        renderer_version="fake-1",
        topic_pack_version="1",
        generation_source="llm",
        fallback_reason=None,
        checkpoint_attempts={"reviewing": 1, "voicing": 1},
        cost={"billable_usd": 0.01},
        stage_durations_s={"rendering": 3.2},
        files={},
        created_at=1_700_000_000.0,
    )


def _files(tmp_path: Path) -> dict[str, Path]:
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    contents = {
        "video.mp4": b"\x00\x00\x00\x18ftypisom video",
        "thumbnail.jpg": b"\xff\xd8thumb",
        "captions.vtt": b"WEBVTT\n\n",
        "storyboard.json": b"{}",
        "quality_report.json": b'{"passed": true}',
        "contact_sheet.jpg": b"\xff\xd8sheet",
    }
    for name, data in contents.items():
        (src / name).write_bytes(data)
    return {name: src / name for name in contents}


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "artifacts")


def test_publish_writes_manifest_last(
    store: LocalArtifactStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replaced: list[str] = []
    real_replace = os.replace

    def recording(src: str | Path, dst: str | Path) -> None:
        replaced.append(Path(src).name + " -> " + Path(dst).name)
        real_replace(src, dst)

    monkeypatch.setattr(store_module.os, "replace", recording)
    store.publish(JOB, _files(tmp_path), _manifest(), attempt=2)

    assert replaced[-1] == "manifest.json.tmp-2 -> manifest.json"
    assert sorted(r.split(" -> ")[1] for r in replaced[:-1]) == sorted(_files(tmp_path))
    assert all(".tmp-2 -> " in r for r in replaced)
    job_dir = tmp_path / "artifacts" / JOB
    assert not list(job_dir.glob("*.tmp-*"))
    manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["video.mp4"]["content_type"] == "video/mp4"
    assert manifest["files"]["video.mp4"]["bytes"] == len(b"\x00\x00\x00\x18ftypisom video")


def test_verify_and_refs_after_publish(store: LocalArtifactStore, tmp_path: Path) -> None:
    store.publish(JOB, _files(tmp_path), _manifest(), attempt=1)

    verified = store.verify(JOB)
    assert verified is not None and verified.checkpoint_attempts == {"reviewing": 1, "voicing": 1}
    refs = {r["name"]: r for r in store.list_refs(JOB)}
    assert set(refs) == set(CONTENT_TYPES)
    assert refs["captions.vtt"]["url"] == f"/v1/videos/{JOB}/artifacts/captions.vtt"
    assert refs["manifest.json"]["content_type"] == "application/json"
    assert store.manifest(JOB) == json.loads(
        (tmp_path / "artifacts" / JOB / "manifest.json").read_text(encoding="utf-8")
    )
    assert store.open(JOB, "video.mp4") == tmp_path / "artifacts" / JOB / "video.mp4"


def test_verify_detects_tampered_file(store: LocalArtifactStore, tmp_path: Path) -> None:
    store.publish(JOB, _files(tmp_path), _manifest(), attempt=1)
    (tmp_path / "artifacts" / JOB / "thumbnail.jpg").write_bytes(b"\xff\xd8other")

    assert store.verify(JOB) is None
    assert store.list_refs(JOB) == []
    assert store.manifest(JOB) is None


def test_verify_detects_missing_file_and_foreign_manifest(
    store: LocalArtifactStore, tmp_path: Path
) -> None:
    store.publish(JOB, _files(tmp_path), _manifest(), attempt=1)
    (tmp_path / "artifacts" / JOB / "captions.vtt").unlink()
    assert store.verify(JOB) is None

    other = "vid_ffff"
    store.publish(other, _files(tmp_path), _manifest(job_id=JOB), attempt=1)
    assert store.verify(other) is None


def test_verify_none_without_manifest(store: LocalArtifactStore, tmp_path: Path) -> None:
    assert store.verify(JOB) is None
    job_dir = tmp_path / "artifacts" / JOB
    job_dir.mkdir(parents=True)
    (job_dir / "video.mp4").write_bytes(b"orphan")  # a crash before the manifest
    assert store.verify(JOB) is None
    (job_dir / "manifest.json").write_text("{not json", encoding="utf-8")
    assert store.verify(JOB) is None


def test_open_rejects_unknown_names_and_traversal(
    store: LocalArtifactStore, tmp_path: Path
) -> None:
    store.publish(JOB, _files(tmp_path), _manifest(), attempt=1)
    (tmp_path / "artifacts" / JOB / "secret.txt").write_text("x", encoding="utf-8")
    (tmp_path / "artifacts" / "video.mp4").write_bytes(b"outside any job directory")

    assert store.open(JOB, "secret.txt") is None
    assert store.open(JOB, "../../etc/passwd") is None
    assert store.open("..", "video.mp4") is None
    assert store.open(f"{JOB}/..", "video.mp4") is None
    assert store.open("vid_missing", "video.mp4") is None
    with pytest.raises(ValueError):
        store.publish(JOB, {"secret.txt": tmp_path / "src" / "video.mp4"}, _manifest(), attempt=1)


def test_republish_same_content_idempotent(store: LocalArtifactStore, tmp_path: Path) -> None:
    store.publish(JOB, _files(tmp_path), _manifest(), attempt=1)
    first = store.list_refs(JOB)

    store.publish(JOB, _files(tmp_path), _manifest(), attempt=2)

    assert store.list_refs(JOB) == first
    assert store.verify(JOB) is not None
    assert not list((tmp_path / "artifacts" / JOB).glob("*.tmp-*"))


def test_io_failure_is_retryable_store_error(
    store: LocalArtifactStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(src: str | Path, dst: str | Path) -> None:
        raise PermissionError("disk full")

    monkeypatch.setattr(store_module.os, "replace", broken)
    with pytest.raises(AppError) as info:
        store.publish(JOB, _files(tmp_path), _manifest(), attempt=1)

    assert (info.value.code, info.value.cause, info.value.retryable) == (
        ErrorCode.ARTIFACT_STORE_ERROR, "io_error", True,
    )  # fmt: skip
    assert store.verify(JOB) is None
