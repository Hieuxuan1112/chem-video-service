import json
from pathlib import Path

import pytest

from app.domain.errors import StaleLeaseError
from app.domain.job import Lease
from app.jobs.clock import FakeClock
from app.jobs.repository import JobRepository
from app.pipeline.checkpoints import CheckpointStore, Versions, file_sha256, stable_hash

LEASE_S = 60.0
VERSIONS = Versions(
    pipeline="1.0.0",
    schema="1",
    prompts={"writer": "w1", "critic": "c1"},
    topic_pack="1",
    renderer="manim-0.21.0+templates-1",
    voice="gemini/Kore",
)


@pytest.fixture
def repo(db_path: Path, clock: FakeClock) -> JobRepository:
    return JobRepository(db_path, clock)


@pytest.fixture
def store(tmp_path: Path, repo: JobRepository) -> CheckpointStore:
    return CheckpointStore(tmp_path / "work", repo)


def _claim(repo: JobRepository, worker: str = "w") -> Lease:
    res = repo.claim(worker_id=worker, lease_s=LEASE_S, max_claims=3)
    assert res is not None
    return res[1]


@pytest.fixture
def lease(repo: JobRepository) -> Lease:
    repo.create(
        query="How does the pH scale work?",
        topic_id="ph_scale",
        pipeline_version="1.0.0",
        idempotency_key=None,
        request_fingerprint=None,
    )
    return _claim(repo)


def _promote(store: CheckpointStore, lease: Lease, *, input_hash: str = "in-1") -> Path:
    scratch = store.scratch_dir(lease, "voicing")
    (scratch / "clip-00.wav").write_bytes(b"RIFF-0")
    (scratch / "clip-01.wav").write_bytes(b"RIFF-1")
    store.promote(
        lease,
        stage_key="voicing",
        input_hash=input_hash,
        versions=VERSIONS,
        files=["clip-00.wav", "clip-01.wav"],
        payload={"provider": "fake"},
    )
    return scratch


def _load(store: CheckpointStore, lease: Lease, **overrides: object) -> tuple[Path, dict] | None:
    kwargs: dict = {"stage_key": "voicing", "input_hash": "in-1", "versions": VERSIONS}
    kwargs.update(overrides)
    return store.load_valid(lease.job_id, **kwargs)


def test_stable_hash_ignores_key_order() -> None:
    assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_promote_then_load_valid(store: CheckpointStore, lease: Lease, tmp_path: Path) -> None:
    scratch = _promote(store, lease)

    assert scratch == tmp_path / "work" / lease.job_id / "attempt-1" / "voicing"
    stage = json.loads((scratch / "stage.json").read_text(encoding="utf-8"))
    assert stage["files"] == {
        "clip-00.wav": file_sha256(scratch / "clip-00.wav"),
        "clip-01.wav": file_sha256(scratch / "clip-01.wav"),
    }
    assert (stage["stage_key"], stage["attempt"], stage["input_hash"]) == ("voicing", 1, "in-1")
    assert _load(store, lease) == (scratch, {"provider": "fake"})
    assert store.attempts_map(lease.job_id) == {"voicing": 1}


def test_scratch_dir_nests_scene_stage_keys(store: CheckpointStore, lease: Lease) -> None:
    path = store.scratch_dir(lease, "rendering/scene-03")
    assert path.is_dir() and path.parts[-2:] == ("rendering", "scene-03")


def test_invalid_when_row_missing(store: CheckpointStore, lease: Lease) -> None:
    assert _load(store, lease) is None


def test_invalid_when_dir_missing(store: CheckpointStore, lease: Lease) -> None:
    scratch = _promote(store, lease)
    for child in scratch.iterdir():
        child.unlink()
    scratch.rmdir()
    assert _load(store, lease) is None


def test_invalid_when_stage_json_hash_mismatch(store: CheckpointStore, lease: Lease) -> None:
    scratch = _promote(store, lease)
    stage = scratch / "stage.json"
    stage.write_text(stage.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert _load(store, lease) is None


def test_invalid_when_file_hash_mismatch(store: CheckpointStore, lease: Lease) -> None:
    scratch = _promote(store, lease)
    (scratch / "clip-01.wav").write_bytes(b"RIFF-truncated")
    assert _load(store, lease) is None


def test_invalid_when_file_missing(store: CheckpointStore, lease: Lease) -> None:
    scratch = _promote(store, lease)
    (scratch / "clip-00.wav").unlink()
    assert _load(store, lease) is None


def test_invalid_when_input_hash_mismatch(store: CheckpointStore, lease: Lease) -> None:
    _promote(store, lease)
    assert _load(store, lease, input_hash="in-2") is None


@pytest.mark.parametrize(
    "change",
    [{"pipeline": "1.0.1"}, {"schema": "2"}, {"prompts": {"writer": "w2", "critic": "c1"}},
     {"topic_pack": "2"}, {"renderer": "manim-0.21.0+templates-2"}, {"voice": "edge/Aria"}],
)  # fmt: skip
def test_invalid_when_versions_mismatch(
    store: CheckpointStore, lease: Lease, change: dict[str, object]
) -> None:
    _promote(store, lease)
    changed = Versions(**{**VERSIONS.__dict__, **change})  # type: ignore[arg-type]
    assert _load(store, lease, versions=changed) is None


def test_promote_with_stale_lease_rejected_and_row_unchanged(
    store: CheckpointStore, repo: JobRepository, lease: Lease, clock: FakeClock
) -> None:
    _promote(store, lease)
    row = repo.get_checkpoint(lease.job_id, "voicing")
    clock.advance(LEASE_S + 1)
    _claim(repo, "B")

    with pytest.raises(StaleLeaseError):
        _promote(store, lease, input_hash="in-stale")

    assert repo.get_checkpoint(lease.job_id, "voicing") == row
    assert _load(store, lease) is not None  # the promoted output is still the valid one


def test_newer_attempt_supersedes_older(
    store: CheckpointStore, repo: JobRepository, lease: Lease, clock: FakeClock
) -> None:
    old = _promote(store, lease)
    clock.advance(LEASE_S + 1)
    lease_b = _claim(repo, "B")
    new = _promote(store, lease_b, input_hash="in-2")

    assert old != new and new.parent.name == "attempt-2"
    assert _load(store, lease_b, input_hash="in-2") == (new, {"provider": "fake"})
    assert _load(store, lease_b, input_hash="in-1") is None
    assert store.attempts_map(lease.job_id) == {"voicing": 2}
