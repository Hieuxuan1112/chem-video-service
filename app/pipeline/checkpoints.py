"""DB-authoritative, hash-verified stage checkpoints (spec 6).

The stage_checkpoints row decides which attempt's output is current. stage.json lists every file
with its sha256; it is written under a temporary name and only moved into place after the fenced
promotion commits, so a stale worker can never replace the manifest of a promoted checkpoint.
"""

import contextlib
import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.domain.job import Lease
from app.jobs.repository import JobRepository

STAGE_FILE = "stage.json"
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Versions:
    pipeline: str
    schema: str
    prompts: Mapping[str, str]
    topic_pack: str
    renderer: str
    voice: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "schema": self.schema,
            "prompts": dict(sorted(self.prompts.items())),
            "topic_pack": self.topic_pack,
            "renderer": self.renderer,
            "voice": self.voice,
        }


def stable_hash(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointStore:
    def __init__(self, work_dir: Path, repo: JobRepository) -> None:
        self._work = work_dir
        self._repo = repo

    def _dir(self, job_id: str, attempt: int, stage_key: str) -> Path:
        return self._work / job_id / f"attempt-{attempt}" / stage_key

    def scratch_dir(self, lease: Lease, stage_key: str) -> Path:
        path = self._dir(lease.job_id, lease.attempt, stage_key)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def promote(
        self,
        lease: Lease,
        *,
        stage_key: str,
        input_hash: str,
        versions: Versions,
        files: Sequence[str],
        payload: dict[str, Any],
    ) -> None:
        directory = self.scratch_dir(lease, stage_key)
        stage = {
            "stage_key": stage_key,
            "attempt": lease.attempt,
            "input_hash": input_hash,
            "versions": versions.to_dict(),
            "files": {name: file_sha256(directory / name) for name in files},
            "payload": payload,
        }
        data = json.dumps(stage, sort_keys=True, indent=2).encode("utf-8")
        tmp = directory / f"{STAGE_FILE}.tmp-{uuid.uuid4().hex}"
        tmp.write_bytes(data)
        try:
            self._repo.promote_checkpoint(
                lease,
                stage_key=stage_key,
                input_hash=input_hash,
                manifest_sha256=hashlib.sha256(data).hexdigest(),
                versions=versions.to_dict(),
            )
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        os.replace(tmp, directory / STAGE_FILE)

    def load_valid(
        self, job_id: str, *, stage_key: str, input_hash: str, versions: Versions
    ) -> tuple[Path, dict[str, Any]] | None:
        row = self._repo.get_checkpoint(job_id, stage_key)
        if row is None or row.input_hash != input_hash or row.versions != versions.to_dict():
            return None
        directory = self._dir(job_id, row.attempt, stage_key)
        stage_path = directory / STAGE_FILE
        if not stage_path.is_file() or file_sha256(stage_path) != row.manifest_sha256:
            return None
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        for name, sha in stage["files"].items():
            path = directory / name
            if not path.is_file() or file_sha256(path) != sha:
                return None
        payload: dict[str, Any] = stage["payload"]
        return directory, payload

    def attempts_map(self, job_id: str) -> dict[str, int]:
        return {row.stage_key: row.attempt for row in self._repo.list_checkpoints(job_id)}
