"""Atomic local artifact publication (spec 9.1): the manifest is written last and marks completion.

A crash can leave copied files but never a manifest that lies about them; verify() accepts a job
only when manifest.json exists and every listed file matches its sha256 and size.
"""

import contextlib
import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from app.domain.errors import AppError, ErrorCode

MANIFEST = "manifest.json"
CONTENT_TYPES: Mapping[str, str] = {
    "video.mp4": "video/mp4",
    "thumbnail.jpg": "image/jpeg",
    "captions.vtt": "text/vtt",
    "storyboard.json": "application/json",
    "quality_report.json": "application/json",
    "contact_sheet.jpg": "image/jpeg",
    MANIFEST: "application/json",
}
_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CHUNK = 1 << 20


class Manifest(BaseModel):
    job_id: str
    pipeline_version: str
    schema_version: str
    prompt_versions: dict[str, Any]
    models: dict[str, Any]
    voice: dict[str, Any]
    renderer_version: str
    topic_pack_version: str
    generation_source: str
    fallback_reason: str | None
    checkpoint_attempts: dict[str, int]
    cost: dict[str, Any]
    stage_durations_s: dict[str, Any]
    files: dict[str, dict[str, Any]]  # name -> {sha256, bytes, content_type}
    created_at: float


class ArtifactStore(Protocol):
    def publish(
        self, job_id: str, files: Mapping[str, Path], manifest: Manifest, *, attempt: int
    ) -> None: ...

    def open(self, job_id: str, name: str) -> Path | None: ...

    def verify(self, job_id: str) -> Manifest | None: ...

    def list_refs(self, job_id: str) -> list[dict[str, Any]]: ...

    def manifest(self, job_id: str) -> dict[str, Any] | None: ...

    def url_for(self, job_id: str, name: str) -> str: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def _job_dir(self, job_id: str) -> Path | None:
        return self._root / job_id if _JOB_ID.fullmatch(job_id) else None

    def url_for(self, job_id: str, name: str) -> str:
        return f"/v1/videos/{job_id}/artifacts/{name}"

    def publish(
        self, job_id: str, files: Mapping[str, Path], manifest: Manifest, *, attempt: int
    ) -> None:
        job_dir = self._job_dir(job_id)
        unknown = [n for n in files if n not in CONTENT_TYPES or n == MANIFEST]
        if job_dir is None or unknown:
            raise ValueError(f"refusing to publish {unknown or job_id!r}")
        try:
            job_dir.mkdir(parents=True, exist_ok=True)
            entries: dict[str, dict[str, Any]] = {}
            for name, source in files.items():
                self._place(job_dir, name, attempt, lambda tmp, s=source: shutil.copyfile(s, tmp))
                target = job_dir / name
                entries[name] = {
                    "sha256": _sha256(target),
                    "bytes": target.stat().st_size,
                    "content_type": CONTENT_TYPES[name],
                }
            final = manifest.model_copy(update={"files": entries})
            data = json.dumps(final.model_dump(), sort_keys=True, indent=2).encode("utf-8")
            self._place(job_dir, MANIFEST, attempt, lambda tmp: tmp.write_bytes(data))
        except OSError as exc:
            raise AppError(
                ErrorCode.ARTIFACT_STORE_ERROR,
                "io_error",
                f"publishing {job_id} failed: {exc}",
                {"job_id": job_id},
            ) from exc

    @staticmethod
    def _place(job_dir: Path, name: str, attempt: int, write: Any) -> None:
        tmp = job_dir / f"{name}.tmp-{attempt}"
        try:
            write(tmp)
            os.replace(tmp, job_dir / name)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

    def open(self, job_id: str, name: str) -> Path | None:
        job_dir = self._job_dir(job_id)
        if job_dir is None or name not in CONTENT_TYPES:
            return None
        path = job_dir / name
        return path if path.is_file() else None

    def verify(self, job_id: str) -> Manifest | None:
        job_dir = self._job_dir(job_id)
        if job_dir is None or not (job_dir / MANIFEST).is_file():
            return None
        try:
            manifest = Manifest.model_validate_json((job_dir / MANIFEST).read_bytes())
        except (ValidationError, OSError):
            return None
        if manifest.job_id != job_id:
            return None
        for name, entry in manifest.files.items():
            path = job_dir / name
            if name not in CONTENT_TYPES or not path.is_file():
                return None
            if path.stat().st_size != entry.get("bytes") or _sha256(path) != entry.get("sha256"):
                return None
        return manifest

    def list_refs(self, job_id: str) -> list[dict[str, Any]]:
        manifest = self.verify(job_id)
        if manifest is None:
            return []
        manifest_path = self._root / job_id / MANIFEST
        entries = {
            **manifest.files,
            MANIFEST: {
                "sha256": _sha256(manifest_path),
                "bytes": manifest_path.stat().st_size,
                "content_type": CONTENT_TYPES[MANIFEST],
            },
        }
        return [
            {"name": name, **entry, "url": self.url_for(job_id, name)}
            for name, entry in sorted(entries.items())
        ]

    def manifest(self, job_id: str) -> dict[str, Any] | None:
        verified = self.verify(job_id)
        if verified is None:
            return None
        data: dict[str, Any] = json.loads((self._root / job_id / MANIFEST).read_text("utf-8"))
        return data
