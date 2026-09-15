"""Renderer boundary (spec §8.1): a renderer turns a RenderPlan into silent per-scene MP4s."""

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from app.render.plan import RenderPlan

RenderErrorType = Literal["template_error", "layout_error", "crash", "timeout"]


@dataclass(frozen=True)
class SceneRenderResult:
    index: int
    ok: bool
    path: Path | None
    frames: int | None
    error_type: RenderErrorType | None
    message: str | None


@dataclass
class RenderResult:
    scenes: list[SceneRenderResult]
    renderer: str
    version: str
    cpu_seconds: float

    def failed(self) -> list[SceneRenderResult]:
        return [s for s in self.scenes if not s.ok]


class Renderer(Protocol):
    name: str
    version: str

    def render(
        self,
        plan: RenderPlan,
        out_dir: Path,
        *,
        scenes: Sequence[int] | None = None,
        cancel: threading.Event | None = None,
    ) -> RenderResult: ...


def scene_video_path(out_dir: Path, index: int) -> Path:
    return out_dir / f"scene-{index:02d}.mp4"
