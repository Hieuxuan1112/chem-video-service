"""Fast ffmpeg-based renderer for tests: exact-frame colour clips with scriptable failures."""

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.render.ffmpeg import make_color_video
from app.render.plan import RenderPlan
from app.render.protocol import (
    RenderErrorType,
    RenderResult,
    SceneRenderResult,
    scene_video_path,
)

_COLOR = "#0F172A"


class FakeRenderer:
    name = "fake"
    version = "fake-1"

    def __init__(self, script: Mapping[int, str] | None = None) -> None:
        """script maps scene index to "ok" | "crash" | "template_error" | "black" | "off_by_one" |
        "blank" (background only) | "same" (the picture of scene 0).

        "ok" draws a white box in a different grid cell per scene, so scenes are distinguishable.
        """
        self._script = dict(script or {})
        self.calls: list[list[int]] = []

    def render(
        self,
        plan: RenderPlan,
        out_dir: Path,
        *,
        scenes: Sequence[int] | None = None,
        cancel: threading.Event | None = None,
    ) -> RenderResult:
        indices = sorted(scenes) if scenes is not None else [s.index for s in plan.scenes]
        self.calls.append(indices)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for index in indices:
            planned = plan.scenes[index]
            mode = self._script.get(index, "ok")
            if mode == "crash" or mode == "template_error":
                error: RenderErrorType = "crash" if mode == "crash" else "template_error"
                results.append(
                    SceneRenderResult(index, False, None, None, error, f"scripted {mode}")
                )
                continue
            frames = planned.frames + (1 if mode == "off_by_one" else 0)
            path = scene_video_path(out_dir, index)
            make_color_video(
                path,
                frames=frames,
                fps=plan.fps,
                size=(plan.width, plan.height),
                color="black" if mode == "black" else _COLOR,
                box_cell=None if mode in ("black", "blank") else 0 if mode == "same" else index % 9,
            )
            results.append(SceneRenderResult(index, True, path, frames, None, None))
        return RenderResult(results, self.name, self.version, cpu_seconds=0.0)
