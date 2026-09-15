"""ManimRenderer (spec §8.2): templates run in one killable subprocess for all requested scenes.

Classification here is per scene and renderer-level only; mapping to AppError codes happens in the
orchestrator. Crashed or timed-out scenes get one fresh subprocess; template and layout errors are
code bugs and are never retried.
"""

import contextlib
import json
import sys
import threading
from collections.abc import Sequence
from importlib.metadata import version as package_version
from pathlib import Path

import psutil

from app.config import Settings
from app.render.ffmpeg import count_frames
from app.render.plan import RenderPlan
from app.render.proc import ProcResult, run_subprocess
from app.render.protocol import (
    RenderErrorType,
    RenderResult,
    SceneRenderResult,
    scene_video_path,
)

TEMPLATES_VERSION = "1"
MAX_FRAME_DRIFT = 1  # assembly trims or clones one frame; more is a template bug
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MESSAGE_TAIL_CHARS = 500


def _result_path(out_dir: Path, index: int) -> Path:
    return out_dir / f"scene-{index:02d}.result.json"


def _children_cpu_seconds() -> float:
    """User + system CPU of reaped child processes (POSIX); 0.0 where psutil cannot report it."""
    times = psutil.Process().cpu_times()
    return float(getattr(times, "children_user", 0.0) + getattr(times, "children_system", 0.0))


class ManimRenderer:
    name = "manim"

    def __init__(self, settings: Settings, *, python_exe: str = sys.executable) -> None:
        self._settings = settings
        self._python = python_exe
        self.version = f"manim-{package_version('manim')}+templates-{TEMPLATES_VERSION}"

    def render(
        self,
        plan: RenderPlan,
        out_dir: Path,
        *,
        scenes: Sequence[int] | None = None,
        cancel: threading.Event | None = None,
    ) -> RenderResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        plan_path = out_dir / "plan.json"
        plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
        indices = sorted(scenes) if scenes is not None else [s.index for s in plan.scenes]
        cpu_before = _children_cpu_seconds()

        results = self._run(plan, plan_path, out_dir, indices, cancel)
        retry = [i for i in indices if results[i].error_type in ("crash", "timeout")]
        if retry and not (cancel is not None and cancel.is_set()):
            results.update(self._run(plan, plan_path, out_dir, retry, cancel))

        return RenderResult(
            scenes=[results[i] for i in indices],
            renderer=self.name,
            version=self.version,
            cpu_seconds=max(0.0, _children_cpu_seconds() - cpu_before),
        )

    def _run(
        self,
        plan: RenderPlan,
        plan_path: Path,
        out_dir: Path,
        indices: list[int],
        cancel: threading.Event | None,
    ) -> dict[int, SceneRenderResult]:
        for index in indices:  # a stale file from an earlier attempt must not look like success
            with contextlib.suppress(FileNotFoundError):
                _result_path(out_dir, index).unlink()
            with contextlib.suppress(FileNotFoundError):
                scene_video_path(out_dir, index).unlink()
        fonts = self._settings.fonts_dir
        proc = run_subprocess(
            [
                self._python,
                "-m",
                "app.render.manim.runner",
                "--plan",
                str(plan_path.resolve()),
                "--out",
                str(out_dir.resolve()),
                "--scenes",
                ",".join(str(i) for i in indices),
                "--fonts",
                str(fonts if fonts.is_absolute() else _REPO_ROOT / fonts),
            ],
            cwd=_REPO_ROOT,
            timeout_s=self._settings.render_timeout_s,
            cancel=cancel,
        )
        return {i: self._scene_result(plan, out_dir, i, proc) for i in indices}

    def _scene_result(
        self, plan: RenderPlan, out_dir: Path, index: int, proc: ProcResult
    ) -> SceneRenderResult:
        path = _result_path(out_dir, index)
        if not path.is_file():
            error: RenderErrorType = "timeout" if proc.timed_out else "crash"
            tail = proc.stderr_tail[-_MESSAGE_TAIL_CHARS:]
            message = f"no result file (runner exit {proc.returncode}): {tail}"
            return SceneRenderResult(index, False, None, None, error, message)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data["ok"]:
            return SceneRenderResult(index, False, None, None, data["error_type"], data["message"])

        video = scene_video_path(out_dir, index)
        frames = count_frames(video)
        planned = plan.scenes[index].frames
        drift = frames - planned
        if drift == 0:
            return SceneRenderResult(index, True, video, frames, None, None)
        message = f"{frames} frames, planned {planned} (drift {drift:+d})"
        if abs(drift) <= MAX_FRAME_DRIFT:
            return SceneRenderResult(index, True, video, frames, None, message)
        return SceneRenderResult(index, False, video, frames, "template_error", message)
