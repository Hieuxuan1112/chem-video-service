import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.render import manim_renderer
from app.render.manim_renderer import ManimRenderer
from app.render.plan import Beat, PlannedScene, RenderPlan, ResolvedTitleCard
from app.render.proc import ProcResult

SETTINGS = Settings(_env_file=None, render_timeout_s=5.0)  # type: ignore[call-arg]
FRAMES = 60


def _plan(scenes: int = 3) -> RenderPlan:
    visual = ResolvedTitleCard(type="title_card", question="q", subtitle="s", title="t")
    planned = [
        PlannedScene(
            index=i,
            start_frame=i * FRAMES,
            end_frame=(i + 1) * FRAMES,
            duration_s=2.0,
            lead_in_s=0.3,
            clip_duration_s=1.0,
            on_screen_title="Title",
            narration="n",
            visual=visual,
            beats=[Beat(target="question", at_s=0.3)],
        )
        for i in range(scenes)
    ]
    return RenderPlan(
        job_id="job", width=640, height=360, fps=30, question="q", total_frames=FRAMES * scenes,
        scenes=planned,
    )  # fmt: skip


def _scenes_arg(args: Sequence[str]) -> list[int]:
    return [int(i) for i in args[args.index("--scenes") + 1].split(",")]


def _out_arg(args: Sequence[str]) -> Path:
    return Path(args[args.index("--out") + 1])


def _write(out: Path, index: int, **fields: Any) -> None:
    result = {
        "ok": False,
        "frames": None,
        "error_type": None,
        "message": None,
        "traceback_tail": None,
    }
    result.update(fields)
    (out / f"scene-{index:02d}.result.json").write_text(json.dumps(result), encoding="utf-8")
    if result["ok"]:
        (out / f"scene-{index:02d}.mp4").write_bytes(b"video")


Behaviour = Callable[[int, list[int], Path], ProcResult]


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> Callable[[Behaviour], list[list[int]]]:
    calls: list[list[int]] = []

    def install(behaviour: Behaviour) -> list[list[int]]:
        def run_subprocess(args: Sequence[str], **kwargs: Any) -> ProcResult:
            assert kwargs["timeout_s"] == SETTINGS.render_timeout_s
            indices = _scenes_arg(args)
            calls.append(indices)
            return behaviour(len(calls), indices, _out_arg(args))

        monkeypatch.setattr(manim_renderer, "run_subprocess", run_subprocess)
        monkeypatch.setattr(manim_renderer, "count_frames", lambda path: FRAMES)
        return calls

    return install


def _ok(returncode: int = 0, *, timed_out: bool = False, stderr: str = "") -> ProcResult:
    return ProcResult(returncode, b"", stderr, timed_out)


def test_missing_result_file_is_crash_and_retried_once(
    tmp_path: Path, fake_runner: Callable[[Behaviour], list[list[int]]]
) -> None:
    def behaviour(call: int, indices: list[int], out: Path) -> ProcResult:
        if call == 1:
            _write(out, 0, ok=True, frames=FRAMES)  # scenes 1 and 2 die with the process
            return _ok(-9, stderr="Segmentation fault")
        return _ok(-9, stderr="Segmentation fault again")

    calls = fake_runner(behaviour)
    result = ManimRenderer(SETTINGS).render(_plan(), tmp_path)

    assert calls == [[0, 1, 2], [1, 2]]
    by_index = {s.index: s for s in result.scenes}
    assert by_index[0].ok and by_index[0].path == tmp_path / "scene-00.mp4"
    for index in (1, 2):
        assert not by_index[index].ok
        assert by_index[index].error_type == "crash"
        assert "again" in (by_index[index].message or "")
    assert [s.index for s in result.failed()] == [1, 2]


def test_crash_retry_can_succeed(
    tmp_path: Path, fake_runner: Callable[[Behaviour], list[list[int]]]
) -> None:
    def behaviour(call: int, indices: list[int], out: Path) -> ProcResult:
        if call == 2:
            for index in indices:
                _write(out, index, ok=True, frames=FRAMES)
        return _ok(0 if call == 2 else 1)

    calls = fake_runner(behaviour)
    result = ManimRenderer(SETTINGS).render(_plan(2), tmp_path, scenes=[1])

    assert calls == [[1], [1]]
    assert [(s.index, s.ok) for s in result.scenes] == [(1, True)]


def test_timeout_marks_unfinished(
    tmp_path: Path, fake_runner: Callable[[Behaviour], list[list[int]]]
) -> None:
    def behaviour(call: int, indices: list[int], out: Path) -> ProcResult:
        if call == 1:
            _write(out, 0, ok=True, frames=FRAMES)
        return _ok(1, timed_out=True)

    calls = fake_runner(behaviour)
    result = ManimRenderer(SETTINGS).render(_plan(), tmp_path)

    assert calls == [[0, 1, 2], [1, 2]]
    assert [(s.index, s.ok, s.error_type) for s in result.scenes] == [
        (0, True, None),
        (1, False, "timeout"),
        (2, False, "timeout"),
    ]


def test_template_and_layout_errors_are_not_retried(
    tmp_path: Path, fake_runner: Callable[[Behaviour], list[list[int]]]
) -> None:
    def behaviour(call: int, indices: list[int], out: Path) -> ProcResult:
        _write(out, 0, error_type="template_error", message="KeyError: 'x'")
        _write(out, 1, error_type="layout_error", message="outside the safe area")
        return _ok()

    calls = fake_runner(behaviour)
    result = ManimRenderer(SETTINGS).render(_plan(2), tmp_path)

    assert calls == [[0, 1]]
    assert [s.error_type for s in result.scenes] == ["template_error", "layout_error"]
    assert result.scenes[0].message == "KeyError: 'x'"


def test_stale_outputs_are_removed_before_rendering(
    tmp_path: Path, fake_runner: Callable[[Behaviour], list[list[int]]]
) -> None:
    _write(tmp_path, 0, ok=True, frames=FRAMES)  # left over from an earlier attempt
    calls = fake_runner(lambda call, indices, out: _ok(-9))

    result = ManimRenderer(SETTINGS).render(_plan(1), tmp_path)

    assert calls == [[0], [0]]
    assert result.scenes[0].error_type == "crash"
    assert not (tmp_path / "scene-00.mp4").exists()


@pytest.mark.parametrize(
    ("actual", "ok", "error_type", "note"),
    [(FRAMES, True, None, False), (FRAMES + 1, True, None, True), (FRAMES - 1, True, None, True),
     (FRAMES + 2, False, "template_error", True)],
)  # fmt: skip
def test_frame_mismatch_classification(
    tmp_path: Path,
    fake_runner: Callable[[Behaviour], list[list[int]]],
    monkeypatch: pytest.MonkeyPatch,
    actual: int,
    ok: bool,
    error_type: str | None,
    note: bool,
) -> None:
    def behaviour(call: int, indices: list[int], out: Path) -> ProcResult:
        _write(out, 0, ok=True, frames=actual)
        return _ok()

    calls = fake_runner(behaviour)
    monkeypatch.setattr(manim_renderer, "count_frames", lambda path: actual)
    result = ManimRenderer(SETTINGS).render(_plan(1), tmp_path)

    [scene] = result.scenes
    assert calls == [[0]]
    assert (scene.ok, scene.error_type, scene.frames) == (ok, error_type, actual)
    assert (scene.message is not None) == note
    if note:
        assert f"{actual} frames, planned {FRAMES}" in (scene.message or "")


def test_version_names_manim_and_templates() -> None:
    renderer = ManimRenderer(SETTINGS)
    assert renderer.name == "manim"
    assert renderer.version.startswith("manim-0.21.0+templates-")
