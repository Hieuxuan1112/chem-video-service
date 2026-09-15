import json
from pathlib import Path

import pytest

from app.render.ffmpeg import count_frames
from app.render.manim import runner, scene_factory
from app.render.plan import (
    Beat,
    PlannedScene,
    RenderPlan,
    ResolvedLogRatio,
    ResolvedTitleCard,
)

pytestmark = pytest.mark.render

REPO_ROOT = Path(__file__).resolve().parents[2]
FONTS = REPO_ROOT / "assets" / "fonts"


def _title_card_plan() -> RenderPlan:
    scene = PlannedScene(
        index=0,
        start_frame=0,
        end_frame=60,
        duration_s=2.0,
        lead_in_s=0.3,
        clip_duration_s=1.0,
        on_screen_title="Your question",
        narration="Let's explore the pH scale.",
        visual=ResolvedTitleCard(
            type="title_card",
            question="What does the pH scale measure?",
            subtitle="Acids, bases and neutral",
            title="How the pH scale works",
        ),
        beats=[Beat(target="question", at_s=0.3), Beat(target="subtitle", at_s=0.8)],
    )
    return RenderPlan(
        job_id="job-test",
        width=640,
        height=360,
        fps=30,
        question="What does the pH scale measure?",
        total_frames=60,
        scenes=[scene],
    )


def _run(tmp_path: Path) -> tuple[int, Path]:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(_title_card_plan().model_dump_json(), encoding="utf-8")
    out = tmp_path / "out"
    code = runner.main(["--plan", str(plan_path), "--out", str(out), "--fonts", str(FONTS)])
    return code, out


def test_runner_renders_title_card_scene_and_writes_result(tmp_path: Path) -> None:
    code, out = _run(tmp_path)

    assert code == 0
    result = json.loads((out / "scene-00.result.json").read_text(encoding="utf-8"))
    assert result["ok"] is True, result
    assert result["error_type"] is None
    assert result["frames"] == 60
    assert count_frames(out / "scene-00.mp4") == 60
    assert not (out / "_media" / "scene-00").exists()


def test_runner_reports_template_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object) -> None:
        raise ValueError("template exploded")

    monkeypatch.setitem(scene_factory.TEMPLATE_REGISTRY, "title_card", boom)
    code, out = _run(tmp_path)

    assert code == 0
    result = json.loads((out / "scene-00.result.json").read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert result["error_type"] == "template_error"
    assert "template exploded" in result["message"]
    assert "ValueError" in result["traceback_tail"]
    assert not (out / "scene-00.mp4").exists()


def test_runner_fits_reveals_crowded_at_scene_end(tmp_path: Path) -> None:
    """Every beat at the latest allowed time: holds are cut short, animations keep their speed."""
    fps, clip = 15, 5.0
    last_beat = 0.3 + clip - 0.5
    scene = PlannedScene(
        index=0,
        start_frame=0,
        end_frame=90,
        duration_s=6.0,
        lead_in_s=0.3,
        clip_duration_s=clip,
        on_screen_title="Ten times",
        narration="n",
        visual=ResolvedLogRatio(type="log_ratio", from_ph=4, to_ph=3, ratio=10),
        beats=[Beat(target=t, at_s=last_beat) for t in ("from", "to", "ratio")],
    )
    plan = RenderPlan(
        job_id="job-crowded",
        width=320,
        height=180,
        fps=fps,
        question="q",
        total_frames=90,
        scenes=[scene],
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    out = tmp_path / "out"

    assert runner.main(["--plan", str(plan_path), "--out", str(out), "--fonts", str(FONTS)]) == 0
    result = json.loads((out / "scene-00.result.json").read_text(encoding="utf-8"))
    assert result["ok"] is True, result
    assert result["frames"] == 90
    assert count_frames(out / "scene-00.mp4") == 90
