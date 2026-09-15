from pathlib import Path

import pytest

from app.config import Settings
from app.render.ffmpeg import count_frames
from app.render.manim_renderer import ManimRenderer
from app.render.plan import build_render_plan
from app.render.protocol import scene_video_path
from app.topics.registry import TOPICS_ROOT, TopicRegistry

pytestmark = pytest.mark.render

SETTINGS = Settings(_env_file=None, width=640, height=360, fps=15)  # type: ignore[call-arg]


def test_render_curated_ph_plan_scene_0_and_1(tmp_path: Path) -> None:
    storyboard = TopicRegistry.load(TOPICS_ROOT).curated_storyboard("ph_scale")
    clips = [4.0 + 0.113 * i for i in range(len(storyboard.scenes))]
    plan = build_render_plan(
        job_id="job-render",
        question="How does the <b>pH</b> scale work?",
        storyboard=storyboard,
        clip_durations_s=clips,
        settings=SETTINGS,
    )

    result = ManimRenderer(SETTINGS).render(plan, tmp_path, scenes=[1, 0])

    assert result.renderer == "manim"
    assert [s.index for s in result.scenes] == [0, 1]
    for scene in result.scenes:
        assert scene.ok, scene
        assert scene.message is None
        assert scene.path == scene_video_path(tmp_path, scene.index)
        assert scene.frames == count_frames(scene.path) == plan.scenes[scene.index].frames
    assert not scene_video_path(tmp_path, 2).exists()
    assert result.cpu_seconds >= 0.0
