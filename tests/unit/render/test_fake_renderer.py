from pathlib import Path

import pytest

from app.config import Settings
from app.render.fake import FakeRenderer
from app.render.ffmpeg import count_frames, ffprobe
from app.render.plan import RenderPlan, build_render_plan
from app.render.protocol import scene_video_path
from app.topics.registry import TOPICS_ROOT, TopicRegistry

pytestmark = pytest.mark.integration  # the fake renders with ffmpeg

SETTINGS = Settings(_env_file=None, width=320, height=180, fps=15)  # type: ignore[call-arg]


def _plan() -> RenderPlan:
    sb = TopicRegistry.load(TOPICS_ROOT).curated_storyboard("ph_scale")
    clips = [1.0 + 0.017 * i for i in range(len(sb.scenes))]
    return build_render_plan(
        job_id="job1", question="q", storyboard=sb, clip_durations_s=clips, settings=SETTINGS
    )


def test_fake_exact_frames(tmp_path: Path) -> None:
    plan = _plan()
    renderer = FakeRenderer(script={4: "off_by_one", 5: "black"})
    result = renderer.render(plan, tmp_path)
    assert (result.renderer, result.version) == ("fake", "fake-1")
    assert result.failed() == []
    assert renderer.calls == [list(range(len(plan.scenes)))]
    for scene, planned in zip(result.scenes, plan.scenes, strict=True):
        assert scene.ok and scene.path == scene_video_path(tmp_path, planned.index)
        expected = planned.frames + (1 if planned.index == 4 else 0)
        assert scene.frames == count_frames(scene.path) == expected
    [video] = [s for s in ffprobe(result.scenes[0].path)["streams"] if s["codec_type"] == "video"]  # type: ignore[arg-type]
    assert (video["width"], video["height"], video["r_frame_rate"]) == (320, 180, "15/1")


def test_fake_crash_marks_scene(tmp_path: Path) -> None:
    result = FakeRenderer(script={1: "crash", 2: "template_error"}).render(_plan(), tmp_path)
    failed = {s.index: s for s in result.failed()}
    assert set(failed) == {1, 2}
    assert (failed[1].error_type, failed[1].path, failed[1].frames) == ("crash", None, None)
    assert failed[2].error_type == "template_error"
    assert not scene_video_path(tmp_path, 1).exists()
    assert scene_video_path(tmp_path, 0).exists()


def test_scenes_subset_respected(tmp_path: Path) -> None:
    renderer = FakeRenderer()
    result = renderer.render(_plan(), tmp_path, scenes=[3, 1])
    assert [s.index for s in result.scenes] == [1, 3]
    assert renderer.calls == [[1, 3]]
    assert sorted(p.name for p in tmp_path.glob("*.mp4")) == ["scene-01.mp4", "scene-03.mp4"]
