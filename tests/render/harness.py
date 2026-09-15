"""Builds template mobjects without rendering video: play/wait apply animations instantly."""

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from manim import Mobject, Scene, Text, VGroup, tempconfig

from app.render.layout import LAYOUT, FontSet, measure_wrap
from app.render.manim import scene_factory, theme
from app.render.plan import Beat, PlannedScene, RenderPlan, ResolvedVisual, beat_targets

REPO_ROOT = Path(__file__).resolve().parents[2]
FONTS_DIR = REPO_ROOT / "assets" / "fonts"
TOLERANCE_UNITS = theme.px_to_units(1)


def load_fonts() -> FontSet:
    fonts = theme.register_fonts(FONTS_DIR)
    scene_factory.set_fonts(fonts)
    return fonts


def widest_accepted(slot_name: str, max_chars: int, fonts: FontSet) -> str:
    """The longest string of wide glyphs (with word breaks) that G2d accepts for the slot."""
    slot = LAYOUT[slot_name]
    text = ""
    for ch in itertools.cycle("WMWMW "):
        candidate = text + ch
        if len(candidate.strip()) > max_chars or not measure_wrap(candidate, slot, fonts).fits:
            break
        text = candidate
    return text.strip()


def plan_for(visual: ResolvedVisual, *, question: str, seconds: float = 12.0) -> RenderPlan:
    targets = beat_targets(visual)
    step = (seconds - 2.0) / max(len(targets), 1)
    fps = 30
    scene = PlannedScene(
        index=0,
        start_frame=0,
        end_frame=round(seconds * fps),
        duration_s=seconds,
        lead_in_s=0.3,
        clip_duration_s=seconds - 1.0,
        on_screen_title=widest_accepted("on_screen_title", 40, load_fonts()),
        narration="narration",
        visual=visual,
        beats=[Beat(target=t, at_s=0.3 + i * step) for i, (t, _) in enumerate(targets)],
    )
    return RenderPlan(
        job_id="job-test",
        width=1280,
        height=720,
        fps=fps,
        question=question,
        total_frames=scene.end_frame,
        scenes=[scene.model_copy(update={"index": i}) for i in range(7)],  # the most dots
    )


@dataclass
class Built:
    scene: Scene
    groups: list[VGroup]  # every slotted text group the template created


def build_offscreen(plan: RenderPlan, monkeypatch: pytest.MonkeyPatch) -> Built:
    """Title + progress dots + template, as scene_factory's construct adds them, minus the fade."""
    planned = plan.scenes[0]
    load_fonts()
    groups: list[VGroup] = []
    original = scene_factory.text_lines

    def recording_text_lines(*args: Any, **kwargs: Any) -> VGroup:
        group = original(*args, **kwargs)
        groups.append(group)
        return group

    monkeypatch.setattr(scene_factory, "text_lines", recording_text_lines)
    with tempconfig(
        {"pixel_width": plan.width, "pixel_height": plan.height, "frame_rate": plan.fps}
    ):
        scene = Scene()

        def play(*animations: Any, run_time: float | None = None, **_: Any) -> None:
            for animation in animations:
                animation._setup_scene(scene)
                animation.begin()
                animation.finish()
                animation.clean_up_from_scene(scene)
            scene.renderer.time += run_time if run_time is not None else 1 / plan.fps

        def wait(duration: float = 1.0, **_: Any) -> None:
            scene.renderer.time += duration

        scene.play = play  # type: ignore[method-assign]
        scene.wait = wait  # type: ignore[method-assign]
        scene.add(scene_factory.text_in_slot(planned.on_screen_title, "on_screen_title"))
        scene.add(scene_factory.progress_dots(len(plan.scenes), planned.index))
        scene_factory.template_for(planned.visual.type)(scene, planned, plan)
    return Built(scene, groups)


def texts(scene: Scene) -> list[Text]:
    return [m for top in scene.mobjects for m in top.get_family() if isinstance(m, Text)]


def box(m: Any) -> tuple[float, float, float, float]:
    return (m.get_left()[0], m.get_bottom()[1], m.get_right()[0], m.get_top()[1])


def tagged(scene: Scene, attr: str) -> list[Mobject]:
    """Distinct mobjects in the scene family that carry a template tag attribute."""
    seen: dict[int, Mobject] = {}
    for top in scene.mobjects:
        for m in top.get_family():
            if hasattr(m, attr):
                seen[id(m)] = m
    return list(seen.values())


def assert_no_text_overlap(scene: Scene) -> None:
    boxes = [box(t) for t in texts(scene)]
    assert len(boxes) >= 3
    for a, b in itertools.combinations(boxes, 2):
        disjoint = (
            a[2] <= b[0] + TOLERANCE_UNITS
            or b[2] <= a[0] + TOLERANCE_UNITS
            or a[3] <= b[1] + TOLERANCE_UNITS
            or b[3] <= a[1] + TOLERANCE_UNITS
        )
        assert disjoint, (a, b)
