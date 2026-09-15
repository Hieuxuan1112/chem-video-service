"""key_points: the learner's question is recapped first, then a tick and a point appear at each
beat."""

from manim import Create, FadeIn, Scene

from app.render.layout import LAYOUT
from app.render.manim import theme
from app.render.manim.scene_factory import (
    fit_user_text,
    play_frames,
    reveal_at_beat,
    seconds_to_frames,
    text_in_slot,
    tick_mark,
)
from app.render.plan import PlannedScene, RenderPlan, ResolvedKeyPoints


def build(scene: Scene, planned: PlannedScene, plan: RenderPlan) -> None:
    v = planned.visual
    assert isinstance(v, ResolvedKeyPoints)
    recap = fit_user_text(v.question, "key_points.recap", min_px=16)
    recap.set_color(theme.MUTED)
    play_frames(scene, FadeIn(recap), frames=seconds_to_frames(0.4))
    for i, point in enumerate(v.points):
        slot = LAYOUT[f"key_points.point.{i}"]
        text = text_in_slot(point, slot.name, align="left")
        tick = tick_mark(slot.x - 34, slot.y + slot.h / 2)
        reveal_at_beat(scene, planned, f"point:{i}", Create(tick), FadeIn(text), seconds=0.5)
