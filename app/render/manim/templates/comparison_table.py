"""comparison_table: headers first, then each row reveals at its beat."""

from manim import FadeIn, Line, Scene

from app.render.layout import LAYOUT
from app.render.manim import theme
from app.render.manim.scene_factory import (
    play_frames,
    reveal_at_beat,
    seconds_to_frames,
    text_in_slot,
)
from app.render.plan import PlannedScene, RenderPlan, ResolvedComparisonTable


def build(scene: Scene, planned: PlannedScene, plan: RenderPlan) -> None:
    v = planned.visual
    assert isinstance(v, ResolvedComparisonTable)
    left = text_in_slot(v.left_header, "comparison_table.header.left", color=theme.ACCENT)
    right = text_in_slot(v.right_header, "comparison_table.header.right", color=theme.ACCENT)
    header = LAYOUT["comparison_table.header.left"]
    rule_y = header.y + header.h + 6
    rule = Line(
        theme.point3(90, rule_y),
        theme.point3(1180, rule_y),
        color=theme.MUTED,
        stroke_width=2,
    )
    play_frames(scene, FadeIn(left), FadeIn(right), FadeIn(rule), frames=seconds_to_frames(0.5))
    for i, row in enumerate(v.rows):
        cells = [
            text_in_slot(
                row.aspect, f"comparison_table.aspect.{i}", color=theme.MUTED, align="left"
            ),
            text_in_slot(row.left, f"comparison_table.left.{i}", align="left"),
            text_in_slot(row.right, f"comparison_table.right.{i}", align="left"),
        ]
        reveal_at_beat(scene, planned, f"row:{i}", *[FadeIn(c) for c in cells], seconds=0.5)
