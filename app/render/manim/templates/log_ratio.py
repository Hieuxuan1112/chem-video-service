"""log_ratio: two H+ dot grids, one dot for the higher pH and `ratio` dots for the lower pH
(10 as 2 x 5, 100 as 10 x 10), then the resolved ratio label with an arrow to the richer side."""

from manim import Arrow, Dot, FadeIn, GrowArrow, Scene, VGroup

from app.render.layout import Slot
from app.render.manim import theme
from app.render.manim.scene_factory import (
    reveal_at_beat,
    slot_text,
)
from app.render.plan import PlannedScene, RenderPlan, ResolvedLogRatio

LEFT_X_PX = 330
RIGHT_X_PX = 950
GRID_CENTER_Y_PX = 300
DOT_SPACING_PX = 26
DOT_RADIUS_PX = 8
PH_LABEL_TOP_PX = 460
RATIO_TOP_PX = 250
ARROW_Y_PX = 330


def _grid_shape(count: int) -> tuple[int, int]:
    return {1: (1, 1), 10: (2, 5), 100: (10, 10)}[count]


def dot_grid(count: int, ph: int, center_x_px: float) -> VGroup:
    rows, cols = _grid_shape(count)
    grid = VGroup()
    for r in range(rows):
        for c in range(cols):
            x = center_x_px + (c - (cols - 1) / 2) * DOT_SPACING_PX
            y = GRID_CENTER_Y_PX + (r - (rows - 1) / 2) * DOT_SPACING_PX
            grid.add(
                Dot(theme.point3(x, y), radius=theme.px_to_units(DOT_RADIUS_PX), color=theme.ACCENT)
            )
    grid.h_ion_ph = ph  # type: ignore[attr-defined]
    return grid


def _panel(count: int, ph: int, center_x_px: float) -> tuple[VGroup, VGroup]:
    grid = dot_grid(count, ph, center_x_px)
    slot = Slot(
        f"log_ratio.ph.{ph}", round(center_x_px) - 150, PH_LABEL_TOP_PX, 300, 48, 32, 1, "bold"
    )
    return grid, slot_text(f"pH {ph}", slot)


def build(scene: Scene, planned: PlannedScene, plan: RenderPlan) -> None:
    v = planned.visual
    assert isinstance(v, ResolvedLogRatio)
    lower = min(v.from_ph, v.to_ph)
    from_count = v.ratio if v.from_ph == lower else 1
    to_count = v.ratio if v.to_ph == lower else 1

    for target, count, ph, x in (
        ("from", from_count, v.from_ph, LEFT_X_PX),
        ("to", to_count, v.to_ph, RIGHT_X_PX),
    ):
        grid, label = _panel(count, ph, x)
        reveal_at_beat(
            scene,
            planned,
            target,
            FadeIn(grid, lag_ratio=min(0.1, 1 / len(grid))),
            FadeIn(label),
            seconds=0.8,
        )

    ratio_slot = Slot("log_ratio.ratio", 490, RATIO_TOP_PX, 300, 48, 34, 1, "bold")
    ratio = slot_text(f"{v.ratio}\u00d7 more H\u207a", ratio_slot, color=theme.ACCENT)
    start, end = (720, 540) if from_count > to_count else (540, 720)  # points at the richer grid
    arrow = Arrow(
        theme.point3(start, ARROW_Y_PX),
        theme.point3(end, ARROW_Y_PX),
        buff=0,
        color=theme.ACCENT,
        stroke_width=5,
    )
    reveal_at_beat(scene, planned, "ratio", FadeIn(ratio), GrowArrow(arrow), seconds=0.6)
