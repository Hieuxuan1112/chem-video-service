"""ph_scale: 15 discrete segments stagger in with numbers and zone labels; markers drop at their
beats at the resolved pH; the highlighted zone pulses.

Marker labels share one row. They keep pH order and are pushed apart only as far as needed, so
five labels never overlap; a leader line joins each label to its arrow at the exact pH.
"""

from manim import (
    DOWN,
    Arrow,
    Create,
    FadeIn,
    Indicate,
    LaggedStart,
    Line,
    Rectangle,
    Scene,
    SurroundingRectangle,
    VGroup,
)

from app.render.layout import LAYOUT, Slot
from app.render.manim import theme
from app.render.manim.scene_factory import (
    play_frames,
    reveal_at_beat,
    seconds_to_frames,
    slot_text,
)
from app.render.plan import PlannedScene, RenderPlan, ResolvedPhScale

SCALE_LEFT_PX = 100
SEGMENT_PX = 72  # 15 segments span 100..1180
SCALE_TOP_PX = 360
SCALE_HEIGHT_PX = 60
NUMBER_TOP_PX = 428
ZONE_TOP_PX = 474
LABEL_TOP_PX = 150
LABEL_GAP_PX = 10
LEADER_TOP_PX = 214
ARROW_TOP_PX = 300
ZONES = {"acidic": (0, 6, "Acidic"), "neutral": (7, 7, "Neutral"), "basic": (8, 14, "Basic")}


def ph_to_x_px(ph: float) -> float:
    """Segment i is centred on pH i; fractional values interpolate linearly."""
    return SCALE_LEFT_PX + (ph + 0.5) * SEGMENT_PX


def label_centers_px(targets: list[float], width: float, gap: float) -> list[float]:
    """1-D placement: as close to each target as possible, in order, without overlap."""
    lo = SCALE_LEFT_PX + width / 2
    hi = SCALE_LEFT_PX + 15 * SEGMENT_PX - width / 2
    xs = [min(max(x, lo), hi) for x in targets]
    for i in range(1, len(xs)):
        xs[i] = max(xs[i], xs[i - 1] + width + gap)
    xs[-1] = min(xs[-1], hi)
    for i in range(len(xs) - 2, -1, -1):
        xs[i] = min(xs[i], xs[i + 1] - width - gap)
    return xs


def _segments() -> VGroup:
    segments = VGroup()
    for i, color in enumerate(theme.PH_COLORS):
        segment = Rectangle(
            width=theme.px_to_units(SEGMENT_PX),
            height=theme.px_to_units(SCALE_HEIGHT_PX),
            fill_color=color,
            fill_opacity=1.0,
            stroke_width=0,
        )
        segment.move_to(theme.point3(ph_to_x_px(i), SCALE_TOP_PX + SCALE_HEIGHT_PX / 2))
        segment.ph_index = i  # type: ignore[attr-defined]
        segments.add(segment)
    return segments


def _numbers() -> VGroup:
    return VGroup(
        *[
            slot_text(
                str(i),
                Slot(
                    f"ph_scale.number.{i}",
                    round(ph_to_x_px(i)) - 30,
                    NUMBER_TOP_PX,
                    60,
                    36,
                    24,
                    1,
                    "semibold",
                ),
                color=theme.MUTED,
            )
            for i in range(15)
        ]
    )


def _zone_labels() -> dict[str, VGroup]:
    labels = {}
    for zone, (first, last, text) in ZONES.items():
        center = (ph_to_x_px(first) + ph_to_x_px(last)) / 2
        slot = Slot(
            f"ph_scale.zone.{zone}", round(center) - 60, ZONE_TOP_PX, 120, 40, 28, 1, "bold"
        )
        labels[zone] = slot_text(text, slot)
    return labels


def build(scene: Scene, planned: PlannedScene, plan: RenderPlan) -> None:
    v = planned.visual
    assert isinstance(v, ResolvedPhScale)
    segments = _segments()
    numbers = _numbers()
    zones = _zone_labels()
    play_frames(
        scene,
        LaggedStart(*[FadeIn(s) for s in segments], lag_ratio=0.08),
        LaggedStart(*[FadeIn(n) for n in numbers], lag_ratio=0.08),
        frames=seconds_to_frames(1.0),
    )
    play_frames(scene, *[FadeIn(z) for z in zones.values()], frames=seconds_to_frames(0.4))

    marker_slot = LAYOUT["ph_scale.marker"]
    ordered = sorted(v.markers, key=lambda m: (m.ph, m.name))
    centers = label_centers_px([ph_to_x_px(m.ph) for m in ordered], marker_slot.w, LABEL_GAP_PX)
    placed = dict(zip([m.name for m in ordered], centers, strict=True))
    for marker in v.markers:
        label_x = placed[marker.name]
        slot = Slot(
            marker_slot.name,
            round(label_x - marker_slot.w / 2),
            LABEL_TOP_PX,
            marker_slot.w,
            marker_slot.h,
            marker_slot.font_px,
            marker_slot.max_lines,
            marker_slot.weight,
        )
        label = slot_text(marker.display, slot)
        tip_x = ph_to_x_px(marker.ph)
        leader = Line(
            theme.point3(label_x, LEADER_TOP_PX),
            theme.point3(tip_x, ARROW_TOP_PX),
            color=theme.MUTED,
            stroke_width=2,
        )
        arrow = Arrow(
            theme.point3(tip_x, ARROW_TOP_PX),
            theme.point3(tip_x, SCALE_TOP_PX - 4),
            buff=0,
            color=theme.TEXT,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.35,
        )
        arrow.marker_name = marker.name  # type: ignore[attr-defined]
        reveal_at_beat(
            scene,
            planned,
            f"marker:{marker.name}",
            FadeIn(VGroup(label, leader, arrow), shift=DOWN * theme.px_to_units(20)),
            seconds=0.5,
        )

    if v.highlight in ZONES:
        first, last, _ = ZONES[v.highlight]
        zone_segments = VGroup(*segments[first : last + 1])
        frame = SurroundingRectangle(
            zone_segments, color=theme.ACCENT, buff=theme.px_to_units(6), stroke_width=4
        )
        reveal_at_beat(
            scene,
            planned,
            "highlight",
            Create(frame),
            Indicate(zones[v.highlight], color=theme.ACCENT, scale_factor=1.1),
            seconds=0.6,
        )
