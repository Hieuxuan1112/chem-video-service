"""Builds one Manim Scene class per planned scene, with exact frame counts (spec §8.2).

Frame accounting (verified against manim 0.21): an animated play of run_time r writes
len(arange(0, r, 1/fps)) frames and a frozen wait writes int(r * fps). Running animations for
(n - 0.5) / fps and frozen holds for (n + 0.5) / fps therefore writes exactly n frames either way.
"""

import importlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from manim import (
    Animation,
    Dot,
    FadeIn,
    FadeOut,
    Mobject,
    MoveToTarget,
    Scene,
    Text,
    VGroup,
    VMobject,
    config,
)

from app.render.layout import LAYOUT, SAFE_AREA, FontSet, Slot, measure_wrap
from app.render.manim import theme
from app.render.plan import PlannedScene, RenderPlan

TemplateBuild = Callable[[Scene, PlannedScene, RenderPlan], None]
TEMPLATE_REGISTRY: dict[str, TemplateBuild] = {}
FADE_OUT_S = 0.3
MAX_REVEAL_S = 1.0  # longest reveal a template may play at a beat
MIN_TEXT_SCALE = 0.9
ELLIPSIS = "\u2026"
_SAFE_TOLERANCE = 1e-3
_FONTS: dict[str, FontSet] = {}


class LayoutAssertionError(Exception):
    pass


class TemplateOverrunError(Exception):
    """Animations used more frames than the scene has; templates may only extend holds."""


def set_fonts(fonts: FontSet) -> None:
    _FONTS["set"] = fonts


def fonts() -> FontSet:
    return _FONTS["set"]


def template_for(visual_type: str) -> TemplateBuild:
    """Imports app.render.manim.templates.<type> on first use and caches its build()."""
    if visual_type not in TEMPLATE_REGISTRY:
        module = importlib.import_module(f"app.render.manim.templates.{visual_type}")
        TEMPLATE_REGISTRY[visual_type] = module.build
    return TEMPLATE_REGISTRY[visual_type]


# --- frame-exact timing ---------------------------------------------------------------------------


def frames_written(scene: Scene) -> int:
    return round(scene.renderer.time * config.frame_rate)


def _moving(scene: Scene) -> bool:
    return any(m.get_family_updaters() for m in scene.mobjects)


def hold_frames(scene: Scene, frames: int) -> None:
    if frames <= 0:
        return
    fps = config.frame_rate
    if _moving(scene) and frames > 1:
        scene.wait((frames - 0.5) / fps, frozen_frame=False)
    else:
        scene.wait((frames + 0.5) / fps, frozen_frame=True)


def play_frames(scene: Scene, *animations: Animation, frames: int) -> None:
    if frames <= 1:
        scene.play(*animations, run_time=1 / config.frame_rate)  # arange(0, 1/fps) -> 1 frame
        return
    scene.play(*animations, run_time=(frames - 0.5) / config.frame_rate)


def change_to(
    mob: Mobject, change: Callable[[Mobject], object], *, path_arc: float = 0.0
) -> Animation:
    """A typed stand-in for mob.animate: apply `change` to a target copy and move to it."""
    mob.generate_target()
    assert mob.target is not None
    change(mob.target)
    return MoveToTarget(mob, path_arc=path_arc)


def move_to_point(mob: Mobject, point: np.ndarray, *, path_arc: float = 0.0) -> Animation:
    return change_to(mob, lambda m: m.move_to(point), path_arc=path_arc)


def seconds_to_frames(seconds: float) -> int:
    return max(1, round(seconds * config.frame_rate))


def beat_time(planned: PlannedScene, target: str) -> float:
    for beat in planned.beats:
        if beat.target == target:
            return beat.at_s
    return planned.lead_in_s


def reveal_at_beat(
    scene: Scene, planned: PlannedScene, target: str, *animations: Animation, seconds: float = 0.5
) -> None:
    """Holds until the beat, then plays at normal speed: animations are never sped up.

    When beats crowd the end of the scene, the hold is cut short so this reveal, every later beat
    target (reserved at MAX_REVEAL_S each) and the fade-out still fit (spec 8.2).
    """
    if seconds > MAX_REVEAL_S:
        raise ValueError(f"reveal of {seconds}s exceeds MAX_REVEAL_S")
    targets = [b.target for b in planned.beats]
    later = len(targets) - targets.index(target) - 1 if target in targets else 0
    frames = seconds_to_frames(seconds)
    latest = (
        planned.frames
        - seconds_to_frames(FADE_OUT_S)
        - frames
        - later * seconds_to_frames(MAX_REVEAL_S)
    )
    beat_frame = round(beat_time(planned, target) * config.frame_rate)
    hold_frames(scene, min(beat_frame, latest) - frames_written(scene))
    play_frames(scene, *animations, frames=frames)


# --- text in layout slots -------------------------------------------------------------------------


def text_lines(
    lines: Sequence[str], slot: Slot, *, color: str = theme.TEXT, align: str = "center"
) -> VGroup:
    """One Text per line, baselines spaced like the G2d measurement, scaled down only if wider
    or taller than the slot and never below 0.9x."""
    size_px = slot.font_px
    group = VGroup(
        *[
            Text(
                line,
                font=theme.FONT_FAMILY,
                font_size=theme.font_size_for(size_px),
                weight=theme.manim_weight(slot.weight),
                color=color,
            )
            for line in lines
        ]
    )
    step = theme.px_to_units(size_px * 1.2)
    x0, y0 = theme.point(slot.x, slot.y)
    width = theme.px_to_units(slot.w)
    height = theme.px_to_units(slot.h)
    top = y0 - (height - step * len(lines)) / 2
    for i, line in enumerate(group):
        y = top - step * (i + 0.5)
        if align == "left":
            line.move_to(np.array([x0 + line.width / 2, y, 0.0]))
        else:
            line.move_to(np.array([x0 + width / 2, y, 0.0]))
    scale = min(1.0, width / group.width if group.width else 1.0, height / group.height)
    if scale < MIN_TEXT_SCALE:
        raise LayoutAssertionError(
            f"text {' '.join(lines)!r} needs scale {scale:.2f} in slot {slot.name}"
        )
    if scale < 1.0:
        anchor = group.get_left() if align == "left" else group.get_center()
        group.scale(scale, about_point=anchor)
    group.text_scale = scale  # type: ignore[attr-defined]
    group.slot_name = slot.name  # type: ignore[attr-defined]
    return group


def text_in_slot(
    text: str, slot_name: str, *, color: str = theme.TEXT, align: str = "center"
) -> VGroup:
    return slot_text(text, LAYOUT[slot_name], color=color, align=align)


def slot_text(text: str, slot: Slot, *, color: str = theme.TEXT, align: str = "center") -> VGroup:
    """Wraps like G2d into a slot that need not be in LAYOUT (e.g. positioned by the template)."""
    return text_lines(measure_wrap(text, slot, fonts()).lines, slot, color=color, align=align)


def fit_user_text(text: str, slot_name: str, *, min_px: int = 28) -> VGroup:
    """User text is not G2d-checked: shrink the font to fit the slot's lines, then ellipsize."""
    slot = LAYOUT[slot_name]
    for size in range(slot.font_px, min_px - 1, -2):
        sized = Slot(slot.name, slot.x, slot.y, slot.w, slot.h, size, slot.max_lines, slot.weight)
        wrap = measure_wrap(text, sized, fonts())
        if wrap.fits:
            return text_lines(wrap.lines, sized)
    sized = Slot(slot.name, slot.x, slot.y, slot.w, slot.h, min_px, slot.max_lines, slot.weight)
    lines = measure_wrap(_break_long_words(text, sized), sized, fonts()).lines
    if len(lines) > slot.max_lines:
        lines = lines[: slot.max_lines]
        last = lines[-1]
        while last and not _fits_one_line(last + ELLIPSIS, sized):
            last = last[:-1].rstrip()
        lines[-1] = last + ELLIPSIS
    return text_lines(lines, sized)


def _fits_one_line(text: str, slot: Slot) -> bool:
    wrap = measure_wrap(text, slot, fonts())
    return wrap.fits and len(wrap.lines) == 1


def _break_long_words(text: str, slot: Slot) -> str:
    """Hard-splits words wider than the slot, so an unbroken user string still wraps."""
    pieces: list[str] = []
    for word in text.split():
        while len(word) > 1 and not _fits_one_line(word, slot):
            cut = len(word) - 1
            while cut > 1 and not _fits_one_line(word[:cut], slot):
                cut -= 1
            pieces.append(word[:cut])
            word = word[cut:]
        pieces.append(word)
    return " ".join(pieces)


# --- scene assembly -------------------------------------------------------------------------------


def tick_mark(x_px: float, y_px: float, size_px: float = 22) -> VMobject:
    size = theme.px_to_units(size_px)
    x, y = theme.point(x_px, y_px)
    mark = VMobject(stroke_color=theme.ACCENT, stroke_width=6)
    mark.set_points_as_corners(
        np.array(
            [
                [x - size / 2, y, 0.0],
                [x - size / 8, y - size / 2.5, 0.0],
                [x + size / 2, y + size / 2, 0.0],
            ]
        )
    )
    return mark


def _safe_box() -> tuple[float, float, float, float]:
    x0, y0 = theme.point(SAFE_AREA[0], SAFE_AREA[1])
    x1, y1 = theme.point(SAFE_AREA[2], SAFE_AREA[3])
    return x0, y1, x1, y0  # left, bottom, right, top


def assert_inside_safe_area(mobjects: Sequence[Mobject]) -> None:
    left, bottom, right, top = _safe_box()
    for mob in mobjects:
        if not mob.get_family() or (mob.width == 0 and mob.height == 0):
            continue
        box = (mob.get_left()[0], mob.get_bottom()[1], mob.get_right()[0], mob.get_top()[1])
        if (
            box[0] < left - _SAFE_TOLERANCE
            or box[1] < bottom - _SAFE_TOLERANCE
            or box[2] > right + _SAFE_TOLERANCE
            or box[3] > top + _SAFE_TOLERANCE
        ):
            raise LayoutAssertionError(
                f"{type(mob).__name__} {getattr(mob, 'slot_name', '')} outside the safe area: "
                f"{tuple(round(v, 3) for v in box)}"
            )


def progress_dots(count: int, current: int) -> VGroup:
    radius = theme.px_to_units(4)
    spacing = theme.px_to_units(16)
    dots = VGroup(
        *[
            Dot(radius=radius, color=theme.ACCENT if i == current else theme.MUTED)
            for i in range(count)
        ]
    )
    dots.arrange(buff=spacing - 2 * radius)
    dots.move_to(theme.point3(theme.DESIGN_WIDTH_PX / 2, 569))
    return dots


def make_scene_class(planned: PlannedScene, plan: RenderPlan) -> type[Scene]:
    build = template_for(planned.visual.type)

    def construct(self: Scene) -> None:
        title = text_in_slot(planned.on_screen_title, "on_screen_title")
        self.add(progress_dots(len(plan.scenes), planned.index))
        play_frames(self, FadeIn(title), frames=seconds_to_frames(planned.lead_in_s))
        build(self, planned, plan)
        assert_inside_safe_area(self.mobjects)
        fade = seconds_to_frames(FADE_OUT_S)
        remaining = planned.frames - fade - frames_written(self)
        if remaining < 0:
            raise TemplateOverrunError(
                f"scene {planned.index} animations need {-remaining} more frames than planned"
            )
        hold_frames(self, remaining)
        play_frames(self, *[FadeOut(m) for m in self.mobjects], frames=fade)

    attrs: dict[str, Any] = {"construct": construct}
    return type(f"Scene{planned.index:02d}", (Scene,), attrs)
