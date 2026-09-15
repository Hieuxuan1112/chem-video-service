"""title_card: the learner's question types in, the subtitle fades in, glyphs drift gently."""

import math
from typing import cast

import numpy as np
from manim import AddTextLetterByLetter, Circle, FadeIn, Mobject, Scene, Text, VGroup

from app.render.manim import theme
from app.render.manim.scene_factory import fit_user_text, reveal_at_beat, text_in_slot
from app.render.plan import PlannedScene, RenderPlan, ResolvedTitleCard

_GLYPHS = ((260, 540), (1040, 520), (1120, 180), (180, 150))  # design px, clear of the text slots
_DRIFT_UNITS = 0.12
QUESTION_TYPE_S = 0.8


def _drifting_glyph(x_px: float, y_px: float, phase: float) -> VGroup:
    radius = theme.px_to_units(14)
    atom = Circle(radius=radius, color=theme.ACCENT, stroke_width=2, stroke_opacity=0.35)
    partner = Circle(radius=radius * 0.7, color=theme.MUTED, stroke_width=2, stroke_opacity=0.35)
    partner.next_to(atom, direction=np.array([1.0, 0.4, 0.0]), buff=0)
    glyph = VGroup(atom, partner)
    home = theme.point3(x_px, y_px)
    glyph.move_to(home)
    clock = {"t": phase}

    def drift(mob: Mobject, dt: float) -> None:
        clock["t"] += dt
        t = clock["t"]
        mob.move_to(home + _DRIFT_UNITS * np.array([math.sin(0.6 * t), math.cos(0.4 * t), 0.0]))

    glyph.add_updater(drift)
    return glyph


def build(scene: Scene, planned: PlannedScene, plan: RenderPlan) -> None:
    v = planned.visual
    assert isinstance(v, ResolvedTitleCard)
    for i, (x, y) in enumerate(_GLYPHS):
        scene.add(_drifting_glyph(x, y, phase=i * 1.7))
    question = fit_user_text(v.question, "title_card.question")  # raw user text, drawn with Text
    subtitle = text_in_slot(v.subtitle, "title_card.subtitle", color=theme.MUTED)
    reveal_at_beat(
        scene,
        planned,
        "question",
        *[AddTextLetterByLetter(cast(Text, line)) for line in question],
        seconds=QUESTION_TYPE_S,
    )
    reveal_at_beat(scene, planned, "subtitle", FadeIn(subtitle), seconds=0.5)
