"""electron_sharing: two atoms approach until their outer shells overlap, `shared_pairs` electrons
from each atom move into the overlap, then each atom's outer-shell count gets a tick."""

import numpy as np
from manim import FadeIn, Scene, VGroup

from app.render.layout import Slot
from app.render.manim import theme
from app.render.manim.scene_factory import (
    change_to,
    move_to_point,
    play_frames,
    reveal_at_beat,
    seconds_to_frames,
    slot_text,
    tick_mark,
)
from app.render.manim.templates._atoms import bohr_atom, place_atom, shell_radius_px
from app.render.plan import PlannedScene, RenderPlan, ResolvedElectronSharing

CENTER_Y_PX = 330
OVERLAP_PX = 48  # width of the lens where the outer shells overlap
START_GAP_PX = 160  # extra distance before the atoms approach
PAIR_SPACING_PX = 22
PARTNER_OFFSET_PX = 9
COUNTER_TOP_PX = 516
TITLE_SLOT = Slot("electron_sharing.molecule", 340, 116, 600, 40, 28, 1, "semibold")


def build(scene: Scene, planned: PlannedScene, plan: RenderPlan) -> None:
    v = planned.visual
    assert isinstance(v, ResolvedElectronSharing)
    radii = [shell_radius_px(len(a.shells)) for a in v.atoms]
    distance = radii[0] + radii[1] - OVERLAP_PX
    left = 640 - (radii[0] + distance + radii[1]) / 2
    centers = [left + radii[0], left + radii[0] + distance]

    atoms = [
        place_atom(
            bohr_atom(atom, radius_px=r),
            x + (START_GAP_PX / 2) * (-1 if i == 0 else 1),
            CENTER_Y_PX,
        )
        for i, (atom, r, x) in enumerate(zip(v.atoms, radii, centers, strict=True))
    ]
    title = slot_text(f"{v.display} ({v.name})", TITLE_SLOT, color=theme.MUTED)
    play_frames(scene, *[FadeIn(a) for a in atoms], FadeIn(title), frames=seconds_to_frames(0.6))
    shift = theme.px_to_units(START_GAP_PX / 2)
    reveal_at_beat(
        scene,
        planned,
        "approach",
        change_to(atoms[0], lambda m: m.shift(np.array([shift, 0.0, 0.0]))),
        change_to(atoms[1], lambda m: m.shift(np.array([-shift, 0.0, 0.0]))),
        seconds=0.8,
    )

    lens_x = (centers[0] + radii[0] + centers[1] - radii[1]) / 2
    moves = []
    for side, bohr in enumerate(atoms):
        facing = sorted(bohr.valence_dots, key=lambda d: d.get_center()[0], reverse=side == 0)
        for k, dot in enumerate(facing[: v.shared_pairs]):
            y = CENTER_Y_PX + (k - (v.shared_pairs - 1) / 2) * PAIR_SPACING_PX
            x = lens_x + (PARTNER_OFFSET_PX if side else -PARTNER_OFFSET_PX)
            dot.shared = True  # type: ignore[attr-defined]
            moves.append(move_to_point(dot, theme.point3(x, y)))
    reveal_at_beat(scene, planned, "share", *moves, seconds=0.8)

    counters = []
    for i, (atom, x) in enumerate(zip(v.atoms, centers, strict=True)):
        cx = min(x, 510.0) if i == 0 else max(x, 770.0)
        slot = Slot(
            f"electron_sharing.count.{i}",
            round(cx) - 115,
            COUNTER_TOP_PX,
            230,
            36,
            22,
            1,
            "semibold",
        )
        text = slot_text(f"{atom.valence + v.shared_pairs} outer electrons", slot)
        text_left_px = 640 + text.get_left()[0] / theme.UNITS_PER_PX
        tick = tick_mark(text_left_px - 20, COUNTER_TOP_PX + 18, 18)
        counters.append(VGroup(tick, text))
    reveal_at_beat(scene, planned, "full_shell", *[FadeIn(c) for c in counters], seconds=0.5)
