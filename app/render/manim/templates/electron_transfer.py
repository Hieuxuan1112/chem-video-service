"""electron_transfer: `electrons` valence dots arc from the donor into the acceptor's free outer
slots, the ion charges appear, then arrows show the attraction between the ions."""

from manim import PI, Animation, Arrow, FadeIn, GrowArrow, Scene

from app.render.layout import Slot
from app.render.manim import theme
from app.render.manim.scene_factory import (
    change_to,
    move_to_point,
    play_frames,
    reveal_at_beat,
    seconds_to_frames,
    slot_text,
)
from app.render.manim.templates._atoms import bohr_atom, ion_label, place_atom, shell_radius_px
from app.render.plan import PlannedScene, RenderPlan, ResolvedElectronTransfer

DONOR_X_PX = 360
ACCEPTOR_X_PX = 920
CENTER_Y_PX = 320
ION_TOP_PX = 494
ARC_ANGLE = -PI / 3
TITLE_SLOT = Slot("electron_transfer.compound", 340, 112, 600, 40, 28, 1, "semibold")


def build(scene: Scene, planned: PlannedScene, plan: RenderPlan) -> None:
    v = planned.visual
    assert isinstance(v, ResolvedElectronTransfer)
    donor = place_atom(
        bohr_atom(v.donor, radius_px=shell_radius_px(len(v.donor.shells))), DONOR_X_PX, CENTER_Y_PX
    )
    acceptor = place_atom(
        bohr_atom(
            v.acceptor,
            radius_px=shell_radius_px(len(v.acceptor.shells)),
            valence_slots=v.acceptor.valence + v.electrons,
        ),
        ACCEPTOR_X_PX,
        CENTER_Y_PX,
    )
    for dot in donor.valence_dots:
        dot.rotate(PI, about_point=donor.nucleus.get_center())
    title = slot_text(f"{v.display} ({v.name})", TITLE_SLOT, color=theme.MUTED)
    play_frames(
        scene, FadeIn(donor), FadeIn(acceptor), FadeIn(title), frames=seconds_to_frames(0.6)
    )

    facing = sorted(donor.valence_dots, key=lambda d: d.get_center()[0], reverse=True)
    gaps = acceptor.gap_offsets
    moves = []
    for dot, offset in zip(facing[: v.electrons], gaps, strict=True):
        dot.transferred = True  # type: ignore[attr-defined]
        target = acceptor.nucleus.get_center() + offset
        moves.append(move_to_point(dot, target, path_arc=ARC_ANGLE))
    extra: list[Animation] = []
    if v.electrons == v.donor.valence:  # the donor's outer shell is now empty
        extra.append(change_to(donor.shell_circles[-1], lambda m: m.set_stroke(opacity=0.0)))
    reveal_at_beat(scene, planned, "transfer", *moves, *extra, seconds=1.0)

    ions = [
        slot_text(
            ion_label(ion),
            Slot(f"electron_transfer.ion.{i}", x - 120, ION_TOP_PX, 240, 56, 44, 1, "bold"),
        )
        for i, (ion, x) in enumerate(((v.donor_ion, DONOR_X_PX), (v.acceptor_ion, ACCEPTOR_X_PX)))
    ]
    reveal_at_beat(scene, planned, "ions", *[FadeIn(i) for i in ions], seconds=0.5)

    gap_left = DONOR_X_PX + shell_radius_px(len(v.donor.shells)) + 24
    gap_right = ACCEPTOR_X_PX - shell_radius_px(len(v.acceptor.shells)) - 24
    middle = (gap_left + gap_right) / 2
    arrows = [
        Arrow(
            theme.point3(gap_left, CENTER_Y_PX),
            theme.point3(middle - 8, CENTER_Y_PX),
            buff=0,
            color=theme.ACCENT,
            stroke_width=5,
        ),
        Arrow(
            theme.point3(gap_right, CENTER_Y_PX),
            theme.point3(middle + 8, CENTER_Y_PX),
            buff=0,
            color=theme.ACCENT,
            stroke_width=5,
        ),
    ]
    reveal_at_beat(scene, planned, "attract", *[GrowArrow(a) for a in arrows], seconds=0.5)
