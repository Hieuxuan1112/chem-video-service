"""atom_shells: one or two Bohr atoms appear at their beats; valence electrons orbit slowly."""

from manim import FadeIn, Mobject, Scene

from app.render.layout import Slot
from app.render.manim.scene_factory import reveal_at_beat, slot_text
from app.render.manim.templates._atoms import bohr_atom, place_atom, shell_radius_px
from app.render.plan import PlannedScene, RenderPlan, ResolvedAtomShells

CENTER_Y_PX = 310
LABEL_TOP_PX = 492
ORBIT_RAD_PER_S = 0.35


def _centers_px(count: int) -> list[float]:
    return [640.0] if count == 1 else [400.0, 880.0]


def build(scene: Scene, planned: PlannedScene, plan: RenderPlan) -> None:
    v = planned.visual
    assert isinstance(v, ResolvedAtomShells)
    for atom, x in zip(v.atoms, _centers_px(len(v.atoms)), strict=True):
        group = place_atom(
            bohr_atom(atom, radius_px=shell_radius_px(len(atom.shells))), x, CENTER_Y_PX
        )
        shells = ", ".join(str(n) for n in atom.shells)
        slot = Slot(
            f"atom_shells.label.{atom.symbol}",
            round(x) - 200,
            LABEL_TOP_PX,
            400,
            44,
            28,
            1,
            "semibold",
        )
        label = slot_text(f"{atom.name.capitalize()}: {shells}", slot)
        reveal_at_beat(
            scene, planned, f"atom:{atom.symbol}", FadeIn(group), FadeIn(label), seconds=0.6
        )

        nucleus = group.nucleus

        def orbit(mob: Mobject, dt: float, nucleus: Mobject = nucleus) -> None:
            mob.rotate(ORBIT_RAD_PER_S * dt, about_point=nucleus.get_center())

        for dot in group.valence_dots:
            dot.add_updater(orbit)
