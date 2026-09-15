"""Bohr-model atoms drawn from the element table: every electron on screen is a table entry."""

import math

import numpy as np
from manim import PI, TAU, Circle, Dot, Text, VGroup

from app.render.manim import theme
from app.render.plan import ResolvedAtom

NUCLEUS_PX = 32
SHELL_STEP_PX = 42
ELECTRON_RADIUS_PX = 7
SYMBOL_PX = 26
_SUPERSCRIPTS = str.maketrans(
    "0123456789+-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207a\u207b"
)


class BohrAtom(VGroup):
    """A VGroup with handles on the parts templates animate."""

    nucleus: Circle
    shell_circles: list[Circle]
    valence_dots: list[Dot]
    gap_offsets: list[np.ndarray]
    atom_symbol: str


def shell_radius_px(shell_count: int) -> float:
    return NUCLEUS_PX + shell_count * SHELL_STEP_PX


def ion_label(ion: str) -> str:
    """'Mg2+' -> 'Mg' followed by superscript 2 and plus."""
    symbol = ion.rstrip("0123456789+-")
    return symbol + ion[len(symbol) :].translate(_SUPERSCRIPTS)


def bohr_atom(
    atom: ResolvedAtom, *, radius_px: float, valence_slots: int | None = None
) -> BohrAtom:
    """Nucleus + symbol, one circle per shell, one Dot per electron (valence in ELECTRON colour).

    Electrons sit evenly on their shell; outer ones start from the left (angle pi), inner ones half
    a step later so they stay off the horizontal bond axis. With valence_slots, the
    outer shell is laid out for that many positions and the first free ones (facing left) are kept
    as gap_offsets, where transferred electrons land.
    """
    step = (radius_px - NUCLEUS_PX) / len(atom.shells)
    nucleus = Circle(
        radius=theme.px_to_units(NUCLEUS_PX),
        stroke_color=atom.cpk,
        stroke_width=4,
        fill_color=atom.cpk,
        fill_opacity=0.25,
    )
    symbol = Text(
        atom.symbol,
        font=theme.FONT_FAMILY,
        font_size=theme.font_size_for(SYMBOL_PX),
        weight=theme.manim_weight("bold"),
        color=theme.TEXT,
    )
    group = BohrAtom(nucleus, symbol)
    shell_circles: list[Circle] = []
    valence_dots: list[Dot] = []
    gap_offsets: list[np.ndarray] = []
    last = len(atom.shells) - 1
    for k, count in enumerate(atom.shells):
        radius = theme.px_to_units(NUCLEUS_PX + (k + 1) * step)
        circle = Circle(radius=radius, stroke_color=theme.MUTED, stroke_width=2, stroke_opacity=0.6)
        shell_circles.append(circle)
        group.add(circle)
        slots = valence_slots if k == last and valence_slots is not None else count
        for i in range(slots):
            start = PI if k == last else PI + PI / slots  # inner electrons stay off the bond axis
            angle = start + TAU * i / slots
            offset = np.array([radius * math.cos(angle), radius * math.sin(angle), 0.0])
            if i < slots - count:
                gap_offsets.append(offset)
                continue
            dot = Dot(
                offset,
                radius=theme.px_to_units(ELECTRON_RADIUS_PX),
                color=theme.ELECTRON if k == last else theme.MUTED,
            )
            dot.shell_index = k  # type: ignore[attr-defined]
            if k == last:
                valence_dots.append(dot)
            group.add(dot)
    group.nucleus = nucleus
    group.shell_circles = shell_circles
    group.valence_dots = valence_dots
    group.gap_offsets = gap_offsets
    group.atom_symbol = atom.symbol
    return group


def place_atom(group: BohrAtom, x_px: float, y_px: float) -> BohrAtom:
    """Moves the atom so its nucleus (not its bounding box) sits on the point."""
    group.shift(theme.point3(x_px, y_px) - group.nucleus.get_center())
    return group
