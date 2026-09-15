"""Theme dark-v1 (spec §8.2): colours, the discrete pH palette, and 1280x720 px -> Manim units."""

from pathlib import Path

import manimpango
import numpy as np
from manim import BOLD, NORMAL, SEMIBOLD

from app.render.layout import FontSet, Slot, Weight

BACKGROUND = "#0F172A"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
ACCENT = "#38BDF8"
ELECTRON = "#FACC15"

# Universal-indicator colours, one per whole pH unit (index = pH 0..14).
PH_COLORS: tuple[str, ...] = (
    "#E11D48", "#EF4444", "#F97316", "#FB923C", "#F59E0B", "#FACC15", "#A3E635", "#22C55E",
    "#10B981", "#14B8A6", "#0EA5E9", "#3B82F6", "#6366F1", "#7C3AED", "#9333EA",
)  # fmt: skip

FONT_FAMILY = "Inter"
DESIGN_WIDTH_PX = 1280
DESIGN_HEIGHT_PX = 720
FRAME_HEIGHT_UNITS = 8.0
UNITS_PER_PX = FRAME_HEIGHT_UNITS / DESIGN_HEIGHT_PX
# Measured with manim 0.21: an Inter em is font_size / 72 units, so 1 px of em = 0.8 font_size.
FONT_SIZE_PER_PX = 72 * UNITS_PER_PX

MANIM_WEIGHTS = {"regular": NORMAL, "semibold": SEMIBOLD, "bold": BOLD}


def px_to_units(px: float) -> float:
    return px * UNITS_PER_PX


def font_size_for(px: float) -> float:
    return px * FONT_SIZE_PER_PX


def point(x_px: float, y_px: float) -> tuple[float, float]:
    """Design-space pixel (origin top-left) -> Manim coordinates (origin centre, y up)."""
    return (
        (x_px - DESIGN_WIDTH_PX / 2) * UNITS_PER_PX,
        (DESIGN_HEIGHT_PX / 2 - y_px) * UNITS_PER_PX,
    )


def point3(x_px: float, y_px: float) -> np.ndarray:
    """point() as the 3-vector Manim positions expect."""
    return np.array([*point(x_px, y_px), 0.0])


def slot_center(slot: Slot) -> tuple[float, float]:
    return point(slot.x + slot.w / 2, slot.y + slot.h / 2)


def manim_weight(weight: Weight) -> str:
    return MANIM_WEIGHTS[weight]


def register_fonts(fonts_dir: Path) -> FontSet:
    fonts = FontSet.from_dir(fonts_dir)
    for path in (fonts.regular, fonts.semibold, fonts.bold):
        if not manimpango.register_font(str(path)):
            raise RuntimeError(f"could not register font {path}")
    return fonts


def relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(a: str, b: str) -> float:
    la, lb = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)
