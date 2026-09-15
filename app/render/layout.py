"""Layout slots shared by the G2d text-fit gate and the Manim templates (spec §5.5, §8.2).

Pixel coordinates on a 1280x720 canvas. Text is measured with the bundled Inter font, so a string
accepted here fits in the rendered frame.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from PIL import ImageFont

Weight = Literal["regular", "semibold", "bold"]

LINE_HEIGHT = 1.2  # multiple of font_px


@dataclass(frozen=True)
class Slot:
    name: str
    x: int
    y: int
    w: int
    h: int
    font_px: int
    max_lines: int
    weight: Weight


SAFE_AREA = (64, 36, 1216, 684)  # 90 % title-safe box (x0, y0, x1, y1)
ZONES = {"title": (0, 108), "visual": (108, 562), "captions": (576, 684)}


def _build_layout() -> dict[str, Slot]:
    slots = [
        Slot("on_screen_title", 64, 36, 1152, 68, 44, 1, "bold"),
        Slot("title_card.question", 140, 200, 1000, 150, 40, 3, "semibold"),
        Slot("title_card.subtitle", 140, 380, 1000, 100, 32, 2, "regular"),
        Slot("key_points.recap", 190, 534, 930, 28, 20, 1, "regular"),
        Slot("comparison_table.header.left", 330, 124, 420, 52, 30, 1, "bold"),
        Slot("comparison_table.header.right", 760, 124, 420, 52, 30, 1, "bold"),
        Slot("ph_scale.marker", 0, 0, 170, 56, 20, 2, "semibold"),
        Slot("caption", 90, 588, 1100, 88, 30, 2, "semibold"),
    ]
    for i in range(4):
        slots.append(Slot(f"key_points.point.{i}", 190, 130 + i * 104, 930, 88, 30, 2, "regular"))
        row_y = 190 + i * 92
        slots.append(Slot(f"comparison_table.aspect.{i}", 90, row_y, 230, 80, 24, 2, "semibold"))
        slots.append(Slot(f"comparison_table.left.{i}", 330, row_y, 420, 80, 24, 2, "regular"))
        slots.append(Slot(f"comparison_table.right.{i}", 760, row_y, 420, 80, 24, 2, "regular"))
    return {s.name: s for s in slots}


LAYOUT: dict[str, Slot] = _build_layout()


@dataclass(frozen=True)
class FontSet:
    regular: Path
    semibold: Path
    bold: Path

    @classmethod
    def from_dir(cls, d: Path) -> "FontSet":
        fonts = cls(
            regular=d / "Inter-Regular.ttf",
            semibold=d / "Inter-SemiBold.ttf",
            bold=d / "Inter-Bold.ttf",
        )
        for path in (fonts.regular, fonts.semibold, fonts.bold):
            if not path.is_file():
                raise FileNotFoundError(path)
        return fonts

    def path_for(self, weight: Weight) -> Path:
        return {"regular": self.regular, "semibold": self.semibold, "bold": self.bold}[weight]


@dataclass(frozen=True)
class WrapResult:
    lines: list[str]
    width_px: int
    height_px: int
    fits: bool
    reason: str | None  # word_too_wide | too_many_lines | too_tall


@lru_cache(maxsize=64)
def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def measure_wrap(text: str, slot: Slot, fonts: FontSet, *, safety: float = 0.05) -> WrapResult:
    """Greedy word wrap into the slot's usable width (width minus the safety margin)."""
    font = _font(fonts.path_for(slot.weight), slot.font_px)
    usable = slot.w * (1 - safety)
    lines: list[str] = []
    current = ""
    word_too_wide = False
    for word in text.split():
        if font.getlength(word) > usable:
            word_too_wide = True
        candidate = f"{current} {word}" if current else word
        if current and font.getlength(candidate) > usable:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    width = max((font.getlength(line) for line in lines), default=0.0)
    height = len(lines) * LINE_HEIGHT * slot.font_px
    reason = None
    if word_too_wide:
        reason = "word_too_wide"
    elif len(lines) > slot.max_lines:
        reason = "too_many_lines"
    elif height > slot.h:
        reason = "too_tall"
    return WrapResult(
        lines=lines,
        width_px=round(width),
        height_px=round(height),
        fits=reason is None,
        reason=reason,
    )
