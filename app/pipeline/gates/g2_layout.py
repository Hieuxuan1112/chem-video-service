"""G2d: every LLM-provided on-screen string must fit its layout slot (spec §5.5)."""

from app.pipeline.gates.types import GateResult, Violation
from app.pipeline.storyboard import ComparisonTable, KeyPoints, Scene, Storyboard, TitleCard
from app.render.layout import LAYOUT, FontSet, measure_wrap

GATE = "G2_layout"


def slotted_strings(scene: Scene) -> list[tuple[str, str, str]]:
    """(field label, slot name, text) for every LLM-provided string shown in the scene."""
    items = [("on_screen_title", "on_screen_title", scene.on_screen_title)]
    v = scene.visual
    if isinstance(v, TitleCard):
        items.append(("subtitle", "title_card.subtitle", v.subtitle))
    elif isinstance(v, KeyPoints):
        items.extend((f"points[{i}]", f"key_points.point.{i}", p) for i, p in enumerate(v.points))
    elif isinstance(v, ComparisonTable):
        items.append(("left_header", "comparison_table.header.left", v.left_header))
        items.append(("right_header", "comparison_table.header.right", v.right_header))
        for i, row in enumerate(v.rows):
            items.append((f"rows[{i}].aspect", f"comparison_table.aspect.{i}", row.aspect))
            items.append((f"rows[{i}].left", f"comparison_table.left.{i}", row.left))
            items.append((f"rows[{i}].right", f"comparison_table.right.{i}", row.right))
    return items


def g2_layout(sb: Storyboard, fonts: FontSet) -> GateResult:
    violations: list[Violation] = []
    checked = 0
    for i, scene in enumerate(sb.scenes):
        for field, slot_name, text in slotted_strings(scene):
            slot = LAYOUT[slot_name]
            res = measure_wrap(text, slot, fonts)
            checked += 1
            if not res.fits:
                violations.append(
                    Violation(
                        "G2d",
                        "text_overflow",
                        f"Scene {i}: '{field}' needs {len(res.lines)} lines / {res.width_px}px; "
                        f"slot allows {slot.max_lines} lines / {slot.w}px. Shorten it.",
                        i,
                    )
                )
    return GateResult(GATE, not violations, violations, {"strings_checked": checked})
