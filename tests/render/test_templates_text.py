from collections.abc import Callable

import pytest

from app.pipeline.storyboard import TableRow
from app.render.layout import LAYOUT, FontSet, Slot
from app.render.manim import scene_factory, theme
from app.render.manim.scene_factory import LayoutAssertionError, assert_inside_safe_area
from app.render.plan import (
    RenderPlan,
    ResolvedComparisonTable,
    ResolvedKeyPoints,
    ResolvedTitleCard,
)
from tests.render.harness import (
    TOLERANCE_UNITS,
    Built,
    assert_no_text_overlap,
    box,
    build_offscreen,
    load_fonts,
    plan_for,
    texts,
    widest_accepted,
)

pytestmark = pytest.mark.render

LONG_WORD_QUESTION = "W" * 300
LONG_WORDS_QUESTION = " ".join(["Wmwmwmwmw"] * 30)[:300]
INNER_LONG_WORD_QUESTION = f"What is {'W' * 120} in the pH scale?"


@pytest.fixture(scope="module")
def fonts() -> FontSet:
    return load_fonts()


def _title_card(fonts: FontSet, question: str) -> RenderPlan:
    visual = ResolvedTitleCard(
        type="title_card",
        question=question,
        subtitle=widest_accepted("title_card.subtitle", 60, fonts),
        title="How the pH scale works",
    )
    return plan_for(visual, question=question)


def _key_points(fonts: FontSet, question: str) -> RenderPlan:
    points = [widest_accepted(f"key_points.point.{i}", 60, fonts) for i in range(4)]
    return plan_for(
        ResolvedKeyPoints(type="key_points", points=points, question=question), question=question
    )


def _comparison_table(fonts: FontSet, question: str) -> RenderPlan:
    rows = [
        TableRow(
            aspect=widest_accepted(f"comparison_table.aspect.{i}", 20, fonts),
            left=widest_accepted(f"comparison_table.left.{i}", 28, fonts),
            right=widest_accepted(f"comparison_table.right.{i}", 28, fonts),
        )
        for i in range(4)
    ]
    visual = ResolvedComparisonTable(
        type="comparison_table",
        left_header=widest_accepted("comparison_table.header.left", 16, fonts),
        right_header=widest_accepted("comparison_table.header.right", 16, fonts),
        rows=rows,
    )
    return plan_for(visual, question=question)


BUILDERS = {
    "title_card": _title_card,
    "key_points": _key_points,
    "comparison_table": _comparison_table,
}
QUESTIONS = (LONG_WORD_QUESTION, LONG_WORDS_QUESTION, INNER_LONG_WORD_QUESTION)
CASES = [(name, q) for name in BUILDERS for q in QUESTIONS]


def _slot_box(slot: Slot) -> tuple[float, float, float, float]:
    left, top = theme.point(slot.x, slot.y)
    right, bottom = theme.point(slot.x + slot.w, slot.y + slot.h)
    return left, bottom, right, top


@pytest.fixture
def built(fonts: FontSet, monkeypatch: pytest.MonkeyPatch) -> Callable[[str, str], Built]:
    return lambda name, question: build_offscreen(BUILDERS[name](fonts, question), monkeypatch)


@pytest.mark.parametrize(("name", "question"), CASES)
def test_worst_case_strings_inside_safe_area(
    built: Callable[[str, str], Built], name: str, question: str
) -> None:
    result = built(name, question)

    assert_inside_safe_area(result.scene.mobjects)
    assert len(result.groups) >= 3
    for group in result.groups:
        left, bottom, right, top = _slot_box(LAYOUT[group.slot_name])
        g_left, g_bottom, g_right, g_top = box(group)
        assert g_left >= left - TOLERANCE_UNITS, group.slot_name
        assert g_right <= right + TOLERANCE_UNITS, group.slot_name
        assert g_bottom >= bottom - TOLERANCE_UNITS, group.slot_name
        assert g_top <= top + TOLERANCE_UNITS, group.slot_name


@pytest.mark.parametrize(("name", "question"), CASES)
def test_no_text_bbox_overlap(built: Callable[[str, str], Built], name: str, question: str) -> None:
    assert_no_text_overlap(built(name, question).scene)


@pytest.mark.parametrize(("name", "question"), CASES)
def test_scale_never_below_0_9(
    built: Callable[[str, str], Built], name: str, question: str
) -> None:
    scales = [g.text_scale for g in built(name, question).groups]
    assert scales and min(scales) >= scene_factory.MIN_TEXT_SCALE


def test_text_needing_more_shrink_is_a_layout_error(fonts: FontSet) -> None:
    with pytest.raises(LayoutAssertionError):
        scene_factory.text_lines(["W" * 60], LAYOUT["comparison_table.left.0"])


@pytest.mark.parametrize("name", ["title_card", "key_points"])
def test_question_markup_rendered_literally(built: Callable[[str, str], Built], name: str) -> None:
    drawn = [t.original_text for t in texts(built(name, "<b>pH</b> & <i>acids</i>?").scene)]
    assert any("<b>pH</b>" in s for s in drawn), drawn
