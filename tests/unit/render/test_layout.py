import itertools

import pytest

from app.render.layout import LAYOUT, SAFE_AREA, ZONES, FontSet, Slot, measure_wrap

TEMPLATE_PREFIXES = ("title_card.", "key_points.", "comparison_table.")


def _rect(slot: Slot) -> tuple[int, int, int, int]:
    return slot.x, slot.y, slot.x + slot.w, slot.y + slot.h


def _zone_of(name: str) -> str:
    if name == "on_screen_title":
        return "title"
    if name == "caption":
        return "captions"
    return "visual"


def test_expected_slots_exist() -> None:
    for i in range(4):
        assert f"key_points.point.{i}" in LAYOUT
        for col in ("aspect", "left", "right"):
            assert f"comparison_table.{col}.{i}" in LAYOUT
    for name in (
        "on_screen_title",
        "title_card.question",
        "title_card.subtitle",
        "comparison_table.header.left",
        "comparison_table.header.right",
        "ph_scale.marker",
        "caption",
    ):
        assert LAYOUT[name].name == name


def test_slots_inside_safe_area_and_zones() -> None:
    x0, y0, x1, y1 = SAFE_AREA
    assert (x1 - x0, y1 - y0) == (1152, 648)  # 90 % of 1280x720
    for name, slot in LAYOUT.items():
        if name == "ph_scale.marker":
            continue  # positioned by the template; only its size is checked
        sx0, sy0, sx1, sy1 = _rect(slot)
        assert x0 <= sx0 and sx1 <= x1 and y0 <= sy0 and sy1 <= y1, name
        zy0, zy1 = ZONES[_zone_of(name)]
        assert zy0 <= sy0 and sy1 <= zy1, name
        # the text block always fits vertically at 1.2 line height
        assert slot.max_lines * 1.2 * slot.font_px <= slot.h, name


def test_slots_do_not_overlap_within_template() -> None:
    for prefix in TEMPLATE_PREFIXES:
        slots = [s for n, s in LAYOUT.items() if n.startswith(prefix)]
        assert len(slots) >= 2
        for a, b in itertools.combinations(slots, 2):
            ax0, ay0, ax1, ay1 = _rect(a)
            bx0, by0, bx1, by1 = _rect(b)
            disjoint = ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0
            assert disjoint, (a.name, b.name)


def test_measure_short_title_fits(fonts: FontSet) -> None:
    res = measure_wrap("How the pH scale works", LAYOUT["on_screen_title"], fonts)
    assert res.fits, res
    assert res.lines == ["How the pH scale works"]
    assert res.reason is None
    assert 0 < res.width_px <= LAYOUT["on_screen_title"].w * 0.95


def test_measure_overlong_title_fails_width_or_lines(fonts: FontSet) -> None:
    text = "Why acids and bases change colour on the universal indicator scale"
    res = measure_wrap(text, LAYOUT["on_screen_title"], fonts)
    assert not res.fits
    assert len(res.lines) > 1
    assert res.reason == "too_many_lines"


def test_single_huge_word_fails(fonts: FontSet) -> None:
    res = measure_wrap("W" * 80, LAYOUT["comparison_table.left.0"], fonts)
    assert not res.fits
    assert res.reason == "word_too_wide"


def test_caption_42_chars_two_lines_fit(fonts: FontSet) -> None:
    line = "Lemon juice has a low pH, so it is acidic."
    assert len(line) == 42
    res = measure_wrap(f"{line} {line}", LAYOUT["caption"], fonts)
    assert res.fits, res
    assert len(res.lines) <= 2


def test_measurement_uses_slot_weight(fonts: FontSet) -> None:
    text = "Shared electrons"
    regular = measure_wrap(text, Slot("r", 0, 0, 1000, 100, 30, 1, "regular"), fonts)
    bold = measure_wrap(text, Slot("b", 0, 0, 1000, 100, 30, 1, "bold"), fonts)
    assert bold.width_px > regular.width_px


def test_from_dir_requires_all_weights(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(FileNotFoundError):
        FontSet.from_dir(tmp_path)
