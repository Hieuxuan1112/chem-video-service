import itertools
import math
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from manim import Dot

from app.pipeline.storyboard import LogRatio, PhScale
from app.render.manim import theme
from app.render.manim.scene_factory import assert_inside_safe_area
from app.render.plan import ResolvedLogRatio, ResolvedPhScale, resolve_visual
from app.topics.chemistry_data import SubstanceName
from tests.render.harness import (
    Built,
    assert_no_text_overlap,
    build_offscreen,
    plan_for,
    tagged,
    texts,
)

pytestmark = pytest.mark.render

QUESTION = "How does the pH scale work?"
LEFT_CLUSTER = ["battery_acid", "stomach_acid", "lemon_juice", "vinegar", "orange_juice"]
RIGHT_CLUSTER = ["hand_soap", "milk_of_magnesia", "household_ammonia", "bleach", "drain_cleaner"]


def _ph_scale(markers: Sequence[str], highlight: str = "acidic") -> ResolvedPhScale:
    visual = PhScale(
        type="ph_scale", markers=[SubstanceName(m) for m in markers], highlight=highlight
    )
    resolved = resolve_visual(visual, question=QUESTION, title="pH")
    assert isinstance(resolved, ResolvedPhScale)
    return resolved


@pytest.fixture
def build(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], Built]:
    return lambda visual: build_offscreen(plan_for(visual, question=QUESTION), monkeypatch)


@pytest.mark.parametrize("markers", [LEFT_CLUSTER, RIGHT_CLUSTER, ["milk", "pure_water"]])
def test_marker_x_matches_ph_position(build: Callable[[Any], Built], markers: list[str]) -> None:
    visual = _ph_scale(markers)
    scene = build(visual).scene

    segments = {s.ph_index: s for s in tagged(scene, "ph_index")}
    arrows = {a.marker_name: a for a in tagged(scene, "marker_name")}
    assert set(arrows) == {m.name for m in visual.markers}
    for marker in visual.markers:
        whole = math.floor(marker.ph)
        segment = segments[whole]
        expected = segment.get_center()[0] + (marker.ph - whole) * segment.width
        assert abs(arrows[marker.name].get_end()[0] - expected) < 0.01, marker.name


def test_segment_colors_discrete_and_ordered(build: Callable[[Any], Built]) -> None:
    scene = build(_ph_scale(["lemon_juice", "bleach"], highlight="none")).scene

    segments = sorted(tagged(scene, "ph_index"), key=lambda s: s.ph_index)
    assert [s.ph_index for s in segments] == list(range(15))
    assert [s.get_fill_color().to_hex().upper() for s in segments] == [
        c.upper() for c in theme.PH_COLORS
    ]
    centers = [s.get_center()[0] for s in segments]
    assert all(a < b for a, b in itertools.pairwise(centers))
    assert len({round(s.width, 6) for s in segments}) == 1


@pytest.mark.parametrize(
    ("from_ph", "to_ph", "from_dots", "to_dots"),
    [(3, 4, 10, 1), (4, 3, 1, 10), (4, 2, 1, 100), (7, 9, 100, 1)],
)
def test_log_ratio_dot_counts(
    build: Callable[[Any], Built], from_ph: int, to_ph: int, from_dots: int, to_dots: int
) -> None:
    visual = resolve_visual(
        LogRatio(type="log_ratio", from_ph=from_ph, to_ph=to_ph), question=QUESTION, title="pH"
    )
    assert isinstance(visual, ResolvedLogRatio)
    scene = build(visual).scene

    grids = {g.h_ion_ph: g for g in tagged(scene, "h_ion_ph")}
    counts = {ph: sum(isinstance(m, Dot) for m in g.get_family()) for ph, g in grids.items()}
    assert counts == {from_ph: from_dots, to_ph: to_dots}
    big = grids[from_ph if from_dots == 100 else to_ph] if 100 in (from_dots, to_dots) else None
    if big is not None:  # 100 dots are drawn as a 10 x 10 grid
        dots = [m for m in big.get_family() if isinstance(m, Dot)]
        assert len({round(d.get_center()[0], 4) for d in dots}) == 10
        assert len({round(d.get_center()[1], 4) for d in dots}) == 10
    drawn = [t.original_text for t in texts(scene)]
    assert f"{visual.ratio}\u00d7 more H\u207a" in drawn


@pytest.mark.parametrize("markers", [LEFT_CLUSTER, RIGHT_CLUSTER])
@pytest.mark.parametrize("highlight", ["acidic", "neutral", "basic"])
def test_inside_safe_area_with_five_markers(
    build: Callable[[Any], Built], markers: list[str], highlight: str
) -> None:
    result = build(_ph_scale(markers, highlight))

    assert_inside_safe_area(result.scene.mobjects)
    assert_no_text_overlap(result.scene)
    assert min(g.text_scale for g in result.groups) >= 0.9
