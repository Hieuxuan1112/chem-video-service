import re
from typing import Any

import pytest

from app.pipeline.gates.g2_layout import g2_layout
from app.pipeline.storyboard import Storyboard
from app.render.layout import FontSet
from app.topics.registry import TOPICS_ROOT, TopicRegistry

REGISTRY = TopicRegistry.load(TOPICS_ROOT)


def _raw(topic_id: str) -> dict[str, Any]:
    return REGISTRY.curated_storyboard(topic_id).model_dump(mode="json")


@pytest.mark.parametrize("topic_id", ["ph_scale", "covalent_bonds", "ionic_vs_covalent"])
def test_valid_base_passes(topic_id: str, fonts: FontSet) -> None:
    result = g2_layout(REGISTRY.curated_storyboard(topic_id), fonts)
    assert result.passed, result.violations
    assert result.gate == "G2_layout"
    assert result.details["strings_checked"] > 10


def test_overlong_title_word_mix_fails_with_numbers_in_message(fonts: FontSet) -> None:
    raw = _raw("ph_scale")
    raw["scenes"][2]["on_screen_title"] = "WOW MMM WWW MMM WWW MMM WWW MMM WWW MMMM"
    result = g2_layout(Storyboard.model_validate(raw), fonts)
    (v,) = result.violations
    assert (v.gate, v.code, v.scene_index) == ("G2d", "text_overflow", 2)
    assert v.message.startswith("Scene 2: 'on_screen_title' needs 2 lines / ")
    assert re.search(r"slot allows 1 lines / 1152px\. Shorten it\.$", v.message)


def test_table_cell_overflow_fails(fonts: FontSet) -> None:
    raw = _raw("ionic_vs_covalent")
    raw["scenes"][4]["visual"]["left_header"] = "WWWWWWWWWWWWWWWW"
    raw["scenes"][4]["visual"]["rows"][1]["right"] = "WWWWWWWWWWWWWWWWWWWWWWWWWWWW"
    result = g2_layout(Storyboard.model_validate(raw), fonts)
    fields = sorted(v.message.split("'")[1] for v in result.violations)
    assert fields == ["left_header", "rows[1].right"]
    assert all(v.scene_index == 4 for v in result.violations)


def test_key_point_three_lines_fails(fonts: FontSet) -> None:
    raw = _raw("covalent_bonds")
    raw["scenes"][5]["visual"]["points"][2] = (
        "WWWWWWWWWWWWWWWWWWW WWWWWWWWWWWWWWWWWWW WWWWWWWWWWWWWWWWWWWW"
    )
    result = g2_layout(Storyboard.model_validate(raw), fonts)
    (v,) = result.violations
    assert "'points[2]' needs 3 lines" in v.message
    assert "slot allows 2 lines" in v.message
