import pytest

from app.config import Settings
from app.pipeline.gates.g1_schema import g1_validate
from app.pipeline.gates.g2 import G2Outcome, run_g2
from app.pipeline.gates.g2_rules import estimated_duration_s
from app.pipeline.storyboard import Storyboard
from app.render.layout import FontSet
from app.topics.registry import TOPICS_ROOT, TopicRegistry

REGISTRY = TopicRegistry.load(TOPICS_ROOT)
SETTINGS = Settings(_env_file=None)
TOPICS = ["ph_scale", "covalent_bonds", "ionic_vs_covalent"]


@pytest.mark.parametrize("topic_id", TOPICS)
def test_each_curated_storyboard_passes_g1_and_g2(topic_id: str, fonts: FontSet) -> None:
    raw_json = (REGISTRY.pack_dir(topic_id) / "curated_storyboard.json").read_text(encoding="utf-8")
    sb, g1 = g1_validate(raw_json)
    assert g1.passed, g1.violations
    assert sb is not None and sb.topic_id == topic_id
    outcome = run_g2(sb, REGISTRY.get(topic_id), SETTINGS, fonts)
    assert outcome.passed, [v.message for v in outcome.violations]
    assert [r.gate for r in outcome.results] == ["G2_rules", "G2_claims", "G2_layout"]


@pytest.mark.parametrize("topic_id", TOPICS)
def test_curated_duration_in_range(topic_id: str) -> None:
    duration = estimated_duration_s(REGISTRY.curated_storyboard(topic_id), SETTINGS)
    assert SETTINGS.min_total_s <= duration <= SETTINGS.max_total_s


def test_outcome_aggregates_violations_across_gates(fonts: FontSet) -> None:
    raw = REGISTRY.curated_storyboard("ph_scale").model_dump(mode="json")
    raw["scenes"][2]["narration"] = "Lemon juice has a pH of 9. Bleach is basic."  # G2a + G2c
    raw["scenes"][2]["on_screen_title"] = "WOW MMM WWW MMM WWW MMM WWW MMM WWW MMMM"  # G2d
    outcome = run_g2(Storyboard.model_validate(raw), REGISTRY.get("ph_scale"), SETTINGS, fonts)
    assert isinstance(outcome, G2Outcome)
    assert not outcome.passed
    assert {v.gate for v in outcome.violations} == {"G2a", "G2c", "G2d"}
    assert outcome.unverified_claims == outcome.results[1].details["unverified_numeric_claims"]
