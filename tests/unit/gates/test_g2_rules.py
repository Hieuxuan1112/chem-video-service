from collections.abc import Callable
from typing import Any

import pytest

from app.config import Settings
from app.pipeline.gates.g2_rules import estimated_duration_s, g2_rules
from app.pipeline.storyboard import Storyboard
from app.topics.registry import TOPICS_ROOT, TopicRegistry

REGISTRY = TopicRegistry.load(TOPICS_ROOT)
SETTINGS = Settings(_env_file=None)
FILLER = " ".join(["word"] * 70)


def _raw(topic_id: str) -> dict[str, Any]:
    return REGISTRY.curated_storyboard(topic_id).model_dump(mode="json")


def _codes(topic_id: str, raw: dict[str, Any]) -> list[str]:
    result = g2_rules(Storyboard.model_validate(raw), REGISTRY.get(topic_id), SETTINGS)
    assert result.passed is (not result.violations)
    return [v.code for v in result.violations]


@pytest.mark.parametrize("topic_id", ["ph_scale", "covalent_bonds", "ionic_vs_covalent"])
def test_valid_base_passes(topic_id: str) -> None:
    result = g2_rules(REGISTRY.curated_storyboard(topic_id), REGISTRY.get(topic_id), SETTINGS)
    assert result.passed, result.violations
    assert 45 <= result.details["estimated_duration_s"] <= 100


def test_estimated_duration_formula() -> None:
    sb = REGISTRY.curated_storyboard("ph_scale")
    words = [33, 36, 32, 35, 35, 31]
    expected = sum(w / 2.5 + 0.3 + 0.7 for w in words) + (1.5 - 0.7)
    assert estimated_duration_s(sb, SETTINGS) == pytest.approx(expected)


def _set_narration(i: int, text: str) -> Callable[[dict], None]:
    def mutate(raw: dict) -> None:
        raw["scenes"][i]["narration"] = text

    return mutate


def _set_visual(i: int, visual: dict) -> Callable[[dict], None]:
    def mutate(raw: dict) -> None:
        raw["scenes"][i]["visual"] = visual

    return mutate


def _swap_first_two(raw: dict) -> None:
    raw["scenes"][0], raw["scenes"][1] = raw["scenes"][1], raw["scenes"][0]


def _drop_last(raw: dict) -> None:
    raw["scenes"].pop()


def _long_narrations(raw: dict) -> None:
    for scene in raw["scenes"]:
        scene["narration"] = scene["narration"] + " " + " ".join(["more"] * 24)


def _short_narrations(raw: dict) -> None:
    raw["scenes"] = raw["scenes"][:5]
    for scene in raw["scenes"]:
        words = scene["narration"].split()
        scene["narration"] = " ".join(words[:12])  # 5 x (12/2.5 + 1) + 0.8 = 29.8 s


def _duplicate_scene_visual(raw: dict) -> None:
    raw["scenes"][2]["visual"] = raw["scenes"][1]["visual"]


def _remove_required_visual(raw: dict) -> None:
    for i in (1, 2):
        raw["scenes"][i]["visual"] = {"type": "log_ratio", "from_ph": 3 + i, "to_ph": 3}


def _remove_fact(raw: dict) -> None:
    for scene in raw["scenes"]:
        scene["narration"] = scene["narration"].replace("0 to 14", "zero up to a high number")


def _wrong_topic(raw: dict) -> None:
    raw["topic_id"] = "covalent_bonds"


CASES: list[tuple[str, str, Callable[[dict], None]]] = [
    ("scene_words_low", "ph_scale", _set_narration(2, "Lemon juice is acidic. Bleach is basic.")),
    ("scene_words_high", "ph_scale", _set_narration(2, "Lemon juice is acidic. " + FILLER)),
    ("duration_out_of_range", "ph_scale", _long_narrations),
    ("duration_out_of_range", "ph_scale", _short_narrations),
    ("first_not_title_card", "ph_scale", _swap_first_two),
    ("last_not_key_points", "ph_scale", _drop_last),
    ("visual_not_allowed", "ph_scale", _set_visual(3, {"type": "atom_shells", "elements": ["H"]})),
    ("required_visual_missing", "ph_scale", _remove_required_visual),
    ("duplicate_visual", "ph_scale", _duplicate_scene_visual),
    ("fact_missing", "ph_scale", _remove_fact),
    (
        "misconception",
        "ph_scale",
        _set_narration(
            3,
            "The pH scale is logarithmic. A pH of 7 is acidic, which surprises many people. "
            "Each step of one pH unit means ten times more hydrogen ions.",
        ),
    ),
    ("topic_mismatch", "ph_scale", _wrong_topic),
    (
        "log_ratio_delta",
        "ph_scale",
        _set_visual(3, {"type": "log_ratio", "from_ph": 2, "to_ph": 9}),
    ),
    (
        "log_ratio_delta",
        "ph_scale",
        _set_visual(3, {"type": "log_ratio", "from_ph": 4, "to_ph": 4}),
    ),
    (
        "element_not_allowed",
        "covalent_bonds",
        _set_visual(1, {"type": "atom_shells", "elements": ["H", "Na"]}),
    ),
    (
        "compound_not_allowed",
        "covalent_bonds",
        _set_visual(2, {"type": "electron_transfer", "compound": "NaCl"}),
    ),
]


@pytest.mark.parametrize(("code", "topic_id", "mutate"), CASES, ids=[c[0] for c in CASES])
def test_each_rule_has_a_failing_fixture(
    code: str, topic_id: str, mutate: Callable[[dict], None]
) -> None:
    raw = _raw(topic_id)
    mutate(raw)
    assert code in _codes(topic_id, raw)


def test_violation_messages_are_located_and_actionable() -> None:
    raw = _raw("ph_scale")
    raw["scenes"][2]["narration"] = "Lemon juice is acidic. Bleach is basic."
    result = g2_rules(Storyboard.model_validate(raw), REGISTRY.get("ph_scale"), SETTINGS)
    low = next(v for v in result.violations if v.code == "scene_words_low")
    assert low.scene_index == 2
    assert low.gate == "G2a"
    assert "Scene 2" in low.message and "7 words" in low.message and "25-60" in low.message


def test_molecule_not_allowed_with_restricted_pack() -> None:
    pack = REGISTRY.get("covalent_bonds").model_copy(update={"allowed_molecules": ["H2", "HCl"]})
    raw = _raw("covalent_bonds")  # uses HCl, O2, N2
    result = g2_rules(Storyboard.model_validate(raw), pack, SETTINGS)
    bad = [v for v in result.violations if v.code == "molecule_not_allowed"]
    assert [v.scene_index for v in bad] == [3, 4]
    assert all(v.gate == "G2b" for v in bad)


def test_ionic_roles_asserted(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.topics import chemistry_data

    broken = dict(chemistry_data.IONIC_COMPOUNDS)
    nacl = broken[chemistry_data.IonicCompound.NaCl]
    broken[chemistry_data.IonicCompound.NaCl] = type(nacl)(
        **{**nacl.__dict__, "donor": chemistry_data.ElementSymbol.Cl}
    )
    monkeypatch.setattr("app.pipeline.gates.g2_rules.IONIC_COMPOUNDS", broken)
    assert "ionic_roles_invalid" in _codes("ionic_vs_covalent", _raw("ionic_vs_covalent"))
