import json

from app.pipeline.gates.g1_schema import g1_validate
from app.pipeline.gates.types import GateResult, Violation, format_violations
from app.pipeline.storyboard import Storyboard
from tests.fixtures.storyboards import minimal_storyboard_dict


def test_valid_passes() -> None:
    raw = minimal_storyboard_dict()
    for payload in (raw, json.dumps(raw)):
        sb, result = g1_validate(payload)
        assert isinstance(sb, Storyboard)
        assert result.gate == "G1"
        assert result.passed and result.violations == []


def test_invalid_json() -> None:
    sb, result = g1_validate('{"schema_version": "1", "scenes": [')
    assert sb is None
    assert not result.passed
    assert [v.code for v in result.violations] == ["invalid_json"]
    assert result.violations[0].scene_index is None

    sb, result = g1_validate("[1, 2, 3]")
    assert sb is None
    assert [v.code for v in result.violations] == ["schema"]


def test_schema_error_messages_name_scene_index() -> None:
    raw = minimal_storyboard_dict()
    raw["scenes"][2]["visual"]["from_ph"] = 21
    raw["scenes"][3]["on_screen_title"] = "x" * 41
    raw["title"] = "t" * 61
    sb, result = g1_validate(raw)
    assert sb is None
    by_scene = {v.scene_index: v for v in result.violations}
    assert set(by_scene) == {None, 2, 3}
    assert all(v.gate == "G1" and v.code == "schema" for v in result.violations)
    assert by_scene[2].message.startswith("Scene 2: visual.")
    assert "from_ph" in by_scene[2].message
    assert "21" in by_scene[2].message
    assert by_scene[3].message.startswith("Scene 3: on_screen_title:")
    assert by_scene[None].message.startswith("title:")


def test_flattened_payload_accepted() -> None:
    raw = minimal_storyboard_dict()
    for scene in raw["scenes"]:
        for key in ("subtitle", "points", "markers", "molecule", "compound", "rows"):
            scene["visual"].setdefault(key, None)
    sb, result = g1_validate(json.dumps(raw))
    assert result.passed, result.violations
    assert sb == Storyboard.model_validate(minimal_storyboard_dict())


def test_format_violations_numbered_verbatim() -> None:
    violations = [
        Violation("G1", "schema", "Scene 2: visual.from_ph: too big"),
        Violation("G2", "fact_missing", "Narration never states that 7 is neutral."),
    ]
    assert format_violations(violations) == (
        "1. Scene 2: visual.from_ph: too big\n2. Narration never states that 7 is neutral."
    )
    result = GateResult("G2", False, violations, {"n": 2})
    assert result.to_dict() == {
        "gate": "G2",
        "passed": False,
        "violations": [
            {
                "gate": "G1",
                "code": "schema",
                "message": "Scene 2: visual.from_ph: too big",
                "scene_index": None,
            },
            {
                "gate": "G2",
                "code": "fact_missing",
                "message": "Narration never states that 7 is neutral.",
                "scene_index": None,
            },
        ],
        "details": {"n": 2},
    }
