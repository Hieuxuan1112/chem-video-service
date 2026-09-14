import copy
import json

import pytest
from pydantic import ValidationError

from app.pipeline.storyboard import (
    SCHEMA_VERSION,
    VISUAL_TYPES,
    AtomShells,
    ComparisonTable,
    ElectronSharing,
    ElectronTransfer,
    KeyPoints,
    LogRatio,
    PhScale,
    Storyboard,
    TitleCard,
    coerce_flattened,
    provider_json_schema,
    strict_json_schema,
    word_count,
)

VISUAL_MODELS_BY_TYPE = {
    "title_card": TitleCard,
    "key_points": KeyPoints,
    "ph_scale": PhScale,
    "log_ratio": LogRatio,
    "atom_shells": AtomShells,
    "electron_sharing": ElectronSharing,
    "electron_transfer": ElectronTransfer,
    "comparison_table": ComparisonTable,
}


def _base() -> dict:
    return {
        "schema_version": "1",
        "topic_id": "ph_scale",
        "title": "How the pH scale works",
        "learning_objective": "Explain what pH numbers mean for acids and bases.",
        "scenes": [
            {
                "on_screen_title": "Your question",
                "narration": "Let's explore the pH scale.",
                "visual": {"type": "title_card", "subtitle": "Acids, bases and neutral"},
            },
            {
                "on_screen_title": "The scale",
                "narration": "pH runs from 0 to 14.",
                "visual": {
                    "type": "ph_scale",
                    "markers": ["lemon_juice", "bleach"],
                    "highlight": "none",
                },
            },
            {
                "on_screen_title": "Ten times",
                "narration": "Each step is ten times.",
                "visual": {"type": "log_ratio", "from_ph": 3, "to_ph": 4},
            },
            {
                "on_screen_title": "Atoms",
                "narration": "Hydrogen ions matter.",
                "visual": {"type": "atom_shells", "elements": ["H"]},
            },
            {
                "on_screen_title": "Compare",
                "narration": "Acids versus bases.",
                "visual": {
                    "type": "comparison_table",
                    "left_header": "Acids",
                    "right_header": "Bases",
                    "rows": [
                        {"aspect": "pH", "left": "Below 7", "right": "Above 7"},
                        {"aspect": "Example", "left": "Lemon juice", "right": "Bleach"},
                    ],
                },
            },
            {
                "on_screen_title": "Key points",
                "narration": "Remember these ideas.",
                "visual": {
                    "type": "key_points",
                    "points": ["Below 7 is acidic", "Above 7 is basic"],
                },
            },
        ],
    }


def test_valid_storyboard_parses() -> None:
    sb = Storyboard.model_validate(_base())
    assert sb.schema_version == SCHEMA_VERSION == "1"
    assert [s.visual.type for s in sb.scenes] == [
        "title_card",
        "ph_scale",
        "log_ratio",
        "atom_shells",
        "comparison_table",
        "key_points",
    ]
    assert isinstance(sb.scenes[4].visual, ComparisonTable)
    roundtrip = Storyboard.model_validate_json(sb.model_dump_json())
    assert roundtrip == sb


def test_visual_catalogue() -> None:
    assert VISUAL_TYPES == (
        "title_card",
        "key_points",
        "ph_scale",
        "log_ratio",
        "atom_shells",
        "electron_sharing",
        "electron_transfer",
        "comparison_table",
    )
    assert set(VISUAL_MODELS_BY_TYPE) == set(VISUAL_TYPES)


def test_unknown_visual_type_rejected() -> None:
    raw = _base()
    raw["scenes"][1]["visual"] = {"type": "bar_chart", "values": [1, 2]}
    with pytest.raises(ValidationError):
        Storyboard.model_validate(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(extra=1),
        lambda r: r["scenes"][0].update(duration_s=5),
        lambda r: r["scenes"][2]["visual"].update(ratio=10),  # code resolves ratios, not the LLM
    ],
)
def test_extra_field_rejected(mutate) -> None:  # type: ignore[no-untyped-def]
    raw = _base()
    mutate(raw)
    with pytest.raises(ValidationError):
        Storyboard.model_validate(raw)


def test_scene_count_bounds() -> None:
    raw = _base()
    raw["scenes"] = raw["scenes"][:4]
    with pytest.raises(ValidationError):
        Storyboard.model_validate(raw)
    raw = _base()
    raw["scenes"].append(copy.deepcopy(raw["scenes"][-1]))
    assert len(Storyboard.model_validate(raw).scenes) == 7
    raw["scenes"].append(copy.deepcopy(raw["scenes"][-1]))
    with pytest.raises(ValidationError):
        Storyboard.model_validate(raw)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("title",), "x" * 61),
        (("learning_objective",), "x" * 161),
        (("scenes", 0, "on_screen_title"), "x" * 41),
        (("scenes", 0, "visual", "subtitle"), "x" * 61),
        (("scenes", 5, "visual", "points"), ["only one"]),
        (("scenes", 1, "visual", "markers"), ["lemon_juice", "unobtainium"]),
        (("scenes", 2, "visual", "from_ph"), 15),
        (("scenes", 3, "visual", "elements"), ["H", "Na", "Cl"]),
        (("scenes", 4, "visual", "left_header"), "x" * 17),
    ],
)
def test_title_length_bound(path: tuple, value: object) -> None:
    raw = _base()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        Storyboard.model_validate(raw)


def test_strict_schema_is_pydantic_schema() -> None:
    assert strict_json_schema() == Storyboard.model_json_schema()


def _walk_keys(node: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(node, dict):
        for k, v in node.items():
            keys.add(k)
            keys |= _walk_keys(v)
    elif isinstance(node, list):
        for item in node:
            keys |= _walk_keys(item)
    return keys


def test_provider_schema_has_no_refs_or_oneof() -> None:
    schema = provider_json_schema()
    assert not _walk_keys(schema) & {"$ref", "$defs", "oneOf", "anyOf", "allOf", "discriminator"}
    json.dumps(schema)
    visual = schema["properties"]["scenes"]["items"]["properties"]["visual"]
    assert visual["properties"]["type"]["enum"] == list(VISUAL_TYPES)
    assert visual["required"] == ["type"]
    # every field of every visual model is representable in the flattened object
    for visual_type, model in VISUAL_MODELS_BY_TYPE.items():
        for field in model.model_fields:
            assert field in visual["properties"], (visual_type, field)


def _keywords(node: object) -> set[str]:
    """JSON Schema keywords used anywhere in the schema; property names are not keywords."""
    found: set[str] = set()
    if isinstance(node, dict):
        for k, v in node.items():
            found.add(k)
            children = v.values() if k == "properties" and isinstance(v, dict) else [v]
            for child in children:
                found |= _keywords(child)
    elif isinstance(node, list):
        for item in node:
            found |= _keywords(item)
    return found


_PROVIDER_DROPPED = {"maxLength", "minLength", "minimum", "maximum", "minItems", "maxItems"}


def test_provider_schema_keeps_only_type_enum_required() -> None:
    # Gemini rejected bounded schemas ("too many states for serving"); limits stay in G1 (spec §5.3)
    schema = provider_json_schema()
    assert not _walk_keys(schema) & _PROVIDER_DROPPED
    assert _keywords(schema) == {"type", "properties", "items", "enum", "required"}
    # the application schema keeps every bound
    assert _walk_keys(strict_json_schema()) >= _PROVIDER_DROPPED
    raw = _base()
    raw["scenes"][0]["narration"] = "word " * 121  # 605 chars > 600
    with pytest.raises(ValidationError):
        Storyboard.model_validate(raw)


def test_flattened_payload_validates_into_strict_model() -> None:
    raw = _base()
    all_fields = provider_json_schema()["properties"]["scenes"]["items"]["properties"]["visual"][
        "properties"
    ]
    for scene in raw["scenes"]:
        for field in all_fields:
            scene["visual"].setdefault(field, None)
    with pytest.raises(ValidationError):
        Storyboard.model_validate(raw)
    coerced = coerce_flattened(raw)
    assert Storyboard.model_validate(coerced) == Storyboard.model_validate(_base())
    assert raw["scenes"][0]["visual"]["points"] is None  # input is not mutated


def test_coerce_keeps_non_null_foreign_keys_so_validation_still_fails() -> None:
    raw = _base()
    raw["scenes"][0]["visual"]["points"] = ["sneaky", "extra"]
    with pytest.raises(ValidationError):
        Storyboard.model_validate(coerce_flattened(raw))


@pytest.mark.parametrize(
    ("text", "n"),
    [
        ("A pH of 7 is neutral.", 6),
        ("  lemon-juice, vinegar & coffee!  ", 3),
        ("", 0),
        ("... —", 0),
    ],
)
def test_word_count(text: str, n: int) -> None:
    assert word_count(text) == n
