"""Storyboard schema v1 (spec §5.3): the LLM selects from a closed visual catalogue; code resolves
every number at render-plan time."""

import copy
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.topics.chemistry_data import ElementSymbol, IonicCompound, Molecule, SubstanceName

SCHEMA_VERSION = "1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=False)


class TitleCard(StrictModel):
    type: Literal["title_card"]
    subtitle: str = Field(max_length=60)


class KeyPoints(StrictModel):
    type: Literal["key_points"]
    points: list[Annotated[str, Field(max_length=60)]] = Field(min_length=2, max_length=4)


class PhScale(StrictModel):
    type: Literal["ph_scale"]
    markers: list[SubstanceName] = Field(min_length=2, max_length=5)
    highlight: Literal["acidic", "neutral", "basic", "none"]


class LogRatio(StrictModel):
    type: Literal["log_ratio"]
    from_ph: int = Field(ge=0, le=14)
    to_ph: int = Field(ge=0, le=14)


class AtomShells(StrictModel):
    type: Literal["atom_shells"]
    elements: list[ElementSymbol] = Field(min_length=1, max_length=2)


class ElectronSharing(StrictModel):
    type: Literal["electron_sharing"]
    molecule: Molecule


class ElectronTransfer(StrictModel):
    type: Literal["electron_transfer"]
    compound: IonicCompound


class TableRow(StrictModel):
    aspect: str = Field(max_length=20)
    left: str = Field(max_length=28)
    right: str = Field(max_length=28)


class ComparisonTable(StrictModel):
    type: Literal["comparison_table"]
    left_header: str = Field(max_length=16)
    right_header: str = Field(max_length=16)
    rows: list[TableRow] = Field(min_length=2, max_length=4)


Visual = Annotated[
    TitleCard
    | KeyPoints
    | PhScale
    | LogRatio
    | AtomShells
    | ElectronSharing
    | ElectronTransfer
    | ComparisonTable,
    Field(discriminator="type"),
]

_VISUAL_MODELS: dict[str, type[StrictModel]] = {
    "title_card": TitleCard,
    "key_points": KeyPoints,
    "ph_scale": PhScale,
    "log_ratio": LogRatio,
    "atom_shells": AtomShells,
    "electron_sharing": ElectronSharing,
    "electron_transfer": ElectronTransfer,
    "comparison_table": ComparisonTable,
}
VISUAL_TYPES: tuple[str, ...] = tuple(_VISUAL_MODELS)


class Scene(StrictModel):
    on_screen_title: str = Field(min_length=1, max_length=40)
    narration: str = Field(min_length=1, max_length=600)  # 25-60 words is checked in G2a
    visual: Visual


class Storyboard(StrictModel):
    schema_version: Literal["1"] = "1"
    topic_id: str
    title: str = Field(max_length=60)
    learning_objective: str = Field(max_length=160)
    scenes: list[Scene] = Field(min_length=5, max_length=7)


def word_count(text: str) -> int:
    """Whitespace-separated tokens that contain at least one letter or digit."""
    return sum(1 for token in text.split() if any(ch.isalnum() for ch in token))


def strict_json_schema() -> dict[str, Any]:
    return Storyboard.model_json_schema()


def _enum(values: object) -> dict[str, Any]:
    return {"type": "string", "enum": [str(v) for v in values]}  # type: ignore[attr-defined]


def _string() -> dict[str, Any]:
    return {"type": "string"}


def provider_json_schema() -> dict[str, Any]:
    """Schema sent to the provider: types, enums, required fields and structure only.

    The visual is flattened into one object with a `type` enum and every visual field optional,
    because $ref/oneOf is not accepted. Length, item-count and numeric bounds are left out:
    Gemini rejected both the strict schema and a bounded flattened one with "too many states
    for serving" (spec §5.3). G1 validates the strict model, which keeps every bound."""
    visual = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": list(VISUAL_TYPES)},
            "subtitle": _string(),
            "points": {"type": "array", "items": _string()},
            "markers": {"type": "array", "items": _enum(SubstanceName)},
            "highlight": _enum(["acidic", "neutral", "basic", "none"]),
            "from_ph": {"type": "integer"},
            "to_ph": {"type": "integer"},
            "elements": {"type": "array", "items": _enum(ElementSymbol)},
            "molecule": _enum(Molecule),
            "compound": _enum(IonicCompound),
            "left_header": _string(),
            "right_header": _string(),
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"aspect": _string(), "left": _string(), "right": _string()},
                    "required": ["aspect", "left", "right"],
                },
            },
        },
        "required": ["type"],
    }
    scene = {
        "type": "object",
        "properties": {"on_screen_title": _string(), "narration": _string(), "visual": visual},
        "required": ["on_screen_title", "narration", "visual"],
    }
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": [SCHEMA_VERSION]},
            "topic_id": _string(),
            "title": _string(),
            "learning_objective": _string(),
            "scenes": {"type": "array", "items": scene},
        },
        "required": ["schema_version", "topic_id", "title", "learning_objective", "scenes"],
    }


def coerce_flattened(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop null visual keys that do not belong to the declared visual type. Non-null foreign keys
    are kept, so strict validation still rejects them. The input is not mutated."""
    out = copy.deepcopy(raw)
    scenes = out.get("scenes")
    if not isinstance(scenes, list):
        return out
    for scene in scenes:
        visual = scene.get("visual") if isinstance(scene, dict) else None
        if not isinstance(visual, dict):
            continue
        model = _VISUAL_MODELS.get(str(visual.get("type")))
        if model is None:
            continue
        allowed = set(model.model_fields)
        for key in [k for k, v in visual.items() if v is None and k not in allowed]:
            del visual[key]
    return out
