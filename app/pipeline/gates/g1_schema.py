"""G1: strict schema validation with scene-addressed violation messages."""

import json
from typing import Any

from pydantic import ValidationError

from app.pipeline.gates.types import GateResult, Violation
from app.pipeline.storyboard import VISUAL_TYPES, Storyboard, coerce_flattened

GATE = "G1"
_MAX_INPUT_REPR = 80


def _describe(error: Any) -> Violation:
    loc = list(error.get("loc", ()))
    scene_index: int | None = None
    if len(loc) >= 2 and loc[0] == "scenes" and isinstance(loc[1], int):
        scene_index = loc[1]
        loc = loc[2:]
    # drop the discriminator tag pydantic inserts after "visual"
    if len(loc) >= 2 and loc[0] == "visual" and loc[1] in VISUAL_TYPES:
        loc = [loc[0], *loc[2:]]
    path = ".".join(str(part) for part in loc) or "storyboard"
    message = f"{path}: {error.get('msg', 'invalid')}"
    value = error.get("input")
    if value is not None and not isinstance(value, dict | list):
        shown = repr(value)
        if len(shown) > _MAX_INPUT_REPR:
            shown = shown[:_MAX_INPUT_REPR] + "..."
        message += f" (got {shown})"
    if scene_index is not None:
        message = f"Scene {scene_index}: {message}"
    return Violation(GATE, "schema", message, scene_index)


def g1_validate(raw: str | dict[str, Any]) -> tuple[Storyboard | None, GateResult]:
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            violation = Violation(GATE, "invalid_json", f"Output is not valid JSON: {exc}")
            return None, GateResult(GATE, False, [violation])
    else:
        data = raw
    if not isinstance(data, dict):
        violation = Violation(GATE, "schema", "storyboard: expected a JSON object at the top level")
        return None, GateResult(GATE, False, [violation])
    try:
        storyboard = Storyboard.model_validate(coerce_flattened(data))
    except ValidationError as exc:
        return None, GateResult(GATE, False, [_describe(e) for e in exc.errors()])
    return storyboard, GateResult(GATE, True, [])
