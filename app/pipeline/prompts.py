"""Versioned writer, rewrite and critic prompts (spec §5.4).

The limits stated in the prompts are read from the strict storyboard schema, so the words the model
reads and the bounds G1 enforces cannot drift apart (the provider schema carries no bounds, §5.3).
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from string import Template
from typing import Any

from app.config import Settings
from app.pipeline.storyboard import SCHEMA_VERSION, strict_json_schema
from app.topics.chemistry_data import ELEMENTS, IONIC_COMPOUNDS, MOLECULES, SUBSTANCE_PH
from app.topics.pack import TopicPack

PROMPT_VERSIONS: Mapping[str, str] = {
    "writer": "writer_v1",
    "rewrite": "rewrite_v1",
    "critic": "critic_v1",
}
PROMPTS_DIR = Path(__file__).parent / "prompts"
_USER_MARKER = "<!-- user -->"


def _template(kind: str) -> tuple[Template, Template]:
    text = (PROMPTS_DIR / f"{PROMPT_VERSIONS[kind]}.md").read_text(encoding="utf-8")
    system, user = text.split(_USER_MARKER, 1)
    return Template(system.strip()), Template(user.strip())


def _escape_query(query: str) -> str:
    """The query is data inside <learner_query>; angle brackets could close the wrapper."""
    return query.replace("<", "(").replace(">", ")")


def _bound(defs: dict[str, Any], *path: str) -> dict[str, Any]:
    node: Any = defs
    for key in path:
        node = node[key]
    return node  # type: ignore[no-any-return]


def _limits() -> dict[str, int]:
    s = strict_json_schema()
    d = s["$defs"]
    return {
        "title_max": s["properties"]["title"]["maxLength"],
        "objective_max": s["properties"]["learning_objective"]["maxLength"],
        "scenes_min": s["properties"]["scenes"]["minItems"],
        "scenes_max": s["properties"]["scenes"]["maxItems"],
        "screen_title_min": _bound(d, "Scene", "properties", "on_screen_title")["minLength"],
        "screen_title_max": _bound(d, "Scene", "properties", "on_screen_title")["maxLength"],
        "narration_max": _bound(d, "Scene", "properties", "narration")["maxLength"],
        "subtitle_max": _bound(d, "TitleCard", "properties", "subtitle")["maxLength"],
        "points_min": _bound(d, "KeyPoints", "properties", "points")["minItems"],
        "points_max": _bound(d, "KeyPoints", "properties", "points")["maxItems"],
        "point_max": _bound(d, "KeyPoints", "properties", "points", "items")["maxLength"],
        "markers_min": _bound(d, "PhScale", "properties", "markers")["minItems"],
        "markers_max": _bound(d, "PhScale", "properties", "markers")["maxItems"],
        "ph_min": _bound(d, "LogRatio", "properties", "from_ph")["minimum"],
        "ph_max": _bound(d, "LogRatio", "properties", "from_ph")["maximum"],
        "elements_min": _bound(d, "AtomShells", "properties", "elements")["minItems"],
        "elements_max": _bound(d, "AtomShells", "properties", "elements")["maxItems"],
        "header_max": _bound(d, "ComparisonTable", "properties", "left_header")["maxLength"],
        "rows_min": _bound(d, "ComparisonTable", "properties", "rows")["minItems"],
        "rows_max": _bound(d, "ComparisonTable", "properties", "rows")["maxItems"],
        "aspect_max": _bound(d, "TableRow", "properties", "aspect")["maxLength"],
        "cell_max": _bound(d, "TableRow", "properties", "left")["maxLength"],
    }


def visual_catalogue_text(pack: TopicPack) -> str:
    lim = _limits()
    required = set(pack.required_visuals)
    elements = ", ".join(f"{e} ({ELEMENTS[e].name})" for e in pack.allowed_elements)
    molecules = ", ".join(f"{m} ({MOLECULES[m].name})" for m in pack.allowed_molecules)
    compounds = ", ".join(f"{c} ({IONIC_COMPOUNDS[c].name})" for c in pack.allowed_compounds)
    substances = ", ".join(f"{s.name} ({s.display})" for s in SUBSTANCE_PH.values())
    entries = {
        "title_card": (f"the opening scene. subtitle: at most {lim['subtitle_max']} characters."),
        "key_points": (
            "a short list that sums up ideas; the last scene uses it. points: "
            f"{lim['points_min']} to {lim['points_max']} strings of at most "
            f"{lim['point_max']} characters each."
        ),
        "ph_scale": (
            "the 0-14 scale with reference substances placed by code. markers: "
            f"{lim['markers_min']} to {lim['markers_max']} of [{substances}]; "
            "highlight: one of acidic, neutral, basic, none."
        ),
        "log_ratio": (
            "compares two pH values to show the tenfold steps. from_ph and to_ph: integers from "
            f"{lim['ph_min']} to {lim['ph_max']} that differ by 1 or 2."
        ),
        "atom_shells": (
            "electron shells of atoms. elements: "
            f"{lim['elements_min']} to {lim['elements_max']} of [{elements}]."
        ),
        "electron_sharing": f"atoms sharing electron pairs. molecule: one of [{molecules}].",
        "electron_transfer": (
            f"a metal giving electrons to a non-metal. compound: one of [{compounds}]."
        ),
        "comparison_table": (
            "two columns compared row by row. left_header and right_header: at most "
            f"{lim['header_max']} characters; rows: {lim['rows_min']} to {lim['rows_max']} "
            f"objects with aspect at most {lim['aspect_max']} characters, left and right at most "
            f"{lim['cell_max']} characters."
        ),
    }
    lines = []
    for visual in pack.allowed_visuals:
        mark = " (required)" if visual in required else ""
        lines.append(f"- {visual}{mark}: {entries[visual]}")
    return "\n".join(lines)


def _rules(pack: TopicPack, settings: Settings) -> str:
    lim = _limits()
    s = settings
    # Word totals that keep the estimated length in range whatever the scene count (G2a).
    padding_max = lim["scenes_max"] * (s.lead_in_s + s.tail_s) + (s.final_tail_s - s.tail_s)
    words_low = math.ceil(s.min_total_s * s.words_per_second)
    words_high = math.floor((s.max_total_s - padding_max) * s.words_per_second)
    rules = [
        f'topic_id: "{pack.id}"; schema_version: "{SCHEMA_VERSION}".',
        f"title: at most {lim['title_max']} characters.",
        f"learning_objective: at most {lim['objective_max']} characters.",
        f"{lim['scenes_min']} to {lim['scenes_max']} scenes.",
        "Scene 0 uses title_card; the last scene uses key_points.",
        f"on_screen_title: {lim['screen_title_min']} to {lim['screen_title_max']} characters.",
        f"narration: {s.min_words} to {s.max_words} words per scene and at most "
        f"{lim['narration_max']} characters.",
        f"Total narration: {words_low} to {words_high} words, so the video lasts "
        f"{s.min_total_s:.0f}-{s.max_total_s:.0f} seconds.",
        "Use only the visuals in the catalogue, include every required visual, and never repeat "
        "the exact same visual in two scenes.",
        "In each visual, include only the fields of its type, within the limits listed in the "
        "catalogue.",
        "State every authoritative fact clearly in the narration. Do not state any misconception.",
        "Plain text only: no markdown, no emoji.",
    ]
    return "\n".join(f"- {r}" for r in rules)


def _facts(pack: TopicPack) -> str:
    return "\n".join(f"- {f.statement}" for f in pack.required_facts)


def _misconceptions(pack: TopicPack) -> str:
    return "\n".join(f"- {m.explanation}" for m in pack.misconceptions)


def render_writer_prompt(pack: TopicPack, query: str, settings: Settings) -> tuple[str, str]:
    system, user = _template("writer")
    values = {
        "query": _escape_query(query),
        "facts": _facts(pack),
        "misconceptions": _misconceptions(pack),
        "catalogue": visual_catalogue_text(pack),
        "rules": _rules(pack, settings),
    }
    return system.substitute(values), user.substitute(values)


def render_rewrite_prompt(
    pack: TopicPack, query: str, previous_json: str, violations_text: str, settings: Settings
) -> tuple[str, str]:
    system, user = _template("rewrite")
    values = {
        "query": _escape_query(query),
        "facts": _facts(pack),
        "misconceptions": _misconceptions(pack),
        "catalogue": visual_catalogue_text(pack),
        "rules": _rules(pack, settings),
        "previous": previous_json,
        "violations": violations_text,
    }
    return system.substitute(values), user.substitute(values)


def render_critic_prompt(
    pack: TopicPack, query: str, storyboard_json: str, unverified: Sequence[str]
) -> tuple[str, str]:
    system, user = _template("critic")
    values = {
        "query": _escape_query(query),
        "facts": _facts(pack),
        "storyboard": storyboard_json,
        "unverified": "\n".join(f"- {u}" for u in unverified) or "- none",
    }
    return system.substitute(values), user.substitute(values)
