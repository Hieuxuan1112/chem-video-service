"""G2a structural and content rules, G2b chemistry consistency (spec §5.5)."""

import re

from app.config import Settings
from app.pipeline.gates.types import GateResult, Violation
from app.pipeline.storyboard import (
    AtomShells,
    ComparisonTable,
    ElectronSharing,
    ElectronTransfer,
    KeyPoints,
    LogRatio,
    Scene,
    Storyboard,
    TitleCard,
    word_count,
)
from app.topics.chemistry_data import ELEMENTS, IONIC_COMPOUNDS
from app.topics.pack import TopicPack
from app.topics.resolver import contains_phrase, normalize

GATE = "G2_rules"


def estimated_duration_s(sb: Storyboard, settings: Settings) -> float:
    total = 0.0
    last = len(sb.scenes) - 1
    for i, scene in enumerate(sb.scenes):
        tail = settings.final_tail_s if i == last else settings.tail_s
        total += word_count(scene.narration) / settings.words_per_second + settings.lead_in_s + tail
    return total


def on_screen_texts(scene: Scene) -> list[str]:
    """Every LLM-provided string shown on screen for a scene."""
    v = scene.visual
    texts = [scene.on_screen_title]
    if isinstance(v, TitleCard):
        texts.append(v.subtitle)
    elif isinstance(v, KeyPoints):
        texts.extend(v.points)
    elif isinstance(v, ComparisonTable):
        texts.extend([v.left_header, v.right_header])
        for row in v.rows:
            texts.extend([row.aspect, row.left, row.right])
    return texts


def _structure(sb: Storyboard, pack: TopicPack, settings: Settings) -> list[Violation]:
    out: list[Violation] = []

    def add(code: str, message: str, scene_index: int | None = None) -> None:
        out.append(Violation("G2a", code, message, scene_index))

    if sb.topic_id != pack.id:
        add(
            "topic_mismatch", f"topic_id is '{sb.topic_id}' but this storyboard is for '{pack.id}'."
        )
    word_range = f"{settings.min_words}-{settings.max_words}"
    for i, scene in enumerate(sb.scenes):
        n = word_count(scene.narration)
        if n < settings.min_words:
            add("scene_words_low", f"Scene {i}: narration has {n} words; use {word_range}.", i)
        elif n > settings.max_words:
            add("scene_words_high", f"Scene {i}: narration has {n} words; use {word_range}.", i)
    duration = estimated_duration_s(sb, settings)
    if not settings.min_total_s <= duration <= settings.max_total_s:
        add(
            "duration_out_of_range",
            f"Estimated video length is {duration:.1f} s; it must be "
            f"{settings.min_total_s:.0f}-{settings.max_total_s:.0f} s. Adjust narration length.",
        )
    if sb.scenes[0].visual.type != "title_card":
        add("first_not_title_card", "Scene 0 must use the title_card visual.", 0)
    last = len(sb.scenes) - 1
    if sb.scenes[last].visual.type != "key_points":
        add("last_not_key_points", f"Scene {last} (the last scene) must use key_points.", last)

    used = [s.visual.type for s in sb.scenes]
    for i, visual_type in enumerate(used):
        if visual_type not in pack.allowed_visuals:
            add(
                "visual_not_allowed",
                f"Scene {i}: visual '{visual_type}' is not allowed for this topic; "
                f"choose from {', '.join(pack.allowed_visuals)}.",
                i,
            )
    for required in pack.required_visuals:
        if required not in used:
            add("required_visual_missing", f"No scene uses the required visual '{required}'.")
    seen: dict[str, int] = {}
    for i, scene in enumerate(sb.scenes):
        key = scene.visual.model_dump_json()
        if key in seen:
            add(
                "duplicate_visual",
                f"Scene {i} repeats the exact visual of scene {seen[key]}; show something new.",
                i,
            )
        else:
            seen[key] = i

    full_narration = normalize(" ".join(s.narration for s in sb.scenes))
    for fact in pack.required_facts:
        if not all(any(contains_phrase(full_narration, p) for p in group) for group in fact.detect):
            add("fact_missing", f"The narration must clearly state: {fact.statement}")
    for i, scene in enumerate(sb.scenes):
        text = normalize(" ".join([scene.narration, *on_screen_texts(scene)]))
        for m in pack.misconceptions:
            if re.search(m.pattern, text, re.IGNORECASE):
                add("misconception", f"Scene {i}: incorrect statement. {m.explanation}", i)
    return out


def _chemistry(sb: Storyboard, pack: TopicPack) -> list[Violation]:
    out: list[Violation] = []

    def add(code: str, message: str, scene_index: int) -> None:
        out.append(Violation("G2b", code, message, scene_index))

    for i, scene in enumerate(sb.scenes):
        v = scene.visual
        if isinstance(v, LogRatio) and abs(v.to_ph - v.from_ph) not in (1, 2):
            add(
                "log_ratio_delta",
                f"Scene {i}: log_ratio from_ph {v.from_ph} and to_ph {v.to_ph} "
                "must differ by 1 or 2.",
                i,
            )
        elif isinstance(v, AtomShells):
            for element in v.elements:
                if element not in pack.allowed_elements:
                    add(
                        "element_not_allowed",
                        f"Scene {i}: element {element} is not allowed for this topic.",
                        i,
                    )
        elif isinstance(v, ElectronSharing):
            if v.molecule not in pack.allowed_molecules:
                add(
                    "molecule_not_allowed",
                    f"Scene {i}: molecule {v.molecule} is not allowed for this topic.",
                    i,
                )
        elif isinstance(v, ElectronTransfer):
            if v.compound not in pack.allowed_compounds:
                add(
                    "compound_not_allowed",
                    f"Scene {i}: compound {v.compound} is not allowed for this topic.",
                    i,
                )
            data = IONIC_COMPOUNDS[v.compound]
            if (
                ELEMENTS[data.donor].category != "metal"
                or ELEMENTS[data.acceptor].category != "nonmetal"
            ):
                add(
                    "ionic_roles_invalid",
                    f"Scene {i}: {v.compound} must transfer electrons from a metal to a non-metal.",
                    i,
                )
    return out


def g2_rules(sb: Storyboard, pack: TopicPack, settings: Settings) -> GateResult:
    violations = _structure(sb, pack, settings) + _chemistry(sb, pack)
    return GateResult(
        GATE,
        not violations,
        violations,
        {"estimated_duration_s": round(estimated_duration_s(sb, settings), 2)},
    )
