"""Renderer-agnostic render plan (spec §8.1): exact frame timeline, code-resolved visuals, beats.

Every number drawn on screen is resolved here from chemistry_data; a renderer only draws.
"""

import re
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.config import Settings
from app.pipeline.storyboard import (
    AtomShells,
    ComparisonTable,
    ElectronSharing,
    ElectronTransfer,
    KeyPoints,
    LogRatio,
    PhScale,
    Storyboard,
    TableRow,
    TitleCard,
    Visual,
)
from app.topics.chemistry_data import (
    ELEMENTS,
    IONIC_COMPOUNDS,
    MOLECULES,
    SUBSTANCE_PH,
    ElementSymbol,
    SubstanceName,
)

BEAT_END_MARGIN_S = 0.5


class Beat(BaseModel):
    target: str
    at_s: float  # seconds from scene start


class ResolvedMarker(BaseModel):
    name: str
    display: str
    ph: float
    ph_class: str


class ResolvedTitleCard(BaseModel):
    type: Literal["title_card"]
    question: str
    subtitle: str
    title: str


class ResolvedKeyPoints(BaseModel):
    type: Literal["key_points"]
    points: list[str]
    question: str


class ResolvedPhScale(BaseModel):
    type: Literal["ph_scale"]
    markers: list[ResolvedMarker]
    highlight: str


class ResolvedLogRatio(BaseModel):
    type: Literal["log_ratio"]
    from_ph: int
    to_ph: int
    ratio: int


class ResolvedAtom(BaseModel):
    symbol: str
    name: str
    shells: list[int]
    valence: int
    cpk: str
    category: str


class ResolvedAtomShells(BaseModel):
    type: Literal["atom_shells"]
    atoms: list[ResolvedAtom]


class ResolvedElectronSharing(BaseModel):
    type: Literal["electron_sharing"]
    formula: str
    display: str
    name: str
    atoms: list[ResolvedAtom]
    shared_pairs: int


class ResolvedElectronTransfer(BaseModel):
    type: Literal["electron_transfer"]
    formula: str
    display: str
    name: str
    donor: ResolvedAtom
    acceptor: ResolvedAtom
    electrons: int
    donor_ion: str
    acceptor_ion: str


class ResolvedComparisonTable(BaseModel):
    type: Literal["comparison_table"]
    left_header: str
    right_header: str
    rows: list[TableRow]


ResolvedVisual = Annotated[
    ResolvedTitleCard
    | ResolvedKeyPoints
    | ResolvedPhScale
    | ResolvedLogRatio
    | ResolvedAtomShells
    | ResolvedElectronSharing
    | ResolvedElectronTransfer
    | ResolvedComparisonTable,
    Field(discriminator="type"),
]


class PlannedScene(BaseModel):
    index: int
    start_frame: int
    end_frame: int
    duration_s: float
    lead_in_s: float
    clip_duration_s: float
    on_screen_title: str
    narration: str
    visual: ResolvedVisual
    beats: list[Beat]

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame


class RenderPlan(BaseModel):
    job_id: str
    width: int
    height: int
    fps: int
    theme: str = "dark-v1"
    question: str  # raw learner text; templates must draw it with Text, never MarkupText
    total_frames: int
    scenes: list[PlannedScene]


def frame_boundaries(durations_s: Sequence[float], fps: int) -> list[tuple[int, int]]:
    """Cumulative rounding, so the total never drifts: sum of frames == round(total * fps)."""
    bounds: list[tuple[int, int]] = []
    elapsed = 0.0
    for duration in durations_s:
        start = round(elapsed * fps)
        elapsed += duration
        bounds.append((start, round(elapsed * fps)))
    return bounds


def _atom(symbol: ElementSymbol) -> ResolvedAtom:
    e = ELEMENTS[symbol]
    return ResolvedAtom(
        symbol=e.symbol,
        name=e.name,
        shells=list(e.shells),
        valence=e.valence,
        cpk=e.cpk,
        category=e.category,
    )


def _ph_class(ph: float) -> str:
    return "acidic" if ph < 7 else "neutral" if ph == 7 else "basic"


def resolve_visual(visual: Visual, *, question: str, title: str) -> ResolvedVisual:
    if isinstance(visual, TitleCard):
        return ResolvedTitleCard(
            type="title_card", question=question, subtitle=visual.subtitle, title=title
        )
    if isinstance(visual, KeyPoints):
        return ResolvedKeyPoints(type="key_points", points=list(visual.points), question=question)
    if isinstance(visual, PhScale):
        markers = [
            ResolvedMarker(name=s.name.value, display=s.display, ph=s.ph, ph_class=_ph_class(s.ph))
            for s in (SUBSTANCE_PH[m] for m in visual.markers)
        ]
        return ResolvedPhScale(type="ph_scale", markers=markers, highlight=visual.highlight)
    if isinstance(visual, LogRatio):
        ratio = 10 ** abs(visual.to_ph - visual.from_ph)
        return ResolvedLogRatio(
            type="log_ratio", from_ph=visual.from_ph, to_ph=visual.to_ph, ratio=ratio
        )
    if isinstance(visual, AtomShells):
        return ResolvedAtomShells(type="atom_shells", atoms=[_atom(e) for e in visual.elements])
    if isinstance(visual, ElectronSharing):
        m = MOLECULES[visual.molecule]
        return ResolvedElectronSharing(
            type="electron_sharing",
            formula=m.formula.value,
            display=m.display,
            name=m.name,
            atoms=[_atom(a) for a in m.atoms],
            shared_pairs=m.shared_pairs,
        )
    if isinstance(visual, ElectronTransfer):
        c = IONIC_COMPOUNDS[visual.compound]
        return ResolvedElectronTransfer(
            type="electron_transfer",
            formula=c.formula.value,
            display=c.display,
            name=c.name,
            donor=_atom(c.donor),
            acceptor=_atom(c.acceptor),
            electrons=c.electrons,
            donor_ion=c.donor_ion,
            acceptor_ion=c.acceptor_ion,
        )
    if isinstance(visual, ComparisonTable):
        return ResolvedComparisonTable(
            type="comparison_table",
            left_header=visual.left_header,
            right_header=visual.right_header,
            rows=list(visual.rows),
        )
    raise TypeError(f"unknown visual {type(visual).__name__}")


_NUMBER_WORDS = {10: ["ten times", "tenfold", "10 times"], 100: ["hundred times", "100 times"]}


def beat_targets(v: ResolvedVisual) -> list[tuple[str, list[str]]]:
    """(target id, mention phrases) in the order the template reveals them."""
    if isinstance(v, ResolvedTitleCard):
        return [("question", []), ("subtitle", [])]
    if isinstance(v, ResolvedKeyPoints):
        return [(f"point:{i}", [p]) for i, p in enumerate(v.points)]
    if isinstance(v, ResolvedPhScale):
        targets = [
            (f"marker:{m.name}", [m.display.lower(), *SUBSTANCE_PH[SubstanceName(m.name)].aliases])
            for m in v.markers
        ]
        if v.highlight != "none":
            targets.append(("highlight", [v.highlight]))
        return targets
    if isinstance(v, ResolvedLogRatio):
        return [
            # Narration usually states the idea before the example numbers, so the concept words
            # reveal the grids while the idea is explained.
            ("from", [f"ph of {v.from_ph}", f"ph {v.from_ph}", "logarithmic", "log scale"]),
            ("to", [f"ph of {v.to_ph}", f"ph {v.to_ph}", "each step", "one step", "ph unit"]),
            ("ratio", _NUMBER_WORDS[v.ratio]),
        ]
    if isinstance(v, ResolvedAtomShells):
        return [(f"atom:{a.symbol}", [a.name]) for a in v.atoms]
    if isinstance(v, ResolvedElectronSharing):
        return [
            ("approach", [a.name for a in v.atoms]),
            ("share", ["share", "shared", "sharing", "shares"]),
            ("full_shell", ["full", "stable"]),
        ]
    if isinstance(v, ResolvedElectronTransfer):
        return [
            ("transfer", ["transfer", "transfers", "transferred", "gives", "loses"]),
            ("ions", ["ion", "ions", "charge", "charged"]),
            ("attract", ["attract", "attracts", "attraction"]),
        ]
    return [(f"row:{i}", [row.aspect]) for i, row in enumerate(v.rows)]


def _first_mention(text: str, phrases: Sequence[str]) -> int | None:
    hits = [
        m.start()
        for p in phrases
        if p
        for m in [re.search(rf"(?<!\w){re.escape(p.lower())}(?!\w)", text)]
        if m is not None
    ]
    return min(hits) if hits else None


def compute_beats(
    narration: str,
    targets: Sequence[tuple[str, Sequence[str]]],
    *,
    lead_in_s: float,
    clip_s: float,
) -> list[Beat]:
    """at = lead_in + first-mention offset share of the clip, clamped and made monotonic in target
    order; unmentioned targets are spread evenly between their mentioned neighbours."""
    lower_bound = lead_in_s
    upper_bound = max(lead_in_s, lead_in_s + clip_s - BEAT_END_MARGIN_S)
    text = narration.lower()
    times: list[float | None] = []
    previous = lower_bound
    for _, phrases in targets:
        offset = _first_mention(text, phrases)
        if offset is None:
            times.append(None)
            continue
        at = lead_in_s + offset / max(len(narration), 1) * clip_s
        at = min(max(at, lower_bound, previous), upper_bound)
        times.append(at)
        previous = at

    i = 0
    while i < len(times):
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(times) and times[j] is None:
            j += 1
        left = times[i - 1] if i > 0 else lower_bound
        right = times[j] if j < len(times) else upper_bound
        assert left is not None and right is not None
        gap = j - i
        for k in range(gap):
            times[i + k] = left + (k + 1) * (right - left) / (gap + 1)
        i = j
    beats = []
    for (target, _), when in zip(targets, times, strict=True):
        assert when is not None
        beats.append(Beat(target=target, at_s=when))
    return beats


def build_render_plan(
    *,
    job_id: str,
    question: str,
    storyboard: Storyboard,
    clip_durations_s: Sequence[float],
    settings: Settings,
) -> RenderPlan:
    s = settings
    if len(clip_durations_s) != len(storyboard.scenes):
        raise ValueError(
            f"{len(clip_durations_s)} clip durations for {len(storyboard.scenes)} scenes"
        )
    last = len(storyboard.scenes) - 1
    durations = [
        s.lead_in_s + clip + (s.final_tail_s if i == last else s.tail_s)
        for i, clip in enumerate(clip_durations_s)
    ]
    scenes: list[PlannedScene] = []
    for i, ((start, end), scene) in enumerate(
        zip(frame_boundaries(durations, s.fps), storyboard.scenes, strict=True)
    ):
        visual = resolve_visual(scene.visual, question=question, title=storyboard.title)
        scenes.append(
            PlannedScene(
                index=i,
                start_frame=start,
                end_frame=end,
                duration_s=durations[i],
                lead_in_s=s.lead_in_s,
                clip_duration_s=clip_durations_s[i],
                on_screen_title=scene.on_screen_title,
                narration=scene.narration,
                visual=visual,
                beats=compute_beats(
                    scene.narration,
                    beat_targets(visual),
                    lead_in_s=s.lead_in_s,
                    clip_s=clip_durations_s[i],
                ),
            )
        )
    return RenderPlan(
        job_id=job_id,
        width=s.width,
        height=s.height,
        fps=s.fps,
        question=question,
        total_frames=scenes[-1].end_frame if scenes else 0,
        scenes=scenes,
    )
