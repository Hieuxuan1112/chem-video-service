import pytest

from app.config import Settings
from app.pipeline.storyboard import (
    ComparisonTable,
    ElectronSharing,
    ElectronTransfer,
    KeyPoints,
    LogRatio,
    PhScale,
    TableRow,
    TitleCard,
)
from app.render.plan import (
    Beat,
    build_render_plan,
    compute_beats,
    frame_boundaries,
    resolve_visual,
)
from app.topics.chemistry_data import (
    IONIC_COMPOUNDS,
    MOLECULES,
    SUBSTANCE_PH,
    IonicCompound,
    Molecule,
    SubstanceName,
)
from app.topics.registry import TOPICS_ROOT, TopicRegistry

SETTINGS = Settings(_env_file=None)  # type: ignore[call-arg]
REGISTRY = TopicRegistry.load(TOPICS_ROOT)


def test_frame_boundaries_no_drift() -> None:
    durations = [10.017] * 7
    bounds = frame_boundaries(durations, 30)
    assert bounds[0][0] == 0
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(6))  # contiguous
    assert bounds[-1][1] == round(sum(durations) * 30)
    naive = sum(round(d * 30) for d in durations)
    assert naive != bounds[-1][1]  # per-scene rounding would drift by frames
    assert frame_boundaries([], 30) == []


def test_last_scene_uses_final_tail() -> None:
    sb = REGISTRY.curated_storyboard("ph_scale")
    clips = [4.0 + i for i in range(len(sb.scenes))]
    plan = build_render_plan(
        job_id="job1", question="How does pH work?", storyboard=sb, clip_durations_s=clips,
        settings=SETTINGS,
    )  # fmt: skip
    s = SETTINGS
    for i, scene in enumerate(plan.scenes):
        tail = s.final_tail_s if i == len(clips) - 1 else s.tail_s
        assert scene.duration_s == pytest.approx(s.lead_in_s + clips[i] + tail)
        assert (scene.lead_in_s, scene.clip_duration_s) == (s.lead_in_s, clips[i])
        assert scene.frames == scene.end_frame - scene.start_frame
        assert scene.on_screen_title == sb.scenes[i].on_screen_title
        assert scene.narration == sb.scenes[i].narration
    assert (
        plan.total_frames
        == plan.scenes[-1].end_frame
        == round(sum(sc.duration_s for sc in plan.scenes) * s.fps)
    )
    assert (plan.width, plan.height, plan.fps, plan.theme) == (1280, 720, 30, "dark-v1")
    with pytest.raises(ValueError):
        build_render_plan(
            job_id="j", question="q", storyboard=sb, clip_durations_s=clips[:-1], settings=s
        )


def test_resolved_ph_values_come_from_chemistry_data() -> None:
    visual = PhScale(
        type="ph_scale",
        markers=[SubstanceName.LEMON_JUICE, SubstanceName.PURE_WATER, SubstanceName.BLEACH],
        highlight="acidic",
    )
    resolved = resolve_visual(visual, question="q", title="t")
    assert resolved.type == "ph_scale"
    assert [(m.name, m.display, m.ph) for m in resolved.markers] == [
        (s.name.value, s.display, s.ph) for s in (SUBSTANCE_PH[n] for n in visual.markers)
    ]
    assert [m.ph_class for m in resolved.markers] == ["acidic", "neutral", "basic"]
    assert resolved.highlight == "acidic"


def test_sharing_pairs_from_molecule_table() -> None:
    for molecule in Molecule:
        resolved = resolve_visual(
            ElectronSharing(type="electron_sharing", molecule=molecule), question="q", title="t"
        )
        data = MOLECULES[molecule]
        assert resolved.shared_pairs == data.shared_pairs
        assert (resolved.formula, resolved.display, resolved.name) == (
            molecule.value, data.display, data.name
        )  # fmt: skip
        assert [a.symbol for a in resolved.atoms] == [str(a) for a in data.atoms]
    n2 = resolve_visual(
        ElectronSharing(type="electron_sharing", molecule=Molecule.N2), question="q", title="t"
    )
    assert n2.shared_pairs == 3
    assert n2.atoms[0].shells == [2, 5] and n2.atoms[0].valence == 5


def test_transfer_electrons_from_compound_table() -> None:
    mgo = resolve_visual(
        ElectronTransfer(type="electron_transfer", compound=IonicCompound.MgO),
        question="q",
        title="t",
    )
    data = IONIC_COMPOUNDS[IonicCompound.MgO]
    assert mgo.electrons == data.electrons == 2
    assert (mgo.donor.symbol, mgo.acceptor.symbol) == ("Mg", "O")
    assert (mgo.donor_ion, mgo.acceptor_ion) == (data.donor_ion, data.acceptor_ion)
    assert mgo.donor.category == "metal" and mgo.acceptor.category == "nonmetal"
    assert mgo.donor.shells == [2, 8, 2]


def test_log_ratio_is_power_of_ten() -> None:
    one = resolve_visual(LogRatio(type="log_ratio", from_ph=4, to_ph=3), question="q", title="t")
    two = resolve_visual(LogRatio(type="log_ratio", from_ph=5, to_ph=7), question="q", title="t")
    assert (one.from_ph, one.to_ph, one.ratio) == (4, 3, 10)
    assert two.ratio == 100


def test_text_visuals_carry_question_and_title() -> None:
    card = resolve_visual(TitleCard(type="title_card", subtitle="Sub"), question="Q?", title="T")
    assert (card.question, card.subtitle, card.title) == ("Q?", "Sub", "T")
    points = resolve_visual(
        KeyPoints(type="key_points", points=["a", "b"]), question="Q?", title="T"
    )
    assert (points.points, points.question) == (["a", "b"], "Q?")
    rows = [TableRow(aspect="pH", left="low", right="high")] * 2
    visual = ComparisonTable(
        type="comparison_table", left_header="Acids", right_header="Bases", rows=rows
    )
    table = resolve_visual(visual, question="q", title="t")
    assert table.rows == rows


def test_beats_follow_first_mention_order_and_bounds() -> None:
    narration = "Lemon juice is acidic. Pure water sits in the middle. Bleach is strongly basic."
    targets = [
        ("marker:lemon_juice", ["lemon juice"]),
        ("marker:pure_water", ["pure water"]),
        ("marker:bleach", ["bleach"]),
    ]
    beats = compute_beats(narration, targets, lead_in_s=0.3, clip_s=10.0)
    assert [b.target for b in beats] == [t for t, _ in targets]
    offsets = [narration.lower().index(p[0]) for _, p in targets]
    expected = [0.3 + o / len(narration) * 10.0 for o in offsets]
    assert [b.at_s for b in beats] == pytest.approx(expected)
    # clamped to [lead_in, lead_in + clip - 0.5] and never decreasing
    late = compute_beats("x " * 40 + "bleach", [("b", ["bleach"])], lead_in_s=0.3, clip_s=4.0)
    assert [(b.target, b.at_s) for b in late] == [("b", pytest.approx(0.3 + 4.0 - 0.5))]
    assert isinstance(late[0], Beat)
    reversed_mentions = compute_beats(
        "Bleach first, then lemon juice.",
        [("marker:lemon_juice", ["lemon juice"]), ("marker:bleach", ["bleach"])],
        lead_in_s=0.3,
        clip_s=10.0,
    )
    assert reversed_mentions[0].at_s <= reversed_mentions[1].at_s
    # whole words only, case-insensitive
    word = compute_beats("Hydrogen and HYDROGEN", [("h", ["hydrogen"])], lead_in_s=0.0, clip_s=2.0)
    assert word[0].at_s == pytest.approx(0.0)


def test_unmentioned_targets_spread_evenly() -> None:
    targets = [("row:0", ["alpha"]), ("row:1", ["beta"]), ("row:2", ["gamma"])]
    beats = compute_beats("Nothing relevant is said here.", targets, lead_in_s=0.3, clip_s=8.0)
    times = [b.at_s for b in beats]
    upper = 0.3 + 8.0 - 0.5
    assert all(0.3 <= t <= upper for t in times)
    gaps = [times[0] - 0.3, times[1] - times[0], times[2] - times[1], upper - times[2]]
    assert gaps == pytest.approx([gaps[0]] * 4)
    mixed = compute_beats(
        "We start with gamma.", targets, lead_in_s=0.0, clip_s=10.0
    )  # gamma mentioned at 14/19 of the clip; alpha and beta fill the time before it
    assert mixed[0].at_s < mixed[1].at_s < mixed[2].at_s


def test_question_is_raw_text_not_markup() -> None:
    question = '<b>pH</b> & <span foreground="red">acids</span>?'
    sb = REGISTRY.curated_storyboard("ph_scale")
    plan = build_render_plan(
        job_id="job1", question=question, storyboard=sb,
        clip_durations_s=[5.0] * len(sb.scenes), settings=SETTINGS,
    )  # fmt: skip
    assert plan.question == question
    assert plan.scenes[0].visual.type == "title_card"
    assert plan.scenes[0].visual.question == question  # type: ignore[union-attr]
    assert plan.scenes[-1].visual.question == question  # type: ignore[union-attr]
    restored = type(plan).model_validate_json(plan.model_dump_json())
    assert restored == plan
