import itertools
import re

import pytest

from app.config import Settings
from app.render.captions import (
    MAX_LINE_CHARS,
    Cue,
    build_cues,
    escape_ass,
    split_sentences,
    to_ass,
    to_vtt,
)
from app.render.layout import LAYOUT, FontSet, measure_wrap
from app.render.plan import RenderPlan, build_render_plan
from app.topics.registry import TOPICS_ROOT, TopicRegistry

SETTINGS = Settings(_env_file=None)  # type: ignore[call-arg]
LONG = (
    "When sodium meets chlorine, the sodium atom gives away its single outer electron, "
    "the chlorine atom takes it to complete its outer shell, and the two ions that result "
    "attract each other strongly because they carry opposite charges"
)


def _plan(narrations: list[str] | None = None) -> RenderPlan:
    storyboard = TopicRegistry.load(TOPICS_ROOT).curated_storyboard("ionic_vs_covalent")
    if narrations is not None:
        scenes = [
            s.model_copy(update={"narration": n})
            for s, n in zip(storyboard.scenes, narrations, strict=False)
        ]
        storyboard = storyboard.model_copy(update={"scenes": scenes})
    clips = [7.0 + 0.37 * i for i in range(len(storyboard.scenes))]
    return build_render_plan(
        job_id="job", question="q", storyboard=storyboard, clip_durations_s=clips, settings=SETTINGS
    )


def test_split_sentences_keeps_punctuation() -> None:
    assert split_sentences("pH 7 is neutral. Is it? Yes!  Done") == [
        "pH 7 is neutral.",
        "Is it?",
        "Yes!",
        "Done",
    ]


def test_cues_monotonic_and_within_scene_windows(fonts: FontSet) -> None:
    plan = _plan()
    cues = build_cues(plan, fonts)

    assert cues
    for a, b in itertools.pairwise(cues):
        assert a.start_s < a.end_s <= b.start_s + 1e-9
    for scene in plan.scenes:
        start = scene.start_frame / plan.fps + scene.lead_in_s
        end = start + scene.clip_duration_s
        inside = [c for c in cues if start - 1e-9 <= c.start_s < end]
        assert inside, scene.index
        assert inside[0].start_s == pytest.approx(start)
        assert inside[-1].end_s == pytest.approx(end)
        assert all(c.end_s <= end + 1e-9 for c in inside)
        text = " ".join(" ".join(c.lines) for c in inside)
        assert text.split() == scene.narration.split()  # every word, in order, once


def test_line_limits(fonts: FontSet) -> None:
    cues = build_cues(_plan([LONG] * 6), fonts)

    for cue in cues:
        assert 1 <= len(cue.lines) <= 2
        assert all(len(line) <= MAX_LINE_CHARS for line in cue.lines)
        assert measure_wrap(" ".join(cue.lines), LAYOUT["caption"], fonts).fits


def test_long_sentence_split(fonts: FontSet) -> None:
    plan = _plan([LONG] * 6)
    first = plan.scenes[0]
    cues = [
        c
        for c in build_cues(plan, fonts)
        if c.end_s <= first.lead_in_s + first.clip_duration_s + 1e-9
    ]

    assert len(cues) >= 3
    assert " ".join(" ".join(c.lines) for c in cues) == LONG
    assert cues[0].lines[-1].endswith(",")  # split at a comma when one fits
    durations = [c.end_s - c.start_s for c in cues]
    chars = [len(" ".join(c.lines)) for c in cues]
    ratios = [d / n for d, n in zip(durations, chars, strict=True)]
    assert max(ratios) == pytest.approx(min(ratios), rel=0.15)  # time follows character count


def test_vtt_format() -> None:
    vtt = to_vtt([Cue(0.0, 1.5, ["Hello <b>world</b> & more"]), Cue(3661.25, 3662.0, ["a", "b"])])

    assert vtt.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:01.500\nHello &lt;b&gt;world&lt;/b&gt; &amp; more\n" in vtt
    assert "01:01:01.250 --> 01:01:02.000\na\nb\n" in vtt
    assert re.findall(r"\d\d:\d\d:\d\d\.\d{3} --> \d\d:\d\d:\d\d\.\d{3}", vtt).__len__() == 2


def test_ass_escaping_of_override_braces() -> None:
    assert escape_ass("{\\b1}bold\\N") == "\\{\\\\b1\\}bold\\\\N"
    assert escape_ass("two\nlines") == "two\\Nlines"

    ass = to_ass([Cue(1.0, 2.25, ["{\\pos(0,0)}hijack", "line two"])], width=1280, height=720)
    assert "[Script Info]" in ass and "PlayResX: 1280" in ass and "PlayResY: 720" in ass
    assert "Style: Default,Inter," in ass
    [event] = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert event.startswith("Dialogue: 0,0:00:01.00,0:00:02.25,Default,,0,0,0,,")
    assert event.endswith("\\{\\\\pos(0,0)\\}hijack\\Nline two")
    assert "{" not in event.replace("\\{", "")
