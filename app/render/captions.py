"""Sentence-level captions (spec 8.3): cues fit the caption slot, timed by character share.

Scene boundaries are exact (each scene's cues fill its clip window); timing inside a scene is
proportional to characters, so it is approximate.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.render.layout import LAYOUT, FontSet, measure_wrap
from app.render.plan import RenderPlan

MAX_LINES = 2
MAX_LINE_CHARS = 42
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Cue:
    start_s: float
    end_s: float
    lines: list[str]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def _lines_if_fit(text: str, fonts: FontSet) -> list[str] | None:
    wrap = measure_wrap(text, LAYOUT["caption"], fonts)
    if not wrap.fits or len(wrap.lines) > MAX_LINES:
        return None
    if any(len(line) > MAX_LINE_CHARS for line in wrap.lines):
        return None
    return wrap.lines


def _greedy(pieces: list[str], fonts: FontSet) -> list[str] | None:
    """Joins consecutive pieces while they fit; None if a single piece does not fit."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}" if current else piece
        if _lines_if_fit(candidate, fonts) is not None:
            current = candidate
            continue
        if not current:
            return None
        chunks.append(current)
        if _lines_if_fit(piece, fonts) is None:
            return None
        current = piece
    if current:
        chunks.append(current)
    return chunks


def _chunks(sentence: str, fonts: FontSet) -> list[str]:
    if _lines_if_fit(sentence, fonts) is not None:
        return [sentence]
    clauses = [c for c in re.split(r"(?<=,)\s+", sentence) if c]
    by_clause = _greedy(clauses, fonts)
    if by_clause is not None:
        return by_clause
    chunks: list[str] = []
    for clause in clauses:  # a clause too long on its own is split at words
        chunks.extend(_greedy(clause.split(), fonts) or [clause])
    return _greedy(chunks, fonts) or chunks


def build_cues(plan: RenderPlan, fonts: FontSet) -> list[Cue]:
    cues: list[Cue] = []
    for scene in plan.scenes:
        texts = [c for s in split_sentences(scene.narration) for c in _chunks(s, fonts)]
        total_chars = sum(len(t) for t in texts)
        start = scene.start_frame / plan.fps + scene.lead_in_s
        elapsed = 0
        for i, text in enumerate(texts):
            cue_start = start + scene.clip_duration_s * elapsed / total_chars
            elapsed += len(text)
            cue_end = start + scene.clip_duration_s * elapsed / total_chars
            if i == len(texts) - 1:
                cue_end = start + scene.clip_duration_s
            lines = _lines_if_fit(text, fonts) or [text]
            cues.append(Cue(cue_start, cue_end, lines))
    return cues


def _vtt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def _vtt_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_vtt(cues: Sequence[Cue]) -> str:
    blocks = [
        f"{_vtt_time(c.start_s)} --> {_vtt_time(c.end_s)}\n"
        + "\n".join(_vtt_escape(line) for line in c.lines)
        + "\n"
        for c in cues
    ]
    return "WEBVTT\n\n" + "\n".join(blocks)


def escape_ass(text: str) -> str:
    """Model text must never become an ASS override block or a hard line break."""
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")


def _ass_time(seconds: float) -> str:
    cs = round(seconds * 100)
    return f"{cs // 360_000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def to_ass(cues: Sequence[Cue], *, width: int, height: int, font_name: str = "Inter") -> str:
    slot = LAYOUT["caption"]
    scale = height / 720
    font_size = round(slot.font_px * scale)
    margin_v = round((720 - (slot.y + slot.h)) * scale)
    margin_h = round(slot.x * scale)
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment,MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_name},{font_size},&H00FCFAF8,&H00FCFAF8,&H002A170F,&H802A170F,"
        f"0,0,0,0,100,100,0,0,1,{max(1, round(2 * scale))},0,2,{margin_h},{margin_h},{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = [
        f"Dialogue: 0,{_ass_time(c.start_s)},{_ass_time(c.end_s)},Default,,0,0,0,,"
        + "\\N".join(escape_ass(line) for line in c.lines)
        for c in cues
    ]
    return header + "\n".join(events) + "\n"
