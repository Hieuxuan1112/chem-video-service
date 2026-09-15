import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.config import Settings
from app.domain.errors import AppError, ErrorCode
from app.pipeline.audio_checks import write_wav
from app.pipeline.voicing import VoicedClip
from app.providers.tts.fake import FakeTTS
from app.render import assembler as assembler_module
from app.render.assembler import SAMPLE_RATE, AssemblyResult, MediaAssembler
from app.render.fake import FakeRenderer
from app.render.ffmpeg import FFmpegError, count_frames, ffprobe
from app.render.layout import FontSet
from app.render.plan import RenderPlan, build_render_plan
from app.topics.registry import TOPICS_ROOT, TopicRegistry

pytestmark = pytest.mark.integration

SETTINGS = Settings(_env_file=None, width=320, height=180, fps=15)  # type: ignore[call-arg]


def _inputs(
    tmp_path: Path, script: dict[int, str] | None = None
) -> tuple[RenderPlan, list[VoicedClip], list[Path]]:
    storyboard = TopicRegistry.load(TOPICS_ROOT).curated_storyboard("ph_scale")
    tts = FakeTTS()
    clips: list[VoicedClip] = []
    for i, scene in enumerate(storyboard.scenes):
        words = " ".join(scene.narration.split()[: 4 + i])  # short clips keep the test fast
        result = tts.synthesize(words)
        path = tmp_path / "clips" / f"clip-{i:02d}.wav"
        write_wav(path, result.pcm, result.sample_rate)
        clips.append(VoicedClip(i, path, result.duration_s, "fake_tts", "fake", False, [], None))
    plan = build_render_plan(
        job_id="job-asm",
        question="What is {pH}?",
        storyboard=storyboard,
        clip_durations_s=[c.duration_s for c in clips],
        settings=SETTINGS,
    )
    render = FakeRenderer(script=script).render(plan, tmp_path / "scenes")
    videos = [s.path for s in render.scenes]
    assert all(v is not None for v in videos)
    return plan, clips, [v for v in videos if v is not None]


def _assemble(tmp_path: Path, fonts: FontSet, **kwargs: Any) -> tuple[RenderPlan, AssemblyResult]:
    plan, clips, videos = _inputs(tmp_path, kwargs.pop("script", None))
    result = MediaAssembler(SETTINGS, fonts).assemble(
        plan=plan, clips=clips, scene_videos=videos, workdir=tmp_path / "work", **kwargs
    )
    return plan, result


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SAMPLE_RATE, 1, 2)
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768


def test_assemble_produces_av_with_matching_durations(tmp_path: Path, fonts: FontSet) -> None:
    plan, result = _assemble(tmp_path, fonts)

    for path in (result.video, result.narration_wav, result.captions_vtt, result.captions_ass,
                 result.thumbnail, result.contact_sheet):  # fmt: skip
        assert path.is_file() and path.stat().st_size > 0, path
    expected_s = plan.total_frames / plan.fps
    probe = ffprobe(result.video)
    streams = {s["codec_type"]: s for s in probe["streams"]}
    assert streams["video"]["codec_name"] == "h264" and streams["video"]["pix_fmt"] == "yuv420p"
    assert streams["audio"]["codec_name"] == "aac" and streams["audio"]["sample_rate"] == "48000"
    assert abs(float(streams["video"]["duration"]) - expected_s) <= 0.1
    assert abs(float(streams["audio"]["duration"]) - expected_s) <= 0.1
    assert count_frames(result.video) == plan.total_frames
    assert result.frame_fixes == []
    assert len(_read_wav(result.narration_wav)) == round(expected_s * SAMPLE_RATE)
    assert result.captions_vtt.read_text(encoding="utf-8").startswith("WEBVTT")
    head = result.video.read_bytes()[:4096]
    assert head.find(b"moov") != -1 and (
        head.find(b"mdat") == -1 or head.find(b"moov") < head.find(b"mdat")
    )


def test_audio_offsets_match_plan(tmp_path: Path, fonts: FontSet) -> None:
    plan, result = _assemble(tmp_path, fonts)
    audio = np.abs(_read_wav(result.narration_wav))
    window = SAMPLE_RATE // 200  # 5 ms envelope
    envelope = np.convolve(audio, np.ones(window) / window, mode="same")

    for scene in plan.scenes:
        expected = round((scene.start_frame / plan.fps + scene.lead_in_s) * SAMPLE_RATE)
        lo = expected - SAMPLE_RATE // 5
        onset = lo + int(np.argmax(envelope[lo:] > 0.02))
        assert abs(onset - expected) <= SAMPLE_RATE * 0.015, (scene.index, onset, expected)
        before = envelope[expected - SAMPLE_RATE // 10 : expected - window]
        assert before.max() < 0.02  # silence right before each clip


def test_off_by_one_frame_fixed(tmp_path: Path, fonts: FontSet) -> None:
    plan, result = _assemble(tmp_path, fonts, script={1: "off_by_one"})

    assert len(result.frame_fixes) == 1 and "scene 1" in result.frame_fixes[0]
    assert count_frames(result.video) == plan.total_frames


def test_transient_failure_retried_once(
    tmp_path: Path, fonts: FontSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = assembler_module.run_ffmpeg
    calls: list[Sequence[str]] = []

    def flaky(args: Sequence[str], **kwargs: Any) -> Any:
        calls.append(args)
        if "libx264" in args and "aac" in args and sum("aac" in c for c in calls) == 1:
            raise FFmpegError(
                "ffmpeg exited with 1", stderr_tail="Conversion failed!", transient=True
            )
        return real(args, **kwargs)

    monkeypatch.setattr(assembler_module, "run_ffmpeg", flaky)
    plan, result = _assemble(tmp_path, fonts)

    assert sum("aac" in c for c in calls) == 2  # the final encode ran twice
    assert count_frames(result.video) == plan.total_frames


def test_transient_failure_twice_is_retryable_assembly_error(
    tmp_path: Path, fonts: FontSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = assembler_module.run_ffmpeg

    def broken(args: Sequence[str], **kwargs: Any) -> Any:
        if "aac" in args:
            raise FFmpegError(
                "ffmpeg exited with 1", stderr_tail="Conversion failed!", transient=True
            )
        return real(args, **kwargs)

    monkeypatch.setattr(assembler_module, "run_ffmpeg", broken)
    with pytest.raises(AppError) as info:
        _assemble(tmp_path, fonts)

    assert (info.value.code, info.value.cause, info.value.retryable) == (
        ErrorCode.ASSEMBLY_FAILED, "ffmpeg_transient", True,
    )  # fmt: skip


def test_missing_clip_invalid_input_non_retryable(tmp_path: Path, fonts: FontSet) -> None:
    plan, clips, videos = _inputs(tmp_path)
    clips[2].wav_path.unlink()

    with pytest.raises(AppError) as info:
        MediaAssembler(SETTINGS, fonts).assemble(
            plan=plan, clips=clips, scene_videos=videos, workdir=tmp_path / "work"
        )

    assert (info.value.code, info.value.cause, info.value.retryable) == (
        ErrorCode.ASSEMBLY_FAILED, "invalid_input", False,
    )  # fmt: skip
