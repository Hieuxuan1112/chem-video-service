import sys
import time
import wave
from pathlib import Path

import numpy as np
import psutil
import pytest

from app.config import Settings
from app.pipeline.audio_checks import (
    check_clip,
    internal_silences,
    pcm_to_float,
    rms_dbfs,
    trim_silence,
    write_wav,
)
from app.providers.tts.protocol import SynthesisResult
from app.render.proc import kill_tree, run_subprocess

SETTINGS = Settings(_env_file=None)  # type: ignore[call-arg]
SR = 24_000
WORDS = 10  # expected duration 10 / 2.5 = 4 s


def _tone(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(round(seconds * SR)) / SR
    return amplitude * np.sin(2 * np.pi * 220.0 * t)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * SR))


def _result(samples: np.ndarray, sr: int = SR, channels: int = 1) -> SynthesisResult:
    pcm = (samples * 32767).astype("<i2").tobytes()
    return SynthesisResult(pcm, sr, channels, len(pcm) / (2 * channels * sr))


def _names(report: object) -> dict[str, bool]:
    return {c.name: c.passed for c in report.checks}  # type: ignore[attr-defined]


def test_a1_empty_fails(tmp_path: Path) -> None:
    report = check_clip(
        SynthesisResult(b"", SR, 1, 0.0), text_words=WORDS, settings=SETTINGS, workdir=tmp_path
    )
    assert not report.passed
    assert _names(report) == {"A1": False}
    short = check_clip(_result(_tone(0.3)), text_words=1, settings=SETTINGS, workdir=tmp_path)
    assert _names(short) == {"A1": False}


def test_a1_wrong_channels_fails(tmp_path: Path) -> None:
    stereo = _result(np.repeat(_tone(4.0), 2), channels=2)
    assert _names(check_clip(stereo, text_words=WORDS, settings=SETTINGS, workdir=tmp_path)) == {
        "A1": False
    }
    wrong_rate = _result(_tone(4.0), sr=16_000)
    report = check_clip(wrong_rate, text_words=WORDS, settings=SETTINGS, workdir=tmp_path)
    assert _names(report) == {"A1": False}
    assert "sample rate" in report.checks[0].message


def test_a2_truncated_ratio_fails(tmp_path: Path) -> None:
    report = check_clip(_result(_tone(1.0)), text_words=WORDS, settings=SETTINGS, workdir=tmp_path)
    assert _names(report) == {"A1": True, "A2": False, "A3": True}  # A4 skipped: already invalid
    a2 = report.checks[1]
    assert a2.value == pytest.approx(0.25, abs=0.01)
    assert a2.threshold == "0.6-1.6"


def test_a2_too_long_fails(tmp_path: Path) -> None:
    report = check_clip(_result(_tone(7.0)), text_words=WORDS, settings=SETTINGS, workdir=tmp_path)
    assert _names(report)["A2"] is False
    assert report.checks[1].value == pytest.approx(1.75, abs=0.01)


def test_a2_measured_after_trimming(tmp_path: Path) -> None:
    padded = np.concatenate([_silence(2.0), _tone(4.0), _silence(2.0)])  # 8 s raw, 4 s of speech
    report = check_clip(_result(padded), text_words=WORDS, settings=SETTINGS, workdir=tmp_path)
    assert report.checks[1].value == pytest.approx(4.2 / 4.0, abs=0.02)


def test_a3_silence_fails(tmp_path: Path) -> None:
    report = check_clip(
        _result(_silence(4.0)), text_words=WORDS, settings=SETTINGS, workdir=tmp_path
    )
    assert _names(report)["A3"] is False
    quiet = check_clip(
        _result(_tone(4.0, amplitude=0.02)), text_words=WORDS, settings=SETTINGS, workdir=tmp_path
    )  # -37 dBFS RMS: kept by the -40 dBFS trim, below the -35 dBFS A3 floor
    assert _names(quiet) == {"A1": True, "A2": True, "A3": False}


def test_leading_trailing_silence_trimmed_to_keep() -> None:
    samples = np.concatenate([_silence(1.0), _tone(2.0), _silence(1.5)])
    trimmed = trim_silence(samples, SR, threshold_dbfs=-40.0, keep_s=0.1)
    assert len(trimmed) / SR == pytest.approx(2.2, abs=0.02)
    assert not np.any(trimmed[: round(0.08 * SR)])  # kept lead-in is silence
    assert np.any(trimmed[round(0.12 * SR) : round(0.2 * SR)])
    untouched = _tone(1.0)
    assert len(trim_silence(untouched, SR, threshold_dbfs=-40.0, keep_s=0.1)) == len(untouched)
    assert len(trim_silence(_silence(1.0), SR, threshold_dbfs=-40.0, keep_s=0.1)) == 0


def test_rms_and_pcm_helpers() -> None:
    assert rms_dbfs(_tone(1.0, amplitude=1.0)) == pytest.approx(-3.01, abs=0.05)
    assert rms_dbfs(np.zeros(10)) == float("-inf")
    assert rms_dbfs(np.zeros(0)) == float("-inf")
    pcm = np.array([0, 16384, -32768], dtype="<i2").tobytes()
    assert pcm_to_float(pcm).tolist() == [0.0, 0.5, -1.0]


@pytest.mark.integration
def test_a4_internal_gap_detected(tmp_path: Path) -> None:
    gap = np.concatenate([_tone(2.0), _silence(2.0), _tone(2.0)])
    report = check_clip(_result(gap), text_words=16, settings=SETTINGS, workdir=tmp_path)
    assert _names(report) == {"A1": True, "A2": True, "A3": True, "A4": False}
    assert report.checks[3].value == pytest.approx(2.0, abs=0.1)
    wav = tmp_path / "gap.wav"
    write_wav(wav, _result(gap).pcm, SR)
    [(start, end)] = internal_silences(wav, threshold_dbfs=-40.0, min_s=1.5)
    assert (start, end) == (pytest.approx(2.0, abs=0.1), pytest.approx(4.0, abs=0.1))


@pytest.mark.integration
def test_good_clip_passes(tmp_path: Path) -> None:
    samples = np.concatenate([_silence(0.5), _tone(1.5), _silence(1.0), _tone(1.5), _silence(0.5)])
    report = check_clip(_result(samples), text_words=WORDS, settings=SETTINGS, workdir=tmp_path)
    assert report.passed, report.checks
    assert _names(report) == {"A1": True, "A2": True, "A3": True, "A4": True}
    assert report.sample_rate == SR
    assert report.duration_s == pytest.approx(4.2, abs=0.02)
    assert len(report.trimmed_pcm) == round(report.duration_s * SR) * 2
    with wave.open(str(tmp_path / "check.wav")) as w:  # the file A4 inspected
        assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (1, SR, 2)


def test_run_subprocess_captures_output_and_input() -> None:
    code = (
        "import sys; data = sys.stdin.buffer.read();"
        " sys.stdout.buffer.write(data[::-1]); print('warn', file=sys.stderr)"
    )
    result = run_subprocess([sys.executable, "-c", code], timeout_s=30, input_bytes=b"abc")
    assert (result.returncode, result.stdout, result.timed_out) == (0, b"cba", False)
    assert "warn" in result.stderr_tail


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.integration
def test_kill_tree_kills_child() -> None:
    child = "import time; time.sleep(60)"
    parent = (
        "import subprocess, sys, time;"
        f" p = subprocess.Popen([sys.executable, '-c', {child!r}],"
        " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL);"
        " print(p.pid, flush=True); time.sleep(60)"
    )
    started = time.monotonic()
    result = run_subprocess([sys.executable, "-c", parent], timeout_s=3.0)
    child_pid = int(result.stdout.split()[0])
    try:
        assert result.timed_out is True
        assert time.monotonic() - started < 15
        deadline = time.monotonic() + 5
        while _alive(child_pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _alive(child_pid)
    finally:
        kill_tree(child_pid)  # cleanup if the assertion failed; a no-op when already gone
