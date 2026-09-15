"""A1-A4 clip validation (spec §7.3): the media floor every narration clip must clear."""

import math
import re
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Settings
from app.providers.tts.protocol import SynthesisResult
from app.render.proc import run_subprocess

EXPECTED_SAMPLE_RATE = 24_000
MIN_DURATION_S = 0.5
_FRAME_S = 0.02
_FFMPEG_TIMEOUT_S = 60.0
_START = re.compile(r"silence_start: (-?[\d.]+)")
_END = re.compile(r"silence_end: (-?[\d.]+)")


@dataclass
class ClipCheck:
    name: str
    passed: bool
    value: float | None
    threshold: str
    message: str


@dataclass
class ClipReport:
    checks: list[ClipCheck]
    trimmed_pcm: bytes
    sample_rate: int
    duration_s: float

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0


def rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(samples**2)))
    return 20 * math.log10(rms) if rms > 0 else float("-inf")


def trim_silence(
    samples: np.ndarray, sr: int, *, threshold_dbfs: float, keep_s: float
) -> np.ndarray:
    """Cut leading and trailing 20 ms frames below the threshold, keeping keep_s on each side."""
    frame = max(1, round(_FRAME_S * sr))
    loud = [
        i
        for i in range(math.ceil(samples.size / frame))
        if rms_dbfs(samples[i * frame : (i + 1) * frame]) > threshold_dbfs
    ]
    if not loud:
        return samples[:0]
    keep = round(keep_s * sr)
    start = max(0, loud[0] * frame - keep)
    end = min(samples.size, (loud[-1] + 1) * frame + keep)
    return samples[start:end]


def write_wav(path: Path, pcm: bytes, sr: int, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def internal_silences(
    pcm_wav_path: Path, *, threshold_dbfs: float, min_s: float
) -> list[tuple[float, float]]:
    args = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(pcm_wav_path),
        "-af", f"silencedetect=noise={threshold_dbfs}dB:d={min_s}", "-f", "null", "-",
    ]  # fmt: skip
    result = run_subprocess(args, timeout_s=_FFMPEG_TIMEOUT_S)
    if result.returncode != 0 or result.timed_out:
        raise RuntimeError(f"ffmpeg silencedetect failed: {result.stderr_tail}")
    starts = [float(m) for m in _START.findall(result.stderr_tail)]
    ends = [float(m) for m in _END.findall(result.stderr_tail)]
    with wave.open(str(pcm_wav_path)) as w:
        duration = w.getnframes() / w.getframerate()
    return [(s, ends[i] if i < len(ends) else duration) for i, s in enumerate(starts)]


def check_clip(
    result: SynthesisResult, *, text_words: int, settings: Settings, workdir: Path
) -> ClipReport:
    """A1, then A3 trimming, A2 on the trimmed clip, A3 RMS, and A4. Cheapest first: A4 runs
    ffmpeg only when A1-A3 already pass."""
    s = settings
    sr = result.sample_rate
    problems = []
    if len(result.pcm) % 2:
        problems.append("PCM byte length is odd, not s16le")
    if result.channels != 1:
        problems.append(f"{result.channels} channels, expected mono")
    if sr != EXPECTED_SAMPLE_RATE:
        problems.append(f"sample rate {sr} Hz, expected {EXPECTED_SAMPLE_RATE} Hz")
    raw_s = len(result.pcm) / (2 * max(result.channels, 1) * sr) if sr > 0 else 0.0
    if raw_s < MIN_DURATION_S:
        problems.append(f"{raw_s:.2f} s of audio, need at least {MIN_DURATION_S} s")
    a1 = ClipCheck(
        "A1",
        not problems,
        raw_s,
        f">={MIN_DURATION_S}s mono {EXPECTED_SAMPLE_RATE}Hz",
        "; ".join(problems) or "ok",
    )
    if not a1.passed:
        return ClipReport([a1], b"", sr, 0.0)

    samples = trim_silence(
        pcm_to_float(result.pcm), sr, threshold_dbfs=s.silence_threshold_dbfs, keep_s=s.trim_keep_s
    )
    duration = samples.size / sr
    trimmed_pcm = np.round(samples * 32768.0).clip(-32768, 32767).astype("<i2").tobytes()

    expected = text_words / s.words_per_second
    ratio = duration / expected if expected > 0 else float("inf")
    a2 = ClipCheck(
        "A2",
        s.a2_min_ratio <= ratio <= s.a2_max_ratio,
        ratio,
        f"{s.a2_min_ratio}-{s.a2_max_ratio}",
        f"duration {duration:.2f} s vs expected {expected:.2f} s",
    )
    level = rms_dbfs(samples)
    a3 = ClipCheck(
        "A3",
        level > s.a3_min_rms_dbfs,
        level if math.isfinite(level) else None,
        f">{s.a3_min_rms_dbfs}dBFS",
        f"RMS {level:.1f} dBFS after trimming",
    )
    checks = [a1, a2, a3]
    if a2.passed and a3.passed:
        wav = workdir / "check.wav"
        write_wav(wav, trimmed_pcm, sr)
        gaps = internal_silences(
            wav, threshold_dbfs=s.silence_threshold_dbfs, min_s=s.a4_max_internal_silence_s
        )
        longest = max((end - start for start, end in gaps), default=0.0)
        checks.append(
            ClipCheck(
                "A4",
                not gaps,
                longest,
                f"<{s.a4_max_internal_silence_s}s",
                f"{len(gaps)} internal silence(s), longest {longest:.2f} s",
            )
        )
    return ClipReport(checks, trimmed_pcm, sr, duration)
