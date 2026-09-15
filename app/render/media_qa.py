"""Deterministic media QA (spec 8.5) with ffprobe/ffmpeg analysers.

After voicing everything is deterministic, so a failed rule is a bug (non-retryable); only an
unreadable file or a failed analyser run is an I/O error worth retrying.
"""

import itertools
import math
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config import Settings
from app.domain.errors import AppError, ErrorCode
from app.render.ffmpeg import FFmpegError, ffprobe, run_ffmpeg
from app.render.plan import RenderPlan

IO_ERROR = "io_error"
RULE_FAILED = "deterministic_output_bug"
BLACK_PIC_TH = 0.98
BLACK_PIX_TH = 0.05  # the dark theme background (#0F172A) must not count as black
_ANALYSIS_TIMEOUT_S = 600.0


@dataclass
class QACheck:
    name: str
    passed: bool
    value: object
    threshold: str
    cause: str | None


@dataclass
class QAReport:
    checks: list[QACheck]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": [asdict(c) for c in self.checks]}


def moov_before_mdat(path: Path) -> bool:
    """Walks top-level MP4 boxes: True when the moov box precedes the first mdat box."""
    with path.open("rb") as f:
        while header := f.read(8):
            if len(header) < 8:
                return False
            size = int.from_bytes(header[:4], "big")
            kind = header[4:8]
            if kind == b"moov":
                return True
            if kind == b"mdat":
                return False
            if size == 1:
                size = int.from_bytes(f.read(8), "big")
                f.seek(size - 16, 1)
            elif size == 0 or size < 8:
                return False
            else:
                f.seek(size - 8, 1)
    return False


def average_hash(image_path: Path) -> int:
    """64-bit average hash: 8x8 greyscale, one bit per pixel brighter than the mean."""
    with Image.open(image_path) as image:
        pixels = np.asarray(image.convert("L").resize((8, 8), Image.Resampling.BOX), dtype=float)
    bits = (pixels > pixels.mean()).flatten()
    return sum(1 << i for i, bit in enumerate(bits) if bit)


def _check(name: str, passed: bool, value: object, threshold: str) -> QACheck:
    return QACheck(name, passed, value, threshold, None if passed else RULE_FAILED)


def _number(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"analyser output missing {pattern!r}")
    return float(match.group(1))


def _intervals(
    meta: str, start_key: str, end_key: str, end_of_file: float
) -> list[tuple[float, float]]:
    found: list[tuple[float, float]] = []
    start: float | None = None
    for line in meta.splitlines():
        key, _, value = line.partition("=")
        if key == start_key:
            start = float(value)
        elif key == end_key and start is not None:
            found.append((start, float(value)))
            start = None
    if start is not None:
        found.append((start, end_of_file))
    return found


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


class MediaQA:
    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def check(self, *, video: Path, plan: RenderPlan) -> QAReport:
        checks: list[QACheck] = []
        try:
            probe = ffprobe(video)
            checks += self._container(video, probe, plan)
            duration = float(probe["format"]["duration"])
            checks += self._durations(probe, duration, plan)
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                checks += self._audio(video, work, duration, plan)
                checks += self._video(video, work, duration)
                checks += self._sanity(video, work, plan)
        except (FFmpegError, OSError, KeyError, ValueError) as exc:
            checks.append(QACheck("Q0_readable", False, f"{type(exc).__name__}: {exc}"[:500],
                                  "file and analysers readable", IO_ERROR))  # fmt: skip
        return QAReport(checks)

    def raise_for(self, report: QAReport) -> None:
        if report.passed:
            return
        failed = [c for c in report.checks if not c.passed]
        cause = IO_ERROR if any(c.cause == IO_ERROR for c in failed) else RULE_FAILED
        names = ", ".join(c.name for c in failed)
        raise AppError(
            ErrorCode.QUALITY_CHECK_FAILED,
            cause,
            f"media QA failed: {names}",
            {"failed": [asdict(c) for c in failed]},
        )

    def _analyse(self, video: Path, work: Path, args: list[str]) -> str:
        # Relative metadata file names with cwd=work avoid escaping a Windows drive colon.
        result = run_ffmpeg(["-i", str(video.resolve()), *args, "-f", "null", "-"], cwd=work,
                            timeout_s=_ANALYSIS_TIMEOUT_S)  # fmt: skip
        return result.stderr_tail

    def _container(self, video: Path, probe: dict[str, Any], plan: RenderPlan) -> list[QACheck]:
        streams = probe["streams"]
        v = [s for s in streams if s["codec_type"] == "video"]
        a = [s for s in streams if s["codec_type"] == "audio"]
        video_value = [
            (s["codec_name"], s.get("pix_fmt"), s["width"], s["height"], s["r_frame_rate"])
            for s in v
        ]
        audio_value = [(s["codec_name"], s["sample_rate"]) for s in a]
        expected_video = ("h264", "yuv420p", plan.width, plan.height, f"{plan.fps}/1")
        return [
            _check("Q1_video_stream", video_value == [expected_video], video_value,
                   f"one {expected_video}"),
            _check("Q1_audio_stream", audio_value == [("aac", "48000")], audio_value,
                   "one ('aac', '48000')"),
            _check("Q1_faststart", moov_before_mdat(video), None, "moov before mdat"),
        ]  # fmt: skip

    def _durations(self, probe: dict[str, Any], duration: float, plan: RenderPlan) -> list[QACheck]:
        s = self._s
        by_type = {st["codec_type"]: float(st["duration"]) for st in probe["streams"]}
        av = abs(by_type["video"] - by_type["audio"])
        planned = plan.total_frames / plan.fps
        return [
            _check("Q2_total_duration", s.qa_min_total_s <= duration <= s.qa_max_total_s,
                   round(duration, 3), f"[{s.qa_min_total_s}, {s.qa_max_total_s}] s"),
            _check("Q2_av_sync", av <= s.qa_av_tolerance_s, round(av, 3),
                   f"<= {s.qa_av_tolerance_s} s"),
            _check("Q2_plan_duration", abs(duration - planned) <= s.qa_plan_tolerance_s,
                   round(duration - planned, 3),
                   f"|actual - {planned:.3f}| <= {s.qa_plan_tolerance_s} s"),
        ]  # fmt: skip

    def _audio(self, video: Path, work: Path, duration: float, plan: RenderPlan) -> list[QACheck]:
        s = self._s
        log = self._analyse(video, work, [
            "-vn", "-af",
            f"silencedetect=n={s.silence_threshold_dbfs}dB:d={s.qa_max_silence_s},"
            "ametadata=mode=print:file=silence.txt,volumedetect,ebur128=framelog=quiet",
        ])  # fmt: skip
        mean = _number(r"mean_volume: (-?[\d.]+|-inf) dB", log)
        peak = _number(r"max_volume: (-?[\d.]+|-inf) dB", log)
        lufs = _number(r"I:\s+(-?[\d.]+|-inf) LUFS", log)
        silences = _intervals((work / "silence.txt").read_text(encoding="utf-8"),
                              "lavfi.silence_start", "lavfi.silence_end", duration)  # fmt: skip
        speech = [
            (sc.start_frame / plan.fps + sc.lead_in_s,
             sc.start_frame / plan.fps + sc.lead_in_s + sc.clip_duration_s)
            for sc in plan.scenes
        ]  # fmt: skip
        # Silence counts only where speech was planned; lead-ins, tails and gaps are padding.
        worst = max((sum(_overlap(gap, w) for w in speech) for gap in silences), default=0.0)
        return [
            _check("Q3_mean_volume", mean > s.qa_mean_volume_db, mean,
                   f"> {s.qa_mean_volume_db} dB"),
            _check("Q3_max_volume", peak <= s.qa_max_volume_db, peak,
                   f"<= {s.qa_max_volume_db} dB"),
            _check("Q3_loudness", s.qa_lufs_min <= lufs <= s.qa_lufs_max, lufs,
                   f"[{s.qa_lufs_min}, {s.qa_lufs_max}] LUFS"),
            _check("Q3_silence", worst <= s.qa_max_silence_s, round(worst, 3),
                   f"<= {s.qa_max_silence_s} s outside planned padding"),
        ]  # fmt: skip

    def _video(self, video: Path, work: Path, duration: float) -> list[QACheck]:
        s = self._s
        self._analyse(video, work, [
            "-an", "-vf",
            f"blackdetect=d={s.qa_black_s}:pic_th={BLACK_PIC_TH}:pix_th={BLACK_PIX_TH},"
            f"freezedetect=d={s.qa_freeze_s},metadata=mode=print:file=video.txt",
        ])  # fmt: skip
        meta = (work / "video.txt").read_text(encoding="utf-8")
        black = [
            e - b for b, e in _intervals(meta, "lavfi.black_start", "lavfi.black_end", duration)
        ]
        freeze = [
            e - b
            for b, e in _intervals(meta, "lavfi.freezedetect.freeze_start",
                                   "lavfi.freezedetect.freeze_end", duration)
        ]  # fmt: skip
        longest_black = round(max(black, default=0.0), 3)
        longest_freeze = round(max(freeze, default=0.0), 3)
        return [
            _check("Q4_black", longest_black < s.qa_black_s, longest_black, f"< {s.qa_black_s} s"),
            _check("Q4_freeze", longest_freeze < s.qa_freeze_s, longest_freeze,
                   f"< {s.qa_freeze_s} s"),
        ]  # fmt: skip

    def _sanity(self, video: Path, work: Path, plan: RenderPlan) -> list[QACheck]:
        s = self._s
        frames = [(sc.start_frame + sc.end_frame) // 2 for sc in plan.scenes]
        select = "+".join(f"eq(n\\,{f})" for f in frames)
        run_ffmpeg(["-i", str(video.resolve()), "-vf", f"select={select},format=gray",
                    "-fps_mode", "vfr", "mid-%02d.png"], cwd=work,
                   timeout_s=_ANALYSIS_TIMEOUT_S)  # fmt: skip
        images = [work / f"mid-{i + 1:02d}.png" for i in range(len(frames))]
        stds = []
        for image in images:
            with Image.open(image) as img:
                stds.append(round(float(np.asarray(img.convert("L"), dtype=float).std()), 2))
        hashes = [average_hash(image) for image in images]
        # Neighbours drawn by the same template share a layout and legitimately hash alike
        # (spec rev 5); only scenes of different visual types are compared.
        distances = [
            (a ^ b).bit_count()
            for (a, sa), (b, sb) in itertools.pairwise(zip(hashes, plan.scenes, strict=True))
            if sa.visual.type != sb.visual.type
        ]
        return [
            _check("Q5_luma_std", min(stds) > s.qa_min_luma_std, stds, f"> {s.qa_min_luma_std}"),
            _check("Q5_scene_hash_distance",
                   min(distances, default=math.inf) >= s.qa_min_hash_distance, distances,
                   f">= {s.qa_min_hash_distance} between neighbours of different visual types"),
        ]  # fmt: skip
