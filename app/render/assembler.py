"""MediaAssembler (spec 8.4): sample-accurate narration, frame-exact visuals, one final encode.

Every ffmpeg call goes through app.render.ffmpeg. A failing step whose stderr looks transient is
re-run once; a missing or invalid input is never retried.
"""

import math
import shutil
import threading
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TypeVar

import numpy as np

from app.config import Settings
from app.domain.errors import AppError, ErrorCode
from app.pipeline.voicing import VoicedClip
from app.render.captions import build_cues, to_ass, to_vtt
from app.render.ffmpeg import FFmpegError, count_frames, run_ffmpeg
from app.render.layout import FontSet
from app.render.plan import RenderPlan

SAMPLE_RATE = 48_000
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"
FFMPEG_TIMEOUT_S = 600.0
CONTACT_SHEET_COLUMNS = 4
CONTACT_SHEET_TILE_WIDTH = 320

T = TypeVar("T")


@dataclass
class AssemblyResult:
    video: Path
    narration_wav: Path
    captions_vtt: Path
    captions_ass: Path
    thumbnail: Path
    contact_sheet: Path
    frame_fixes: list[str]


def _invalid(message: str, **details: object) -> AppError:
    return AppError(ErrorCode.ASSEMBLY_FAILED, "invalid_input", message, dict(details))


class MediaAssembler:
    def __init__(self, settings: Settings, fonts: FontSet) -> None:
        self._settings = settings
        self._fonts = fonts
        self._cancel: threading.Event | None = None

    def assemble(
        self,
        *,
        plan: RenderPlan,
        clips: Sequence[VoicedClip],
        scene_videos: Sequence[Path],
        workdir: Path,
        cancel: threading.Event | None = None,
    ) -> AssemblyResult:
        self._cancel = cancel
        if not (len(clips) == len(scene_videos) == len(plan.scenes)):
            raise _invalid(
                f"{len(clips)} clips and {len(scene_videos)} videos for {len(plan.scenes)} scenes"
            )
        for path in [*(c.wav_path for c in clips), *scene_videos]:
            if not path.is_file():
                raise _invalid(f"missing input {path.name}", path=str(path))
        workdir.mkdir(parents=True, exist_ok=True)

        narration = workdir / "narration.wav"
        self._step("narration", lambda: self._narration(plan, clips, narration))
        fixes: list[str] = []
        videos = [
            self._step("frame_fix", partial(self._fix_frames, plan, i, video, workdir, fixes))
            for i, video in enumerate(scene_videos)
        ]
        visual = workdir / "visual.mp4"
        self._step("concat", lambda: self._concat(videos, workdir, visual))

        cues = build_cues(plan, self._fonts)
        vtt = workdir / "captions.vtt"
        ass = workdir / "captions.ass"
        vtt.write_text(to_vtt(cues), encoding="utf-8")
        ass.write_text(to_ass(cues, width=plan.width, height=plan.height), encoding="utf-8")
        fonts_dir = workdir / "fonts"
        fonts_dir.mkdir(exist_ok=True)
        for font in (self._fonts.regular, self._fonts.semibold, self._fonts.bold):
            shutil.copyfile(font, fonts_dir / font.name)

        video = workdir / "video.mp4"
        self._step("final_encode", lambda: self._final_encode(plan, workdir))
        thumbnail = workdir / "thumbnail.jpg"
        self._step("thumbnail", lambda: self._thumbnail(plan, visual, thumbnail))
        sheet = workdir / "contact_sheet.jpg"
        self._step("contact_sheet", lambda: self._contact_sheet(plan, visual, sheet))
        return AssemblyResult(video, narration, vtt, ass, thumbnail, sheet, fixes)

    # --- retry policy --------------------------------------------------------------------------

    def _step(self, name: str, action: Callable[[], T]) -> T:
        try:
            return action()
        except FFmpegError as first:
            if not first.transient:
                raise self._to_app_error(name, first) from first
            try:
                return action()
            except FFmpegError as second:
                raise self._to_app_error(name, second) from second

    @staticmethod
    def _to_app_error(step: str, exc: FFmpegError) -> AppError:
        cause = "ffmpeg_transient" if exc.transient else "invalid_input"
        return AppError(
            ErrorCode.ASSEMBLY_FAILED,
            cause,
            f"{step}: {exc}",
            {"step": step, "stderr_tail": exc.stderr_tail[-1000:]},
        )

    def _ffmpeg(self, args: Sequence[str], *, cwd: Path | None = None) -> bytes:
        return run_ffmpeg(args, cwd=cwd, timeout_s=FFMPEG_TIMEOUT_S, cancel=self._cancel).stdout

    # --- steps -----------------------------------------------------------------------------------

    def _narration(self, plan: RenderPlan, clips: Sequence[VoicedClip], out: Path) -> None:
        total = round(plan.total_frames / plan.fps * SAMPLE_RATE)
        buffer = np.zeros(total, dtype=np.float32)
        for clip, scene in zip(clips, plan.scenes, strict=True):
            raw = self._ffmpeg(
                ["-i", str(clip.wav_path), "-af", LOUDNORM, "-ac", "1", "-ar", str(SAMPLE_RATE),
                 "-f", "f32le", "-"]
            )  # fmt: skip
            samples = np.frombuffer(raw, dtype="<f4")
            offset = round((scene.start_frame / plan.fps + scene.lead_in_s) * SAMPLE_RATE)
            end = min(total, offset + len(samples))
            buffer[offset:end] += samples[: end - offset]
        pcm = (np.clip(buffer, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)

    def _fix_frames(
        self, plan: RenderPlan, index: int, video: Path, workdir: Path, fixes: list[str]
    ) -> Path:
        planned = plan.scenes[index].frames
        drift = count_frames(video) - planned
        if drift == 0:
            return video
        if abs(drift) > 1:
            raise _invalid(f"scene {index} has {drift:+d} frames", scene=index)
        fixed = workdir / f"scene-{index:02d}.fixed.mp4"
        vf = f"trim=end_frame={planned}" if drift > 0 else "tpad=stop_mode=clone:stop=1"
        self._ffmpeg(
            ["-i", str(video), "-vf", vf, "-r", str(plan.fps), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-an", str(fixed)]
        )  # fmt: skip
        fixes.append(f"scene {index}: {'trimmed' if drift > 0 else 'cloned'} 1 frame")
        return fixed

    def _concat(self, videos: Sequence[Path], workdir: Path, out: Path) -> None:
        listing = workdir / "concat.txt"
        lines = ["file '" + v.resolve().as_posix().replace("'", "'\\''") + "'" for v in videos]
        listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)])

    def _final_encode(self, plan: RenderPlan, workdir: Path) -> None:
        # Relative names with cwd=workdir avoid escaping a Windows drive colon inside the filter.
        self._ffmpeg(
            [
                "-i", "visual.mp4", "-i", "narration.wav",
                "-map", "0:v:0", "-map", "1:a:0",
                "-vf", "subtitles=captions.ass:fontsdir=fonts",
                "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
                "-crf", "23", "-preset", "medium", "-r", str(plan.fps),
                "-c:a", "aac", "-b:a", "128k", "-ar", str(SAMPLE_RATE),
                "-movflags", "+faststart", "video.mp4",
            ],
            cwd=workdir,
        )  # fmt: skip

    def _thumbnail(self, plan: RenderPlan, visual: Path, out: Path) -> None:
        scene = next((s for s in plan.scenes if s.visual.type == "title_card"), plan.scenes[0])
        frame = (scene.start_frame + scene.end_frame) // 2
        self._ffmpeg(
            ["-i", str(visual), "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1",
             "-q:v", "3", str(out)]
        )  # fmt: skip

    def _contact_sheet(self, plan: RenderPlan, visual: Path, out: Path) -> None:
        frames = [(s.start_frame + s.end_frame) // 2 for s in plan.scenes]
        columns = min(CONTACT_SHEET_COLUMNS, len(frames))
        rows = math.ceil(len(frames) / columns)
        select = "+".join(f"eq(n\\,{f})" for f in frames)
        self._ffmpeg(
            ["-i", str(visual),
             "-vf", f"select={select},scale={CONTACT_SHEET_TILE_WIDTH}:-2,tile={columns}x{rows}",
             "-fps_mode", "vfr", "-frames:v", "1", "-q:v", "3", str(out)]
        )  # fmt: skip
