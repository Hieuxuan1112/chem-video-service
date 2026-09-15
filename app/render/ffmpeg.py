"""ffmpeg/ffprobe wrapper that classifies failures as transient or invalid input (spec §8.4)."""

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.render.proc import ProcResult, run_subprocess

_PROBE_TIMEOUT_S = 60.0
_INVALID_INPUT_MARKERS = (
    "No such file or directory",
    "Invalid data found when processing input",
    "does not contain any stream",
    "matches no streams",
    "Error opening input",
    "Invalid argument",
    "Unrecognized option",
    "Option not found",
    "Unknown encoder",
    "Unknown decoder",
    "Error parsing",
    "moov atom not found",
)


class FFmpegError(Exception):
    def __init__(self, message: str, *, stderr_tail: str, transient: bool) -> None:
        super().__init__(message)
        self.stderr_tail = stderr_tail
        self.transient = transient


def _looks_like_invalid_input(stderr: str) -> bool:
    return any(marker in stderr for marker in _INVALID_INPUT_MARKERS)


def _run(
    args: list[str], *, cwd: Path | None, timeout_s: float, cancel: threading.Event | None
) -> ProcResult:
    result = run_subprocess(args, cwd=cwd, timeout_s=timeout_s, cancel=cancel)
    if result.timed_out:
        raise FFmpegError(
            f"{args[0]} timed out after {timeout_s:.0f} s",
            stderr_tail=result.stderr_tail,
            transient=True,
        )
    if result.returncode != 0:
        raise FFmpegError(
            f"{args[0]} exited with {result.returncode}",
            stderr_tail=result.stderr_tail,
            transient=not _looks_like_invalid_input(result.stderr_tail),
        )
    return result


def run_ffmpeg(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout_s: float = 300.0,
    cancel: threading.Event | None = None,
) -> ProcResult:
    return _run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", *args],
        cwd=cwd,
        timeout_s=timeout_s,
        cancel=cancel,
    )


def _probe(args: list[str], path: Path) -> dict[str, Any]:
    result = _run(
        ["ffprobe", "-v", "error", *args, "-of", "json", str(path)],
        cwd=None,
        timeout_s=_PROBE_TIMEOUT_S,
        cancel=None,
    )
    data: dict[str, Any] = json.loads(result.stdout)
    return data


def ffprobe(path: Path) -> dict[str, Any]:
    return _probe(["-show_streams", "-show_format"], path)


def count_frames(path: Path) -> int:
    data = _probe(
        ["-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames"], path
    )
    return int(data["streams"][0]["nb_read_frames"])


def media_duration(path: Path) -> float:
    return float(_probe(["-show_entries", "format=duration"], path)["format"]["duration"])


def make_color_video(
    path: Path,
    *,
    frames: int,
    fps: int,
    size: tuple[int, int],
    color: str,
    box_cell: int | None = None,
) -> None:
    """Solid colour; with box_cell (0..8) also a white box filling that cell of a 3x3 grid."""
    width, height = size
    source = f"color=c={color}:s={width}x{height}:r={fps}"
    if box_cell is not None:
        x, y = box_cell % 3 * width // 3, box_cell // 3 % 3 * height // 3
        source += f",drawbox=x={x}:y={y}:w={width // 3}:h={height // 3}:color=white:t=fill"
    run_ffmpeg(
        [
            "-f", "lavfi", "-i", source,
            "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ]
    )  # fmt: skip
