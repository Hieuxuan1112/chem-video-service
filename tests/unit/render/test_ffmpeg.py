import time
from pathlib import Path

import pytest

from app.render.ffmpeg import (
    FFmpegError,
    count_frames,
    ffprobe,
    make_color_video,
    media_duration,
    run_ffmpeg,
)

pytestmark = pytest.mark.integration  # requires ffmpeg and ffprobe on PATH


def test_make_color_video_exact_frames(tmp_path: Path) -> None:
    path = tmp_path / "blue.mp4"
    make_color_video(path, frames=47, fps=30, size=(320, 180), color="#0F172A")
    assert count_frames(path) == 47
    assert media_duration(path) == pytest.approx(47 / 30, abs=0.05)


def test_ffprobe_reports_stream(tmp_path: Path) -> None:
    path = tmp_path / "red.mp4"
    make_color_video(path, frames=15, fps=15, size=(640, 360), color="red")
    info = ffprobe(path)
    [video] = [s for s in info["streams"] if s["codec_type"] == "video"]
    assert (video["codec_name"], video["width"], video["height"]) == ("h264", 640, 360)
    assert video["pix_fmt"] == "yuv420p"
    assert video["r_frame_rate"] == "15/1"
    assert "duration" in info["format"]


def test_missing_input_is_non_transient(tmp_path: Path) -> None:
    with pytest.raises(FFmpegError) as info:
        run_ffmpeg(["-i", str(tmp_path / "missing.mp4"), str(tmp_path / "out.mp4")])
    assert info.value.transient is False
    assert "missing.mp4" in info.value.stderr_tail
    with pytest.raises(FFmpegError) as probe:
        ffprobe(tmp_path / "missing.mp4")
    assert probe.value.transient is False
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"not a video at all")
    with pytest.raises(FFmpegError) as bad:
        run_ffmpeg(["-i", str(garbage), str(tmp_path / "out2.mp4")])
    assert bad.value.transient is False


def test_timeout_raises_or_flags(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(FFmpegError) as info:
        # -re reads the input in real time, so 30 s of input cannot finish within 1 s
        run_ffmpeg(
            [
                "-re",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:r=10",
                "-t",
                "30",
                "-f",
                "null",
                "-",
            ],
            timeout_s=1.0,
        )
    assert info.value.transient is True
    assert "timed out" in str(info.value)
    assert time.monotonic() - started < 15
