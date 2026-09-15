from pathlib import Path

import pytest
from PIL import Image

from app.config import Settings
from app.domain.errors import AppError, ErrorCode
from app.providers.tts import fake as fake_tts_module
from app.providers.tts.fake import FakeTTS
from app.render.assembler import MediaAssembler
from app.render.layout import FontSet
from app.render.media_qa import MediaQA, QAReport, average_hash, moov_before_mdat
from app.render.plan import RenderPlan
from tests.integration import test_assembler as asm

pytestmark = pytest.mark.integration

SETTINGS = Settings(_env_file=None, width=320, height=180, fps=15, qa_min_total_s=10.0)  # type: ignore[call-arg]


def _video(
    tmp_path: Path,
    fonts: FontSet,
    *,
    script: dict[int, str] | None = None,
    tts: FakeTTS | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> tuple[RenderPlan, Path]:
    if tts is not None and monkeypatch is not None:
        monkeypatch.setattr(asm, "FakeTTS", lambda: tts)
    plan, clips, videos = asm._inputs(tmp_path, script)
    result = MediaAssembler(SETTINGS, fonts).assemble(
        plan=plan, clips=clips, scene_videos=videos, workdir=tmp_path / "work"
    )
    return plan, result.video


def _failed(report: QAReport) -> list[str]:
    return [c.name for c in report.checks if not c.passed]


@pytest.fixture(scope="module")
def good(tmp_path_factory: pytest.TempPathFactory, fonts: FontSet) -> tuple[RenderPlan, Path]:
    return _video(tmp_path_factory.mktemp("good"), fonts)


def test_good_video_passes(good: tuple[RenderPlan, Path]) -> None:
    plan, video = good
    qa = MediaQA(SETTINGS)
    report = qa.check(video=video, plan=plan)

    assert report.passed, report.to_dict()
    assert {c.name[:2] for c in report.checks} == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    assert moov_before_mdat(video)
    qa.raise_for(report)  # no exception
    data = report.to_dict()
    assert data["passed"] is True and all("threshold" in c for c in data["checks"])


def test_black_segment_fails_q4_deterministic(tmp_path: Path, fonts: FontSet) -> None:
    plan, video = _video(tmp_path, fonts, script={5: "black"})  # last scene: 1.5 s tail, no caption
    qa = MediaQA(SETTINGS)
    report = qa.check(video=video, plan=plan)

    assert any(name.startswith("Q4") for name in _failed(report)), report.to_dict()
    with pytest.raises(AppError) as info:
        qa.raise_for(report)
    assert (info.value.code, info.value.cause, info.value.retryable) == (
        ErrorCode.QUALITY_CHECK_FAILED, "deterministic_output_bug", False,
    )  # fmt: skip


def test_duration_mismatch_fails_q2(good: tuple[RenderPlan, Path]) -> None:
    plan, video = good
    longer = plan.model_copy(update={"total_frames": plan.total_frames + plan.fps})

    report = MediaQA(SETTINGS).check(video=video, plan=longer)

    assert _failed(report) == ["Q2_plan_duration"], report.to_dict()


def test_silent_audio_fails_q3(
    tmp_path: Path, fonts: FontSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    silent = FakeTTS(script={i: "silence" for i in range(10)})
    plan, video = _video(tmp_path, fonts, tts=silent, monkeypatch=monkeypatch)

    report = MediaQA(SETTINGS).check(video=video, plan=plan)

    failed = _failed(report)
    assert failed and all(name.startswith("Q3") for name in failed), report.to_dict()


def test_long_gap_inside_speech_fails_q3_silence(
    tmp_path: Path, fonts: FontSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fake_tts_module, "_GAP_S", 3.0)
    gappy = FakeTTS(script={2: "gap"})
    plan, video = _video(tmp_path, fonts, tts=gappy, monkeypatch=monkeypatch)

    report = MediaQA(SETTINGS).check(video=video, plan=plan)

    assert "Q3_silence" in _failed(report), report.to_dict()


def test_identical_scenes_fail_q5(tmp_path: Path, fonts: FontSet) -> None:
    plan, video = _video(tmp_path, fonts, script=dict.fromkeys(range(6), "same"))

    report = MediaQA(SETTINGS).check(video=video, plan=plan)

    assert "Q5_scene_hash_distance" in _failed(report), report.to_dict()
    assert "Q5_luma_std" not in _failed(report)


def test_blank_scene_fails_q5_luma(tmp_path: Path, fonts: FontSet) -> None:
    plan, video = _video(tmp_path, fonts, script={3: "blank"})

    report = MediaQA(SETTINGS).check(video=video, plan=plan)

    assert "Q5_luma_std" in _failed(report), report.to_dict()


def test_unreadable_file_io_error_retryable(tmp_path: Path, good: tuple[RenderPlan, Path]) -> None:
    plan, _ = good
    broken = tmp_path / "video.mp4"
    broken.write_bytes(b"not an mp4 at all")
    qa = MediaQA(SETTINGS)

    report = qa.check(video=broken, plan=plan)

    assert not report.passed
    assert {c.cause for c in report.checks if not c.passed} == {"io_error"}
    with pytest.raises(AppError) as info:
        qa.raise_for(report)
    assert (info.value.cause, info.value.retryable) == ("io_error", True)


def test_moov_before_mdat_detects_order(tmp_path: Path) -> None:
    def box(kind: bytes, payload: bytes = b"") -> bytes:
        return (8 + len(payload)).to_bytes(4, "big") + kind + payload

    fast = tmp_path / "fast.mp4"
    fast.write_bytes(box(b"ftyp", b"isom") + box(b"moov", b"x" * 16) + box(b"mdat", b"y" * 32))
    slow = tmp_path / "slow.mp4"
    slow.write_bytes(box(b"ftyp", b"isom") + box(b"mdat", b"y" * 32) + box(b"moov", b"x" * 16))

    assert moov_before_mdat(fast) and not moov_before_mdat(slow)


def test_average_hash_is_64_bits(tmp_path: Path) -> None:
    image = tmp_path / "half.png"
    canvas = Image.new("L", (64, 64), 0)
    canvas.paste(255, (0, 0, 32, 64))
    canvas.save(image)

    value = average_hash(image)
    assert value.bit_count() == 32 and value < 2**64
