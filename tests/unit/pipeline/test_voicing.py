"""Voicer tests: FakeTTS/FakeASR, real A1-A4 checks (A4 runs ffmpeg), a real ledger in tmp."""

import wave
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.cost.budget import Settlement, SpendGuard
from app.cost.ledger import Ledger
from app.domain.errors import AppError, BudgetRefused, ErrorCode
from app.jobs.clock import FakeClock
from app.observability.context import job_context
from app.pipeline.storyboard import Storyboard
from app.pipeline.tts_cache import TTSCache
from app.pipeline.voicing import Voicer, spoken_text
from app.providers.asr.fake import FakeASR
from app.providers.errors import ProviderError
from app.providers.tts.fake import FakeTTS
from app.providers.tts.protocol import SynthesisResult, VoiceSpec
from app.topics.registry import TOPICS_ROOT, TopicRegistry

pytestmark = pytest.mark.integration  # A4 runs ffmpeg silencedetect on every accepted clip

REGISTRY = TopicRegistry.load(TOPICS_ROOT)
PACK = REGISTRY.get("ph_scale")
N = len(REGISTRY.curated_storyboard("ph_scale").scenes)


def _storyboard() -> Storyboard:
    return REGISTRY.curated_storyboard("ph_scale")


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"tts_concurrency": 1}
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


class Events:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, type: str, data: dict[str, Any]) -> None:
        self.items.append((type, data))

    def types(self) -> list[str]:
        return [t for t, _ in self.items]


def _voicer(
    tmp_path: Path,
    clock: FakeClock,
    primary: Any,
    fallback: Any,
    asr: Any = None,
    settings: Settings | None = None,
    events: Events | None = None,
) -> tuple[Voicer, Ledger]:
    s = settings or _settings()
    ledger = Ledger(tmp_path / "ledger.sqlite", clock)
    voicer = Voicer(
        primary,
        fallback,
        asr,
        TTSCache(tmp_path / "cache"),
        SpendGuard(ledger, s, clock),
        s,
        emit=events or Events(),
    )
    return voicer, ledger


def _spoken(i: int) -> str:
    return spoken_text(_storyboard().scenes[i].narration, PACK.pronunciations)


def test_pronunciations_applied_to_spoken_text_only(tmp_path: Path, clock: FakeClock) -> None:
    assert spoken_text("What is pH? The pH scale.", {"pH": "P H"}) == "What is P H? The P H scale."
    assert spoken_text("phosphate and PH", {"pH": "P H"}) == "phosphate and PH"  # whole word, case
    assert (
        spoken_text("HCl and H2 and NaCl", {"H2": "H two", "HCl": "hydrogen chloride"})
        == "hydrogen chloride and H two and NaCl"
    )
    sb = _storyboard()
    original = [s.narration for s in sb.scenes]
    primary = FakeTTS(name="primary")
    voicer, _ = _voicer(tmp_path, clock, primary, FakeTTS(name="fallback"))
    voicer.voice(storyboard=sb, pack=PACK, out_dir=tmp_path / "out")
    assert [s.narration for s in sb.scenes] == original
    assert any("P H" in call for call in primary.calls)
    assert not any("pH" in call for call in primary.calls)


def test_all_scenes_primary_success(tmp_path: Path, clock: FakeClock) -> None:
    primary, fallback, events = FakeTTS(name="primary"), FakeTTS(name="fallback"), Events()
    settings = _settings(tts_concurrency=2)
    voicer, _ = _voicer(tmp_path, clock, primary, fallback, settings=settings, events=events)

    outcome = voicer.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "out")

    assert (outcome.provider, outcome.voice, outcome.voice_fallback_reason) == (
        "primary", "fake", None
    )  # fmt: skip
    assert [c.scene_index for c in outcome.clips] == list(range(N))
    assert len(primary.calls) == N and fallback.calls == []
    assert outcome.asr_skipped is True and outcome.asr_skip_reason == "unavailable"
    assert outcome.a5_mode == "report_only"
    for clip in outcome.clips:
        assert clip.wav_path == tmp_path / "out" / f"scene-{clip.scene_index:02d}.wav"
        with wave.open(str(clip.wav_path)) as w:
            assert (w.getnchannels(), w.getframerate()) == (1, 24_000)
            assert w.getnframes() / 24_000 == pytest.approx(clip.duration_s)
        assert [c["name"] for c in clip.checks] == ["A1", "A2", "A3", "A4"]
        assert all(c["passed"] for c in clip.checks)
        assert (clip.provider, clip.cache_hit, clip.wer) == ("primary", False, None)
    assert "voice_provider_switched" not in events.types()
    report = outcome.report()
    assert report["provider"] == "primary" and len(report["clips"]) == N
    assert isinstance(report["clips"][0]["wav_path"], str)


def test_scene_truncated_then_resynth_success_same_provider(
    tmp_path: Path, clock: FakeClock
) -> None:
    primary, events = FakeTTS(name="primary", script={0: "truncate", 1: "gap"}), Events()
    voicer, _ = _voicer(tmp_path, clock, primary, FakeTTS(name="fallback"), events=events)
    outcome = voicer.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "out")
    assert outcome.provider == "primary"
    assert len(primary.calls) == N + 2
    assert primary.calls[:3] == [_spoken(0)] * 3
    resynth = [d for t, d in events.items if t == "clip_resynthesis"]
    assert [(d["scene_index"], d["attempt"], d["failed_checks"]) for d in resynth] == [
        (0, 2, ["A2"]),
        (0, 3, ["A4"]),
    ]


def test_scene_exhausts_retries_switches_whole_video(tmp_path: Path, clock: FakeClock) -> None:
    primary = FakeTTS(name="primary", script={0: "silence", 1: "silence", 2: "silence"})
    fallback, events = FakeTTS(name="fallback"), Events()
    voicer, _ = _voicer(tmp_path, clock, primary, fallback, events=events)

    outcome = voicer.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "out")

    assert len(primary.calls) == 3  # 1 synthesis + 2 re-syntheses, then no other scene is voiced
    assert len(fallback.calls) == N
    assert outcome.provider == "fallback"
    assert outcome.voice_fallback_reason == "PRIMARY_VALIDATION_FAILED"
    assert {c.provider for c in outcome.clips} == {"fallback"}
    [switch] = [d for t, d in events.items if t == "voice_provider_switched"]
    assert switch == {
        "from": "primary", "to": "fallback", "reason": "PRIMARY_VALIDATION_FAILED", "scene_index": 0
    }  # fmt: skip


def test_primary_provider_error_switches_whole_video(tmp_path: Path, clock: FakeClock) -> None:
    primary = FakeTTS(name="primary", script={2: ProviderError("server", "503", status=503)})
    fallback = FakeTTS(name="fallback")
    voicer, _ = _voicer(tmp_path, clock, primary, fallback)
    outcome = voicer.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "out")
    assert len(primary.calls) == 3
    assert len(fallback.calls) == N  # already-voiced primary clips are discarded
    assert outcome.voice_fallback_reason == "PRIMARY_PROVIDER_ERROR"


def test_cached_valid_clip_not_resynthesized(tmp_path: Path, clock: FakeClock) -> None:
    first = FakeTTS(name="primary")
    voicer, _ = _voicer(tmp_path, clock, first, FakeTTS(name="fallback"))
    voicer.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "run1")

    second = FakeTTS(name="primary")
    voicer2, _ = _voicer(tmp_path, clock, second, FakeTTS(name="fallback"))
    outcome = voicer2.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "run2")
    assert second.calls == []
    assert all(c.cache_hit for c in outcome.clips)
    assert all((tmp_path / "run2" / f"scene-{i:02d}.wav").is_file() for i in range(N))


def test_budget_precheck_routes_to_fallback_from_start(tmp_path: Path, clock: FakeClock) -> None:
    primary, fallback, events = (
        FakeTTS(name="primary", paid=True),
        FakeTTS(name="fallback"),
        Events(),
    )
    settings = _settings(cost_job_budget_usd=0.001)
    voicer, _ = _voicer(tmp_path, clock, primary, fallback, settings=settings, events=events)
    with job_context(job_id="job1", attempt=1, worker_id="w"):
        outcome = voicer.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "out")
    assert primary.calls == []
    assert len(fallback.calls) == N
    assert outcome.voice_fallback_reason == "BUDGET_EXCEEDED_JOB"
    assert (
        "budget_degraded",
        {"stage": "voicing", "reason": "BUDGET_EXCEEDED_JOB"},
    ) in events.items

    capped = _settings(hard_project_cap_usd=0.001)
    primary2 = FakeTTS(name="primary", paid=True)
    voicer2, _ = _voicer(tmp_path / "b", clock, primary2, FakeTTS(name="fallback"), settings=capped)
    outcome2 = voicer2.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "out2")
    assert primary2.calls == [] and outcome2.voice_fallback_reason == "PROJECT_CAP_REACHED"

    free = FakeTTS(name="primary")  # a free primary needs no pre-check
    voicer3, _ = _voicer(tmp_path / "c", clock, free, FakeTTS(name="fallback"), settings=capped)
    assert voicer3.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "o3").provider == (
        "primary"
    )


def test_both_providers_error_raises_unavailable(tmp_path: Path, clock: FakeClock) -> None:
    err = ProviderError("server", "503", status=503)
    voicer, _ = _voicer(
        tmp_path,
        clock,
        FakeTTS(name="primary", script={0: err}),
        FakeTTS(name="fb", script={0: err}),
    )
    with pytest.raises(AppError) as info:
        voicer.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "out")
    assert (info.value.code, info.value.cause) == (
        ErrorCode.TTS_PROVIDER_UNAVAILABLE, "provider_outage"
    )  # fmt: skip
    assert info.value.retryable is True


def test_validation_exhausted_raises_validation_failed(tmp_path: Path, clock: FakeClock) -> None:
    silent = {0: "silence", 1: "silence", 2: "silence"}
    voicer, _ = _voicer(
        tmp_path, clock, FakeTTS(name="primary", script=silent), FakeTTS(name="fb", script=silent)
    )
    with pytest.raises(AppError) as info:
        voicer.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "out")
    assert (info.value.code, info.value.cause) == (
        ErrorCode.TTS_VALIDATION_FAILED, "output_invalid_after_retries"
    )  # fmt: skip
    assert info.value.details["scene_index"] == 0


def test_a5_report_only_records_wer_without_gating(tmp_path: Path, clock: FakeClock) -> None:
    transcripts = [_spoken(i) for i in range(N)]
    transcripts[1] = "completely different words here"
    asr = FakeASR(transcripts)
    primary = FakeTTS(name="primary")
    voicer, _ = _voicer(tmp_path, clock, primary, FakeTTS(name="fb"), asr=asr)
    outcome = voicer.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "out")
    assert outcome.a5_mode == "report_only" and outcome.asr_skipped is False
    assert len(primary.calls) == N  # a bad WER does not trigger re-synthesis
    assert outcome.clips[0].wer == 0.0
    assert outcome.clips[1].wer is not None and outcome.clips[1].wer > 0.5
    assert asr.calls == [pytest.approx(c.duration_s) for c in outcome.clips]


def test_a5_gating_when_threshold_set(tmp_path: Path, clock: FakeClock) -> None:
    transcripts = [_spoken(i) for i in range(N)]
    transcripts.insert(0, "nothing like the narration at all")
    asr, primary = FakeASR(transcripts), FakeTTS(name="primary")
    settings = _settings(a5_wer_threshold=0.3)
    voicer, _ = _voicer(tmp_path, clock, primary, FakeTTS(name="fb"), asr=asr, settings=settings)
    outcome = voicer.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "out")
    assert outcome.a5_mode == "gating"
    assert len(primary.calls) == N + 1
    assert outcome.clips[0].wer == 0.0
    assert outcome.clips[0].checks[-1] == {
        "name": "A5", "passed": True, "value": 0.0, "threshold": "<=0.3", "message": "WER 0.00"
    }  # fmt: skip


def test_asr_unavailable_sets_skipped(tmp_path: Path, clock: FakeClock) -> None:
    voicer, _ = _voicer(tmp_path, clock, FakeTTS(name="p"), FakeTTS(name="fb"), asr=None)
    outcome = voicer.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "o1")
    assert (outcome.asr_skipped, outcome.asr_skip_reason) == (True, "unavailable")

    erroring = FakeASR([ProviderError("server", "503", status=503)])
    voicer2, _ = _voicer(tmp_path / "b", clock, FakeTTS(name="p"), FakeTTS(name="fb"), asr=erroring)
    outcome2 = voicer2.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "o2")
    assert (outcome2.asr_skipped, outcome2.asr_skip_reason) == (True, "provider_error")
    assert erroring.calls and len(erroring.calls) == 1  # skipped for the rest of the video
    assert all(c.wer is None for c in outcome2.clips)

    class RefusingASR:
        name = "refusing"

        def transcribe(self, wav_bytes: bytes, *, audio_s: float) -> str:
            raise BudgetRefused("job", 0.001, 0.2, 0.2)

    events = Events()
    voicer3, _ = _voicer(
        tmp_path / "c", clock, FakeTTS(name="p"), FakeTTS(name="fb"), asr=RefusingASR(),
        events=events,
    )  # fmt: skip
    outcome3 = voicer3.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "o3")
    assert (outcome3.asr_skipped, outcome3.asr_skip_reason) == (True, "budget")
    assert ("budget_degraded", {"stage": "asr", "reason": "BUDGET_EXCEEDED_JOB"}) in events.items


class GuardedFakeTTS:
    """A paid adapter stand-in: every synthesis reserves and settles through the guard."""

    paid = True

    def __init__(self, inner: FakeTTS, guard: SpendGuard) -> None:
        self.inner = inner
        self.guard = guard
        self.name = inner.name

    def voice_spec(self) -> VoiceSpec:
        return self.inner.voice_spec()

    def synthesize(self, text: str) -> SynthesisResult:
        def settle(result: object, exc: BaseException | None) -> Settlement:
            if exc is not None:
                return Settlement("error", 0.0, "estimated", {}, "server", str(exc))
            return Settlement("ok", 0.0001, "actual", {"characters": len(text)})

        return self.guard.run(
            operation="tts",
            provider="fake_paid",
            model="gemini-2.5-flash-preview-tts",
            projected_usd=0.001,
            call=lambda: self.inner.synthesize(text),
            settle=settle,
        )


def test_failed_calls_in_ledger(tmp_path: Path, clock: FakeClock) -> None:
    s = _settings()
    ledger = Ledger(tmp_path / "ledger.sqlite", clock)
    guard = SpendGuard(ledger, s, clock)
    inner = FakeTTS(name="primary", script={1: ProviderError("server", "503", status=503)})
    primary = GuardedFakeTTS(inner, guard)
    fallback = FakeTTS(name="fallback")
    voicer = Voicer(primary, fallback, None, TTSCache(tmp_path / "cache"), guard, s)

    outcome = voicer.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "out")

    assert outcome.provider == "fallback"
    rows = [dict(r) for r in ledger._conn().execute("SELECT * FROM cost_entries ORDER BY id")]
    assert [(r["operation"], r["status"]) for r in rows] == [("tts", "ok"), ("tts", "error")]
    assert ledger.unsettled() == []


def test_asr_skip_during_fallback_run_is_reported(tmp_path: Path, clock: FakeClock) -> None:
    primary = FakeTTS(name="primary", script={0: ProviderError("server", "503", status=503)})
    asr = FakeASR([ProviderError("timeout", "slow")])
    voicer, _ = _voicer(tmp_path, clock, primary, FakeTTS(name="fallback"), asr=asr)
    outcome = voicer.voice(storyboard=_storyboard(), pack=PACK, out_dir=tmp_path / "out")
    assert outcome.provider == "fallback"
    assert (outcome.asr_skipped, outcome.asr_skip_reason) == (True, "provider_error")
