"""Per-scene voicing with A1-A5 validation, a TTS cache and a one-voice video fallback (spec §7)."""

import contextvars
import io
import re
import threading
import wave
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.config import Settings
from app.cost.budget import SpendGuard
from app.cost.pricing import project_tts_call
from app.domain.errors import AppError, BudgetRefused, ErrorCode
from app.observability.context import job_id_var
from app.pipeline.audio_checks import ClipReport, check_clip, write_wav
from app.pipeline.storyboard import Storyboard, word_count
from app.pipeline.tts_cache import TTSCache, tts_cache_key
from app.pipeline.wer import word_error_rate
from app.providers.asr.protocol import ASRProvider
from app.providers.errors import ProviderError
from app.providers.tts.protocol import TTSProvider
from app.topics.pack import TopicPack

BUDGET_REASONS = {
    "job": "BUDGET_EXCEEDED_JOB",
    "project": "PROJECT_CAP_REACHED",
    "daily": "BUDGET_EXCEEDED_DAILY",
}


def spoken_text(narration: str, pronunciations: Mapping[str, str]) -> str:
    """Whole-word, case-sensitive replacements; longer display forms first."""
    text = narration
    for display in sorted(pronunciations, key=len, reverse=True):
        spoken = pronunciations[display]

        def replacement(_: re.Match[str], spoken: str = spoken) -> str:
            return spoken

        text = re.sub(rf"(?<!\w){re.escape(display)}(?!\w)", replacement, text)
    return text


@dataclass
class VoicedClip:
    scene_index: int
    wav_path: Path
    duration_s: float
    provider: str
    voice: str
    cache_hit: bool
    checks: list[dict[str, Any]]
    wer: float | None


@dataclass
class VoicingOutcome:
    clips: list[VoicedClip]
    provider: str
    voice: str
    voice_fallback_reason: str | None
    asr_skipped: bool
    asr_skip_reason: str | None
    a5_mode: Literal["report_only", "gating"]

    def report(self) -> dict[str, Any]:
        return {
            "clips": [{**asdict(c), "wav_path": str(c.wav_path)} for c in self.clips],
            "provider": self.provider,
            "voice": self.voice,
            "voice_fallback_reason": self.voice_fallback_reason,
            "asr_skipped": self.asr_skipped,
            "asr_skip_reason": self.asr_skip_reason,
            "a5_mode": self.a5_mode,
        }


class _Aborted(Exception):
    """Another scene already failed on this provider; stop before spending more."""


class _ValidationExhausted(Exception):
    def __init__(self, scene_index: int, provider: str, checks: list[dict[str, Any]]) -> None:
        super().__init__(f"scene {scene_index} failed validation on {provider}")
        self.scene_index = scene_index
        self.provider = provider
        self.checks = checks


class _ProviderFailed(Exception):
    def __init__(self, scene_index: int, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.scene_index = scene_index
        self.cause = cause


@dataclass
class _Run:
    abort: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    asr_skip_reason: str | None = None


def _wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buffer.getvalue()


class Voicer:
    def __init__(
        self,
        primary: TTSProvider,
        fallback: TTSProvider,
        asr: ASRProvider | None,
        cache: TTSCache,
        guard: SpendGuard,
        settings: Settings,
        *,
        emit: Callable[[str, dict[str, Any]], None] = lambda t, d: None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._asr = asr
        self._cache = cache
        self._guard = guard
        self._settings = settings
        self._emit = emit

    # --- orchestration ---------------------------------------------------------------------------

    def voice(self, *, storyboard: Storyboard, pack: TopicPack, out_dir: Path) -> VoicingOutcome:
        texts = [spoken_text(s.narration, pack.pronunciations) for s in storyboard.scenes]
        run = _Run(asr_skip_reason=None if self._asr is not None else "unavailable")
        reason = self._precheck(texts)
        primary_failure: _ProviderFailed | None = None
        if reason is None:
            try:
                return self._outcome(
                    self._primary, self._voice_all(self._primary, texts, out_dir, run), None, run
                )
            except _ProviderFailed as failure:
                primary_failure = failure
                reason = self._switch_reason(failure)
                self._emit(
                    "voice_provider_switched",
                    {
                        "from": self._primary.name,
                        "to": self._fallback.name,
                        "reason": reason,
                        "scene_index": failure.scene_index,
                    },
                )
        fallback_run = _Run(asr_skip_reason=run.asr_skip_reason)  # a skipped ASR stays skipped
        try:
            clips = self._voice_all(self._fallback, texts, out_dir, fallback_run)
        except _ProviderFailed as failure:
            raise self._final_error(primary_failure, failure) from failure.cause
        return self._outcome(self._fallback, clips, reason, fallback_run)

    def _outcome(
        self, provider: TTSProvider, clips: list[VoicedClip], reason: str | None, run: _Run
    ) -> VoicingOutcome:
        return VoicingOutcome(
            clips=clips,
            provider=provider.name,
            voice=provider.voice_spec().voice,
            voice_fallback_reason=reason,
            asr_skipped=run.asr_skip_reason is not None,
            asr_skip_reason=run.asr_skip_reason,
            a5_mode="report_only" if self._settings.a5_wer_threshold is None else "gating",
        )

    def _precheck(self, texts: list[str]) -> str | None:
        """Chooses the provider only; every clip still reserves and settles through the guard."""
        if not self._primary.paid:
            return None
        s = self._settings
        model = self._primary.voice_spec().model or s.tts_model
        projected = sum(
            project_tts_call(model, len(t), word_count(t) / s.words_per_second, s.a2_max_ratio)[0]
            for t in texts
        )
        headroom = self._guard.headroom(job_id=job_id_var.get())
        for scope, key in (("project", "project_usd"), ("job", "job_usd"), ("daily", "daily_usd")):
            remaining = headroom[key]
            if remaining is not None and projected > remaining:
                reason = BUDGET_REASONS[scope]
                self._emit("budget_degraded", {"stage": "voicing", "reason": reason})
                return reason
        return None

    def _switch_reason(self, failure: _ProviderFailed) -> str:
        cause = failure.cause
        if isinstance(cause, BudgetRefused):
            reason = BUDGET_REASONS[cause.scope]
            self._emit("budget_degraded", {"stage": "voicing", "reason": reason})
            return reason
        if isinstance(cause, ProviderError):
            return "PRIMARY_PROVIDER_ERROR"
        return "PRIMARY_VALIDATION_FAILED"

    @staticmethod
    def _final_error(primary: _ProviderFailed | None, fallback: _ProviderFailed) -> AppError:
        def outage(f: _ProviderFailed | None) -> bool:
            return f is None or isinstance(f.cause, ProviderError | BudgetRefused)

        details: dict[str, Any] = {"scene_index": fallback.scene_index}
        if isinstance(fallback.cause, _ValidationExhausted):
            details["checks"] = fallback.cause.checks
        if outage(fallback) and outage(primary):
            return AppError(
                ErrorCode.TTS_PROVIDER_UNAVAILABLE,
                "provider_outage",
                f"both TTS providers failed: {fallback.cause}",
                details,
            )
        return AppError(
            ErrorCode.TTS_VALIDATION_FAILED,
            "output_invalid_after_retries",
            f"narration clips failed validation on every provider: {fallback.cause}",
            details,
        )

    def _voice_all(
        self, provider: TTSProvider, texts: list[str], out_dir: Path, run: _Run
    ) -> list[VoicedClip]:
        with ThreadPoolExecutor(max_workers=self._settings.tts_concurrency) as pool:
            futures = [
                # copy_context: job/attempt/stage contextvars reach the ledger from pool threads
                pool.submit(
                    contextvars.copy_context().run,
                    self._voice_scene,
                    i,
                    text,
                    provider,
                    out_dir,
                    run,
                )
                for i, text in enumerate(texts)
            ]
        clips: list[VoicedClip] = []
        failure: _ProviderFailed | None = None
        for i, future in enumerate(futures):
            try:
                clips.append(future.result())
            except _Aborted:
                continue
            except (ProviderError, BudgetRefused, _ValidationExhausted) as exc:
                failure = failure or _ProviderFailed(i, exc)
        if failure is not None:
            raise failure
        return clips

    # --- one scene -------------------------------------------------------------------------------

    def _voice_scene(
        self, index: int, text: str, provider: TTSProvider, out_dir: Path, run: _Run
    ) -> VoicedClip:
        try:
            return self._voice_scene_inner(index, text, provider, out_dir, run)
        except _Aborted:
            raise
        except BaseException:
            run.abort.set()
            raise

    def _voice_scene_inner(
        self, index: int, text: str, provider: TTSProvider, out_dir: Path, run: _Run
    ) -> VoicedClip:
        s = self._settings
        spec = provider.voice_spec()
        key = tts_cache_key(spec, text)
        words = word_count(text)
        workdir = out_dir / "work" / f"scene-{index:02d}-{spec.provider}"

        cached = self._cache.get(key)
        if cached is not None:
            report = check_clip(cached, text_words=words, settings=s, workdir=workdir)
            if report.passed:
                return self._clip(index, report, provider, out_dir, True, None, [])

        checks: list[dict[str, Any]] = []
        for attempt in range(1, s.clip_resynth_attempts + 2):
            if run.abort.is_set():
                raise _Aborted
            if attempt > 1:
                failed = [c["name"] for c in checks if not c["passed"]]
                self._emit(
                    "clip_resynthesis",
                    {
                        "scene_index": index,
                        "provider": provider.name,
                        "attempt": attempt,
                        "failed_checks": failed,
                    },
                )
            result = provider.synthesize(text)
            report = check_clip(result, text_words=words, settings=s, workdir=workdir)
            checks = [asdict(c) for c in report.checks]
            if not report.passed:
                continue
            wer, a5 = self._a5(report, text, run)
            if a5 is not None:
                checks.append(a5)
                if not a5["passed"]:
                    continue
            self._cache.put(key, result)
            a5_checks = checks[len(report.checks) :]
            return self._clip(index, report, provider, out_dir, False, wer, a5_checks)
        raise _ValidationExhausted(index, provider.name, checks)

    def _a5(
        self, report: ClipReport, text: str, run: _Run
    ) -> tuple[float | None, dict[str, Any] | None]:
        with run.lock:
            if self._asr is None or run.asr_skip_reason is not None:
                return None, None
        try:
            transcript = self._asr.transcribe(
                _wav_bytes(report.trimmed_pcm, report.sample_rate), audio_s=report.duration_s
            )
        except BudgetRefused as exc:
            self._skip_asr(run, "budget")
            self._emit("budget_degraded", {"stage": "asr", "reason": BUDGET_REASONS[exc.scope]})
            return None, None
        except ProviderError:
            self._skip_asr(run, "provider_error")
            return None, None
        wer = word_error_rate(text, transcript)
        threshold = self._settings.a5_wer_threshold
        if threshold is None:
            check = {"name": "A5", "passed": True, "value": wer, "threshold": "report-only"}
        else:
            check = {
                "name": "A5",
                "passed": wer <= threshold,
                "value": wer,
                "threshold": f"<={threshold}",
            }
        check["message"] = f"WER {wer:.2f}"
        return wer, check

    @staticmethod
    def _skip_asr(run: _Run, reason: str) -> None:
        with run.lock:
            if run.asr_skip_reason is None:
                run.asr_skip_reason = reason

    def _clip(
        self,
        index: int,
        report: ClipReport,
        provider: TTSProvider,
        out_dir: Path,
        cache_hit: bool,
        wer: float | None,
        extra_checks: list[dict[str, Any]],
    ) -> VoicedClip:
        path = out_dir / f"scene-{index:02d}.wav"
        write_wav(path, report.trimmed_pcm, report.sample_rate)
        return VoicedClip(
            scene_index=index,
            wav_path=path,
            duration_s=report.duration_s,
            provider=provider.name,
            voice=provider.voice_spec().voice,
            cache_hit=cache_hit,
            checks=[asdict(c) for c in report.checks] + extra_checks,
            wer=wer,
        )
