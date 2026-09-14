import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from google.genai import errors

from app.config import Settings
from app.cost.budget import SpendGuard
from app.cost.ledger import Ledger
from app.cost.pricing import project_tts_call
from app.jobs.clock import FakeClock
from app.observability.context import job_context, stage_context
from app.providers.errors import ProviderError
from app.providers.retry import RetryPolicy
from app.providers.tts import edge, gemini
from app.providers.tts.edge import EdgeTTSProvider
from app.providers.tts.fake import FakeTTS
from app.providers.tts.gemini import STYLE_PROMPTS, GeminiTTSProvider
from app.providers.tts.protocol import SynthesisResult, VoiceSpec

SETTINGS = Settings(_env_file=None)  # type: ignore[call-arg]
TEXT = "The pH scale runs from zero to fourteen."  # 8 words
SR = 24_000


def _samples(result: SynthesisResult) -> np.ndarray:
    return np.frombuffer(result.pcm, dtype="<i2").astype(np.float64) / 32768.0


def _rows(ledger: Ledger) -> list[dict[str, Any]]:
    return [dict(r) for r in ledger._conn().execute("SELECT * FROM cost_entries ORDER BY id")]


def test_fake_tts_duration_and_modes() -> None:
    boom = ProviderError("server", "503", status=503)
    tts = FakeTTS(script={1: "truncate", 2: "silence", 3: "gap", 4: boom})
    assert tts.paid is False
    assert tts.voice_spec() == VoiceSpec("fake_tts", "fake", "fake-v1", None)

    ok = tts.synthesize(TEXT)
    assert (ok.sample_rate, ok.channels) == (SR, 1)
    assert ok.duration_s == pytest.approx(8 * 0.4)
    assert len(ok.pcm) == round(8 * 0.4 * SR) * 2
    assert np.sqrt(np.mean(_samples(ok) ** 2)) > 0.1

    truncated = tts.synthesize(TEXT)
    assert truncated.duration_s < 0.5 * ok.duration_s

    silence = tts.synthesize(TEXT)
    assert silence.duration_s == pytest.approx(ok.duration_s)
    assert not np.any(_samples(silence))

    gap = tts.synthesize(TEXT)
    samples = _samples(gap)
    middle = samples[len(samples) // 2 - SR // 2 : len(samples) // 2 + SR // 2]
    assert not np.any(middle)  # at least one second of digital silence mid-clip
    assert np.any(samples[: SR // 2])

    with pytest.raises(ProviderError):
        tts.synthesize(TEXT)
    assert tts.calls == [TEXT] * 5
    assert tts.synthesize(TEXT).duration_s == pytest.approx(ok.duration_s)  # unscripted -> ok


class StubModels:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _audio_response(seconds: float, usage: bool = True) -> Any:
    pcm = b"\x10\x00" * round(seconds * SR)
    part = SimpleNamespace(
        inline_data=SimpleNamespace(data=pcm, mime_type="audio/L16;codec=pcm;rate=24000")
    )
    metadata = (
        SimpleNamespace(
            prompt_token_count=20,
            candidates_token_count=round(seconds * 25),
            thoughts_token_count=None,
        )
        if usage
        else None
    )
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))], usage_metadata=metadata
    )


def _gemini_tts(monkeypatch: pytest.MonkeyPatch, models: StubModels, guard: SpendGuard) -> Any:
    clients: list[dict[str, Any]] = []

    class StubClient:
        def __init__(self, **kwargs: Any) -> None:
            clients.append(kwargs)
            self.models = models

    monkeypatch.setattr(gemini.genai, "Client", StubClient)
    provider = GeminiTTSProvider(
        "secret-key",
        guard,
        RetryPolicy(base_s=0.0),
        model=SETTINGS.tts_model,
        voice=SETTINGS.tts_voice,
        style_version=SETTINGS.tts_style_version,
        timeout_s=30.0,
        words_per_second=SETTINGS.words_per_second,
        max_ratio=SETTINGS.a2_max_ratio,
    )
    return provider, clients


def test_gemini_tts_sets_max_output_tokens_from_projection(
    monkeypatch: pytest.MonkeyPatch, db_path: Path, clock: FakeClock
) -> None:
    ledger = Ledger(db_path, clock)
    guard = SpendGuard(ledger, SETTINGS, clock)
    quota = errors.ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    models = StubModels([quota, _audio_response(3.1)])
    provider, clients = _gemini_tts(monkeypatch, models, guard)
    assert provider.paid is True
    assert provider.voice_spec() == VoiceSpec(
        "gemini", SETTINGS.tts_voice, SETTINGS.tts_style_version, SETTINGS.tts_model
    )

    with job_context(job_id="job1", attempt=1, worker_id="w"), stage_context("voicing"):
        result = provider.synthesize(TEXT)

    assert result == SynthesisResult(b"\x10\x00" * round(3.1 * SR), SR, 1, pytest.approx(3.1))
    prompt = f"{STYLE_PROMPTS['teacher-v1']} {TEXT}"
    usd, max_tokens = project_tts_call(SETTINGS.tts_model, len(prompt), 8 / 2.5, 1.6)
    assert max_tokens == math.ceil(8 / 2.5 * 1.6 * 25)
    config = models.calls[1]["config"]
    assert config.max_output_tokens == max_tokens
    assert config.response_modalities == ["AUDIO"]
    assert config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Kore"
    assert config.automatic_function_calling.disable is True
    assert models.calls[1]["contents"] == prompt
    assert clients[0]["http_options"].retry_options.attempts == 1

    rows = _rows(ledger)
    assert [(r["operation"], r["status"], r["stage"], r["job_id"]) for r in rows] == [
        ("tts", "error", "voicing", "job1"),
        ("tts", "ok", "voicing", "job1"),
    ]
    assert rows[1]["projected_usd"] == pytest.approx(usd)
    assert rows[1]["cost_source"] == "actual"
    assert '"audio_tokens": 78' in rows[1]["units_json"]


def test_gemini_tts_missing_audio_is_error(
    monkeypatch: pytest.MonkeyPatch, db_path: Path, clock: FakeClock
) -> None:
    ledger = Ledger(db_path, clock)
    empty = SimpleNamespace(candidates=[], usage_metadata=None)
    provider, _ = _gemini_tts(monkeypatch, StubModels([empty]), SpendGuard(ledger, SETTINGS, clock))
    with pytest.raises(ProviderError) as info:
        provider.synthesize(TEXT)
    assert info.value.kind == "unknown"
    assert [r["status"] for r in _rows(ledger)] == ["error"]


def test_edge_records_free_row(
    monkeypatch: pytest.MonkeyPatch, db_path: Path, clock: FakeClock
) -> None:
    streamed: list[dict[str, Any]] = []

    class StubCommunicate:
        def __init__(self, text: str, voice: str, **kwargs: Any) -> None:
            streamed.append({"text": text, "voice": voice, **kwargs})

        async def stream(self) -> Any:
            yield {"type": "WordBoundary", "offset": 0}
            yield {"type": "audio", "data": b"mp3-part-1"}
            yield {"type": "audio", "data": b"mp3-part-2"}

    decoded: list[bytes] = []

    def fake_decode(mp3: bytes, timeout_s: float) -> bytes:
        decoded.append(mp3)
        return b"\x00\x01" * SR  # one second

    monkeypatch.setattr(edge.edge_tts, "Communicate", StubCommunicate)
    monkeypatch.setattr(edge, "decode_mp3_to_pcm", fake_decode)
    ledger = Ledger(db_path, clock)
    provider = EdgeTTSProvider(ledger, RetryPolicy(), voice="en-US-AriaNeural", timeout_s=20.0)
    assert provider.paid is False
    assert provider.voice_spec() == VoiceSpec("edge", "en-US-AriaNeural", "none", None)

    with job_context(job_id="job9", attempt=2, worker_id="w"), stage_context("voicing"):
        result = provider.synthesize(TEXT)

    assert result == SynthesisResult(b"\x00\x01" * SR, SR, 1, 1.0)
    assert decoded == [b"mp3-part-1mp3-part-2"]
    assert streamed[0]["voice"] == "en-US-AriaNeural" and streamed[0]["text"] == TEXT
    [row] = _rows(ledger)
    assert (row["operation"], row["provider"], row["status"]) == ("tts", "edge", "ok")
    assert (row["cost_source"], row["usd"], row["billable"]) == ("free", 0.0, 1)
    assert (row["job_id"], row["attempt"], row["stage"]) == ("job9", 2, "voicing")
    assert '"characters": 40' in row["units_json"]


def test_edge_failures_are_retried_and_recorded(
    monkeypatch: pytest.MonkeyPatch, db_path: Path, clock: FakeClock
) -> None:
    attempts: list[int] = []

    class FlakyCommunicate:
        def __init__(self, text: str, voice: str, **kwargs: Any) -> None:
            pass

        async def stream(self) -> Any:
            attempts.append(1)
            if len(attempts) == 1:
                raise edge.edge_tts.exceptions.NoAudioReceived("no audio")
            yield {"type": "audio", "data": b"mp3"}

    monkeypatch.setattr(edge.edge_tts, "Communicate", FlakyCommunicate)
    monkeypatch.setattr(edge, "decode_mp3_to_pcm", lambda mp3, timeout_s: b"\x00\x00" * 2400)
    monkeypatch.setattr(edge.time, "sleep", lambda _: None)
    ledger = Ledger(db_path, clock)
    provider = EdgeTTSProvider(ledger, RetryPolicy(base_s=0.0), voice="v", timeout_s=5.0)

    assert provider.synthesize(TEXT).duration_s == pytest.approx(0.1)
    rows = _rows(ledger)
    assert [(r["status"], r["error_code"]) for r in rows] == [("error", "server"), ("ok", None)]
    assert "no audio" in rows[0]["error_message"]


def test_paid_flags() -> None:
    assert GeminiTTSProvider.paid is True
    assert EdgeTTSProvider.paid is False
    assert FakeTTS().paid is False
    assert FakeTTS(paid=True).paid is True


@pytest.mark.integration
def test_decode_mp3_to_pcm_with_ffmpeg(tmp_path: Path) -> None:
    import subprocess

    mp3 = tmp_path / "tone.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=1",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(mp3),
        ],
        check=True,
    )
    pcm = edge.decode_mp3_to_pcm(mp3.read_bytes(), timeout_s=20.0)
    assert abs(len(pcm) / 2 / SR - 1.0) < 0.1  # 24 kHz mono s16le, encoder padding tolerated
    with pytest.raises(ProviderError):
        edge.decode_mp3_to_pcm(b"not an mp3", timeout_s=20.0)
