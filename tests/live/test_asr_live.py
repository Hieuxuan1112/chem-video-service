import io
import re
import wave

import pytest

from app.config import Settings
from app.cost.budget import SpendGuard
from app.cost.ledger import Ledger
from app.providers.asr.gemini import GeminiASRProvider
from app.providers.retry import RetryPolicy
from app.providers.tts.edge import EdgeTTSProvider

pytestmark = pytest.mark.live

SENTENCE = "The pH scale runs from zero to fourteen, and seven is neutral."


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def test_asr_live(live_settings: Settings, live_guard: SpendGuard, live_ledger: Ledger) -> None:
    """Loose smoke check. Word recall stands in for WER, which arrives with Task 6.1."""
    s = live_settings
    assert s.google_api_key is not None
    policy = RetryPolicy.from_settings(s)
    clip = EdgeTTSProvider(
        live_ledger, policy, voice=s.edge_voice, timeout_s=s.provider_timeout_s
    ).synthesize(SENTENCE)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(clip.channels)
        wav.setsampwidth(2)
        wav.setframerate(clip.sample_rate)
        wav.writeframes(clip.pcm)
    asr = GeminiASRProvider(
        s.google_api_key.get_secret_value(),
        live_guard,
        policy,
        model=s.asr_model,
        timeout_s=s.provider_timeout_s,
    )
    before = live_ledger.project_totals().committed_usd

    transcript = asr.transcribe(buffer.getvalue(), audio_s=clip.duration_s)

    delta = live_ledger.project_totals().committed_usd - before
    reference = _words(SENTENCE)
    heard = set(_words(transcript))
    recall = sum(1 for w in reference if w in heard) / len(reference)
    print(f"audio={clip.duration_s:.2f}s transcript={transcript!r} recall={recall:.2f}")
    print(f"ledger delta=${delta:.6f}")
    assert recall >= 0.5
    assert live_ledger.unsettled() == []
