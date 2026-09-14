"""Scripted ASR for tests."""

from collections.abc import Sequence

from app.providers.errors import ProviderError


class FakeASR:
    name = "fake_asr"

    def __init__(self, transcripts: Sequence[str | ProviderError]) -> None:
        self._script = list(transcripts)
        self.calls: list[float] = []

    def transcribe(self, wav_bytes: bytes, *, audio_s: float) -> str:
        self.calls.append(audio_s)
        if not self._script:
            raise AssertionError("FakeASR script exhausted")
        item = self._script.pop(0)
        if isinstance(item, ProviderError):
            raise item
        return item
