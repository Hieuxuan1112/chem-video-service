"""ASR provider protocol (spec §7.1), used for A5 reverse-ASR verification."""

from typing import Protocol


class ASRProvider(Protocol):
    name: str

    def transcribe(self, wav_bytes: bytes, *, audio_s: float) -> str: ...
