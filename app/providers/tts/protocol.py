"""TTS provider protocol (spec §7.1): PCM s16le out, one voice identity per provider instance."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VoiceSpec:
    provider: str
    voice: str
    style_version: str
    model: str | None


@dataclass(frozen=True)
class SynthesisResult:
    pcm: bytes  # s16le
    sample_rate: int
    channels: int
    duration_s: float


class TTSProvider(Protocol):
    name: str
    paid: bool

    def voice_spec(self) -> VoiceSpec: ...

    def synthesize(self, text: str) -> SynthesisResult: ...


def pcm_duration_s(pcm: bytes, sample_rate: int, channels: int = 1) -> float:
    return len(pcm) / (2 * channels * sample_rate)
