"""Deterministic TTS for tests: a 220 Hz tone, 0.4 s per word, with scriptable failure modes."""

from collections.abc import Mapping

import numpy as np

from app.providers.errors import ProviderError
from app.providers.tts.protocol import SynthesisResult, VoiceSpec

SAMPLE_RATE = 24_000
SECONDS_PER_WORD = 0.4
_AMPLITUDE = 0.3
_GAP_S = 2.0


class FakeTTS:
    def __init__(
        self,
        *,
        name: str = "fake_tts",
        script: Mapping[int, object] | None = None,
        paid: bool = False,
    ) -> None:
        """script maps call index to "ok" | "truncate" | "silence" | "gap" | ProviderError."""
        self.name = name
        self.paid = paid
        self._script = dict(script or {})
        self.calls: list[str] = []

    def voice_spec(self) -> VoiceSpec:
        return VoiceSpec(self.name, "fake", "fake-v1", None)

    def synthesize(self, text: str) -> SynthesisResult:
        mode = self._script.get(len(self.calls), "ok")
        self.calls.append(text)
        if isinstance(mode, ProviderError):
            raise mode
        n = round(len(text.split()) * SECONDS_PER_WORD * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE
        samples = _AMPLITUDE * np.sin(2 * np.pi * 220.0 * t)
        if mode == "truncate":
            samples = samples[: n // 4]
        elif mode == "silence":
            samples = np.zeros(n)
        elif mode == "gap":
            gap = round(_GAP_S * SAMPLE_RATE)
            mid = n // 2
            samples = np.concatenate([samples[:mid], np.zeros(gap), samples[mid:]])
        elif mode != "ok":
            raise ValueError(f"unknown FakeTTS mode {mode!r}")
        pcm = (samples * 32767).astype("<i2").tobytes()
        return SynthesisResult(pcm, SAMPLE_RATE, 1, len(samples) / SAMPLE_RATE)
