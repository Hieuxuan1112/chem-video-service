"""Gemini TTS adapter (spec §7.1): paid, guarded, PCM s16le 24 kHz mono."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from app.cost.budget import SpendGuard, settle_tts
from app.cost.pricing import project_tts_call
from app.providers.errors import ProviderError
from app.providers.llm.gemini import map_google_error
from app.providers.llm.protocol import Usage
from app.providers.retry import RetryPolicy, call_with_retry
from app.providers.tts.protocol import SynthesisResult, VoiceSpec, pcm_duration_s

PROVIDER = "gemini"
DEFAULT_SAMPLE_RATE = 24_000

STYLE_PROMPTS: Mapping[str, str] = {
    "teacher-v1": "Read aloud in a warm, clear, encouraging teacher voice at a moderate pace:",
}


@dataclass(frozen=True)
class _Clip:
    """What settle_tts reads: usage plus the clip duration for the estimate path."""

    result: SynthesisResult
    usage: Usage | None

    @property
    def duration_s(self) -> float:
        return self.result.duration_s


def _audio(response: Any) -> tuple[bytes, int]:
    for candidate in response.candidates or []:
        content = candidate.content
        for part in (content.parts if content is not None else None) or []:
            inline = part.inline_data
            if inline is not None and inline.data:
                match = re.search(r"rate=(\d+)", inline.mime_type or "")
                return inline.data, int(match.group(1)) if match else DEFAULT_SAMPLE_RATE
    raise ProviderError("unknown", "Gemini TTS response contained no audio", provider=PROVIDER)


class GeminiTTSProvider:
    name = "gemini"
    paid = True

    def __init__(
        self,
        api_key: str,
        guard: SpendGuard,
        retry: RetryPolicy,
        *,
        model: str,
        voice: str,
        style_version: str,
        timeout_s: float,
        words_per_second: float,
        max_ratio: float,
    ) -> None:
        self._guard = guard
        self._retry = retry
        self._model = model
        self._voice = voice
        self._style_version = style_version
        self._style = STYLE_PROMPTS[style_version]
        self._words_per_second = words_per_second
        self._max_ratio = max_ratio
        self._secrets = (api_key,)
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000),  # the SDK takes milliseconds
                retry_options=types.HttpRetryOptions(attempts=1),  # RetryPolicy is the only layer
            ),
        )

    def voice_spec(self) -> VoiceSpec:
        return VoiceSpec(PROVIDER, self._voice, self._style_version, self._model)

    def synthesize(self, text: str) -> SynthesisResult:
        prompt = f"{self._style} {text}"
        expected_s = len(text.split()) / self._words_per_second
        projected, max_output_tokens = project_tts_call(
            self._model, len(prompt), expected_s, self._max_ratio
        )
        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self._voice)
                )
            ),
            max_output_tokens=max_output_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        def call() -> _Clip:
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=prompt, config=config
                )
            except Exception as exc:
                raise map_google_error(exc, secrets=self._secrets) from exc
            pcm, rate = _audio(response)
            metadata = response.usage_metadata
            usage = (
                None
                if metadata is None or metadata.prompt_token_count is None
                else Usage(metadata.prompt_token_count, metadata.candidates_token_count or 0)
            )
            return _Clip(SynthesisResult(pcm, rate, 1, pcm_duration_s(pcm, rate)), usage)

        clip = call_with_retry(
            lambda: self._guard.run(
                operation="tts",
                provider=PROVIDER,
                model=self._model,
                projected_usd=projected,
                call=call,
                settle=settle_tts(self._model, len(prompt)),
            ),
            policy=self._retry,
        )
        return clip.result
