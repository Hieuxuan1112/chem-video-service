"""Gemini ASR adapter (spec §7.1): Flash-Lite, temperature 0, verbatim transcription, guarded."""

import math

from google import genai
from google.genai import types

from app.cost.budget import SpendGuard, settle_asr
from app.cost.pricing import project_asr_call
from app.providers.llm.gemini import map_google_error
from app.providers.llm.protocol import LLMResponse, Usage
from app.providers.retry import RetryPolicy, call_with_retry

PROVIDER = "gemini"
ASR_PROMPT = (
    "Transcribe this audio verbatim. Return only the spoken words as plain text, with no"
    " timestamps, speaker labels or commentary."
)


def asr_max_output_tokens(audio_s: float) -> int:
    """Speech is well under 10 text tokens per second; the bound keeps the projection finite."""
    return 64 + math.ceil(audio_s * 10)


class GeminiASRProvider:
    name = "gemini"

    def __init__(
        self, api_key: str, guard: SpendGuard, retry: RetryPolicy, *, model: str, timeout_s: float
    ) -> None:
        self._guard = guard
        self._retry = retry
        self._model = model
        self._secrets = (api_key,)
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000),  # the SDK takes milliseconds
                retry_options=types.HttpRetryOptions(attempts=1),  # RetryPolicy is the only layer
            ),
        )

    def transcribe(self, wav_bytes: bytes, *, audio_s: float) -> str:
        max_output_tokens = asr_max_output_tokens(audio_s)
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents: list[types.PartUnionDict] = [
            types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
            ASR_PROMPT,
        ]

        def call() -> LLMResponse:
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
            except Exception as exc:
                raise map_google_error(exc, secrets=self._secrets) from exc
            metadata = response.usage_metadata
            usage = (
                None
                if metadata is None or metadata.prompt_token_count is None
                else Usage(metadata.prompt_token_count, metadata.candidates_token_count or 0)
            )
            return LLMResponse(text=(response.text or "").strip(), usage=usage, model=self._model)

        response = call_with_retry(
            lambda: self._guard.run(
                operation="asr",
                provider=PROVIDER,
                model=self._model,
                projected_usd=project_asr_call(self._model, audio_s, max_output_tokens),
                call=call,
                settle=settle_asr(self._model, audio_s),
            ),
            policy=self._retry,
        )
        return response.text
