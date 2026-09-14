"""Gemini structured-output adapter. The SDK, its errors and its schema quirks stay here."""

from collections.abc import Sequence
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from app.cost.budget import SpendGuard, settle_text
from app.cost.pricing import project_text_call
from app.providers.errors import ProviderError
from app.providers.llm.protocol import LLMRequest, LLMResponse, Usage
from app.providers.retry import RetryPolicy, call_with_retry

PROVIDER = "gemini"


def _redact(text: str, secrets: Sequence[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def map_google_error(exc: Exception, *, secrets: Sequence[str] = ()) -> ProviderError:
    """The provider's own message is kept (e.g. why a schema was rejected); secrets are redacted."""
    if isinstance(exc, errors.APIError):
        code = exc.code
        message = _redact(
            f"{code} {exc.status}: {exc.message}" if exc.message else str(exc), secrets
        )
        if code == 429:
            return ProviderError("rate_limited", message, status=code, provider=PROVIDER)
        if 500 <= code < 600:
            return ProviderError("server", message, status=code, provider=PROVIDER)
        if code in (401, 403):
            return ProviderError("auth", message, status=code, provider=PROVIDER)
        if code == 400:
            return ProviderError("invalid_request", message, status=code, provider=PROVIDER)
        return ProviderError("unknown", message, status=code, provider=PROVIDER)
    if isinstance(exc, httpx.TimeoutException | TimeoutError):
        message = _redact(str(exc) or type(exc).__name__, secrets)
        return ProviderError("timeout", message, provider=PROVIDER)
    return ProviderError(
        "unknown", _redact(f"{type(exc).__name__}: {exc}", secrets), provider=PROVIDER
    )


def _usage(metadata: Any) -> Usage | None:
    if metadata is None or metadata.prompt_token_count is None:
        return None
    return Usage(
        input_tokens=metadata.prompt_token_count,
        output_tokens=metadata.candidates_token_count or 0,
        thinking_tokens=metadata.thoughts_token_count or 0,
    )


class GeminiLLMProvider:
    name = "gemini"

    def __init__(
        self, api_key: str, guard: SpendGuard, retry: RetryPolicy, *, timeout_s: float
    ) -> None:
        self._guard = guard
        self._retry = retry
        self._secrets = (api_key,)
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000),  # the SDK takes milliseconds
                retry_options=types.HttpRetryOptions(attempts=1),  # RetryPolicy is the only layer
            ),
        )

    def generate_json(self, req: LLMRequest) -> LLMResponse:
        prompt_chars = len(req.system) + len(req.prompt)
        projected = project_text_call(
            req.model, prompt_chars, req.max_output_tokens, req.thinking_budget
        )
        config = types.GenerateContentConfig(
            system_instruction=req.system,
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=req.json_schema,
            thinking_config=types.ThinkingConfig(thinking_budget=req.thinking_budget),
            # no tools are passed; disabling AFC also silences the SDK's AFC warning
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        def call() -> LLMResponse:
            try:
                response = self._client.models.generate_content(
                    model=req.model, contents=req.prompt, config=config
                )
            except Exception as exc:
                raise map_google_error(exc, secrets=self._secrets) from exc
            return LLMResponse(
                text=response.text or "", usage=_usage(response.usage_metadata), model=req.model
            )

        # Retry outside the guard: every attempt is reserved and settled on its own.
        return call_with_retry(
            lambda: self._guard.run(
                operation=req.operation,
                provider=PROVIDER,
                model=req.model,
                projected_usd=projected,
                call=call,
                settle=settle_text(req.model, prompt_chars),
            ),
            policy=self._retry,
        )
