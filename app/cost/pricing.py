"""Prices, upper-bound projections and settled costs (spec §9.3).

Prices were read from the Gemini pricing page on 2026-09-13 and must be re-verified on the
official page before they are published in docs/COST.md.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

PRICING_VERSION = "2026-09-13"

GEMINI_PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"


@dataclass(frozen=True)
class ModelPrice:
    model: str
    text_input_per_m: float
    output_per_m: float
    audio_input_per_m: float | None
    audio_output_per_m: float | None
    source_url: str
    verified_on: str | None
    note: str = ""


PRICES: Mapping[str, ModelPrice] = {
    "gemini-2.5-flash": ModelPrice(
        model="gemini-2.5-flash",
        text_input_per_m=0.30,
        output_per_m=2.50,
        audio_input_per_m=None,
        audio_output_per_m=None,
        source_url=GEMINI_PRICING_URL,
        verified_on="2026-09-13",
        note="Read from a page summary; re-verify before publication. Audio input not used.",
    ),
    "gemini-2.5-flash-lite": ModelPrice(
        model="gemini-2.5-flash-lite",
        text_input_per_m=0.10,
        output_per_m=0.40,
        audio_input_per_m=None,
        audio_output_per_m=None,
        source_url=GEMINI_PRICING_URL,
        verified_on=None,
        note="Audio input price not captured; ASR projections use a labelled placeholder.",
    ),
    "gemini-2.5-flash-preview-tts": ModelPrice(
        model="gemini-2.5-flash-preview-tts",
        text_input_per_m=0.50,
        output_per_m=10.00,
        audio_input_per_m=None,
        audio_output_per_m=10.00,
        source_url=GEMINI_PRICING_URL,
        verified_on="2026-09-13",
        note="Read from a page summary; re-verify before publication.",
    ),
}

AUDIO_INPUT_PLACEHOLDER_PER_M = 1.00  # a deliberately high placeholder, not a price
TTS_AUDIO_TOKENS_PER_S = 25  # single 5.5 s sample (spec §2); re-measured in Task 6.5
AUDIO_INPUT_TOKENS_PER_S = 32  # Gemini docs

CLOUD_RUN_VCPU_S = 0.000018
CLOUD_RUN_GIB_S = 0.000002
CLOUD_RUN_SOURCE_URL = "https://cloud.google.com/run/pricing"
CLOUD_RUN_VERIFIED_ON: str | None = None  # secondary source; re-verify before publication

_M = 1_000_000


def estimate_text_tokens(chars: int) -> int:
    return math.ceil(chars / 4)


def _audio_input_per_m(price: ModelPrice) -> float:
    if price.audio_input_per_m is None:
        return AUDIO_INPUT_PLACEHOLDER_PER_M
    return price.audio_input_per_m


def _audio_output_per_m(price: ModelPrice) -> float:
    if price.audio_output_per_m is None:
        return price.output_per_m
    return price.audio_output_per_m


def project_text_call(
    model: str, prompt_chars: int, max_output_tokens: int, thinking_budget: int
) -> float:
    """Thinking tokens are added on top of max_output_tokens so the bound holds either way."""
    p = PRICES[model]
    return (
        estimate_text_tokens(prompt_chars) * p.text_input_per_m
        + (max_output_tokens + thinking_budget) * p.output_per_m
    ) / _M


def project_tts_call(
    model: str, text_chars: int, expected_audio_s: float, max_ratio: float
) -> tuple[float, int]:
    p = PRICES[model]
    max_output_tokens = math.ceil(expected_audio_s * max_ratio * TTS_AUDIO_TOKENS_PER_S)
    usd = (
        estimate_text_tokens(text_chars) * p.text_input_per_m
        + max_output_tokens * _audio_output_per_m(p)
    ) / _M
    return usd, max_output_tokens


def project_asr_call(model: str, audio_s: float, max_output_tokens: int) -> float:
    p = PRICES[model]
    audio_tokens = math.ceil(audio_s * AUDIO_INPUT_TOKENS_PER_S)
    return (audio_tokens * _audio_input_per_m(p) + max_output_tokens * p.output_per_m) / _M


def text_call_cost(
    model: str, input_tokens: int, output_tokens: int, thinking_tokens: int
) -> float:
    p = PRICES[model]
    return (
        input_tokens * p.text_input_per_m + (output_tokens + thinking_tokens) * p.output_per_m
    ) / _M


def tts_call_cost(model: str, text_tokens: int, audio_tokens: int) -> float:
    p = PRICES[model]
    return (text_tokens * p.text_input_per_m + audio_tokens * _audio_output_per_m(p)) / _M


def asr_call_cost(
    model: str, audio_tokens: int, text_tokens_in: int, output_tokens: int
) -> tuple[float, Literal["actual", "estimated"]]:
    """The cost is only an estimate while the audio input price is a placeholder."""
    p = PRICES[model]
    usd = (
        audio_tokens * _audio_input_per_m(p)
        + text_tokens_in * p.text_input_per_m
        + output_tokens * p.output_per_m
    ) / _M
    return usd, "actual" if p.audio_input_per_m is not None else "estimated"


def render_compute_estimate(cpu_seconds: float, vcpu: int = 2, gib: float = 4.0) -> float:
    """Production estimate only (billable=0): nothing is spent when rendering locally."""
    return cpu_seconds * (vcpu * CLOUD_RUN_VCPU_S + gib * CLOUD_RUN_GIB_S)
