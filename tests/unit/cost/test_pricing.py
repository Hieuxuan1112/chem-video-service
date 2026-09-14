import math

from app.cost import pricing
from app.cost.pricing import (
    AUDIO_INPUT_PLACEHOLDER_PER_M,
    PRICES,
    PRICING_VERSION,
    asr_call_cost,
    estimate_text_tokens,
    project_asr_call,
    project_text_call,
    project_tts_call,
    render_compute_estimate,
    text_call_cost,
    tts_call_cost,
)

FLASH = "gemini-2.5-flash"
LITE = "gemini-2.5-flash-lite"
TTS = "gemini-2.5-flash-preview-tts"


def test_pricing_entries_have_source_and_version() -> None:
    assert PRICING_VERSION == "2026-09-13"
    assert set(PRICES) == {FLASH, LITE, TTS}
    for name, price in PRICES.items():
        assert price.model == name
        assert price.source_url.startswith("https://")
        assert price.text_input_per_m > 0
        assert price.output_per_m > 0
    assert (PRICES[FLASH].text_input_per_m, PRICES[FLASH].output_per_m) == (0.30, 2.50)
    assert (PRICES[LITE].text_input_per_m, PRICES[LITE].output_per_m) == (0.10, 0.40)
    assert (PRICES[TTS].text_input_per_m, PRICES[TTS].audio_output_per_m) == (0.50, 10.00)


def test_unverified_prices_flagged() -> None:
    lite = PRICES[LITE]
    assert lite.audio_input_per_m is None
    assert lite.verified_on is None
    assert "placeholder" in lite.note.lower()
    assert AUDIO_INPUT_PLACEHOLDER_PER_M == 1.00
    assert pricing.CLOUD_RUN_VERIFIED_ON is None
    assert pricing.CLOUD_RUN_SOURCE_URL.startswith("https://")
    # ASR cost can only be an estimate while the audio input price is unknown
    _, source = asr_call_cost(LITE, audio_tokens=1000, text_tokens_in=10, output_tokens=50)
    assert source == "estimated"


def test_projection_upper_bounds_actual_for_max_tokens() -> None:
    chars, max_out, thinking = 6000, 4096, 1024
    projected = project_text_call(FLASH, chars, max_out, thinking)
    est_in = estimate_text_tokens(chars)
    assert est_in == math.ceil(chars / 4)
    for out in (0, 100, max_out):
        for think in (0, thinking):
            assert text_call_cost(FLASH, est_in, out, think) <= projected
    # thinking is priced on top of max_output_tokens, not inside it
    assert projected > project_text_call(FLASH, chars, max_out, 0)

    usd, max_audio_tokens = project_tts_call(TTS, 300, expected_audio_s=12.0, max_ratio=1.6)
    assert max_audio_tokens == math.ceil(12.0 * 1.6 * 25)
    assert tts_call_cost(TTS, estimate_text_tokens(300), max_audio_tokens) <= usd

    asr_projected = project_asr_call(LITE, audio_s=12.0, max_output_tokens=256)
    actual, _ = asr_call_cost(
        LITE, audio_tokens=math.ceil(12.0 * 32), text_tokens_in=0, output_tokens=256
    )
    assert actual <= asr_projected


def test_render_compute_estimate() -> None:
    expected = 100 * (2 * pricing.CLOUD_RUN_VCPU_S + 4.0 * pricing.CLOUD_RUN_GIB_S)
    assert math.isclose(render_compute_estimate(100.0), expected)
