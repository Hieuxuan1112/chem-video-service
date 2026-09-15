import json
from pathlib import Path

import pytest

from app.config import Settings
from app.cost.budget import SpendGuard
from app.cost.ledger import Ledger
from app.pipeline.tts_cache import TTSCache
from app.pipeline.voicing import Voicer, spoken_text
from app.providers.asr.gemini import GeminiASRProvider
from app.providers.retry import RetryPolicy
from app.providers.tts.edge import EdgeTTSProvider
from app.providers.tts.gemini import GeminiTTSProvider
from app.topics.registry import TOPICS_ROOT, TopicRegistry

pytestmark = pytest.mark.live


def test_voice_curated_ph_storyboard_live(
    tmp_path: Path, live_settings: Settings, live_guard: SpendGuard, live_ledger: Ledger
) -> None:
    s = live_settings
    assert s.google_api_key is not None
    key = s.google_api_key.get_secret_value()
    policy = RetryPolicy.from_settings(s)
    primary = GeminiTTSProvider(
        key,
        live_guard,
        policy,
        model=s.tts_model,
        voice=s.tts_voice,
        style_version=s.tts_style_version,
        timeout_s=s.provider_timeout_s,
        words_per_second=s.words_per_second,
        max_ratio=s.a2_max_ratio,
    )
    fallback = EdgeTTSProvider(
        live_ledger, policy, voice=s.edge_voice, timeout_s=s.provider_timeout_s
    )
    asr = GeminiASRProvider(
        key, live_guard, policy, model=s.asr_model, timeout_s=s.provider_timeout_s
    )
    events: list[tuple[str, dict[str, object]]] = []
    # A fresh cache in tmp: a cache hit would spend nothing and measure nothing.
    voicer = Voicer(
        primary, fallback, asr, TTSCache(tmp_path / "cache"), live_guard, s,
        emit=lambda t, d: events.append((t, d)),
    )  # fmt: skip
    registry = TopicRegistry.load(TOPICS_ROOT)
    storyboard = registry.curated_storyboard("ph_scale")
    pack = registry.get("ph_scale")
    conn = live_ledger._conn()
    last_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM cost_entries").fetchone()[0]
    before = live_ledger.project_totals().committed_usd

    outcome = voicer.voice(storyboard=storyboard, pack=pack, out_dir=tmp_path / "voicing")

    for clip in outcome.clips:
        words = len(
            spoken_text(storyboard.scenes[clip.scene_index].narration, pack.pronunciations).split()
        )
        a2 = next(c for c in clip.checks if c["name"] == "A2")
        print(
            f"scene {clip.scene_index}: provider={clip.provider} duration={clip.duration_s:.2f}s"
            f" words={words} A2={a2['value']:.2f} wer={clip.wer} cache_hit={clip.cache_hit}"
            f" checks={[(c['name'], c['passed']) for c in clip.checks]}"
        )
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT id, operation, provider, model, status, projected_usd, usd, cost_source,"
            " units_json, error_code, error_message FROM cost_entries WHERE id > ? ORDER BY id",
            (last_id,),
        )
    ]
    audio_tokens = audio_seconds = 0.0
    for row in rows:
        print("ledger row:", json.dumps(row))
        units = json.loads(str(row["units_json"]))
        if row["operation"] == "tts" and row["provider"] == "gemini" and "audio_tokens" in units:
            audio_tokens += units["audio_tokens"]
            audio_seconds += units["audio_seconds"]
    delta = live_ledger.project_totals().committed_usd - before
    rate = audio_tokens / audio_seconds if audio_seconds else None
    print(
        f"provider={outcome.provider} voice={outcome.voice}"
        f" fallback_reason={outcome.voice_fallback_reason} asr_skipped={outcome.asr_skipped}"
        f" ({outcome.asr_skip_reason}) a5_mode={outcome.a5_mode}"
    )
    print(f"gemini audio tokens/s (untrimmed clips) = {rate}")
    print(f"provider_calls={len(rows)} ledger_delta=${delta:.6f}")
    print("events:", [t for t, _ in events])

    assert len(outcome.clips) == len(storyboard.scenes)
    assert live_ledger.unsettled() == []
    assert delta < 0.05
