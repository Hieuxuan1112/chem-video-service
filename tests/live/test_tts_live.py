import sqlite3

import numpy as np
import pytest

from app.config import Settings
from app.cost.budget import SpendGuard
from app.cost.ledger import Ledger
from app.providers.retry import RetryPolicy
from app.providers.tts.edge import EdgeTTSProvider
from app.providers.tts.gemini import GeminiTTSProvider
from app.providers.tts.protocol import SynthesisResult

pytestmark = pytest.mark.live

SENTENCE = "The pH scale runs from zero to fourteen, and seven is neutral."


def _check_clip(result: SynthesisResult, s: Settings) -> float:
    samples = np.frombuffer(result.pcm, dtype="<i2")
    assert result.channels == 1 and result.sample_rate == 24_000
    assert len(result.pcm) % 2 == 0 and samples.size > 0
    expected_s = len(SENTENCE.split()) / s.words_per_second
    ratio = result.duration_s / expected_s
    print(
        f"duration={result.duration_s:.2f}s expected={expected_s:.2f}s ratio={ratio:.2f}"
        f" peak={int(np.abs(samples).max())}"
    )
    assert s.a2_min_ratio <= ratio <= s.a2_max_ratio
    return ratio


def _new_rows(ledger: Ledger, last_id: int) -> list[dict[str, object]]:
    conn: sqlite3.Connection = ledger._conn()
    return [
        dict(r)
        for r in conn.execute(
            "SELECT id, operation, provider, model, status, projected_usd, usd, cost_source,"
            " units_json, error_code, error_message FROM cost_entries WHERE id > ? ORDER BY id",
            (last_id,),
        )
    ]


def _last_id(ledger: Ledger) -> int:
    return int(
        ledger._conn().execute("SELECT COALESCE(MAX(id), 0) FROM cost_entries").fetchone()[0]
    )


def test_gemini_tts_clip_live(
    live_settings: Settings, live_guard: SpendGuard, live_ledger: Ledger
) -> None:
    s = live_settings
    assert s.google_api_key is not None
    provider = GeminiTTSProvider(
        s.google_api_key.get_secret_value(),
        live_guard,
        RetryPolicy.from_settings(s),
        model=s.tts_model,
        voice=s.tts_voice,
        style_version=s.tts_style_version,
        timeout_s=s.provider_timeout_s,
        words_per_second=s.words_per_second,
        max_ratio=s.a2_max_ratio,
    )
    last_id = _last_id(live_ledger)
    result = provider.synthesize(SENTENCE)
    rows = _new_rows(live_ledger, last_id)
    for row in rows:
        print("ledger row:", row)
    _check_clip(result, s)
    assert rows[-1]["status"] == "ok"
    assert live_ledger.unsettled() == []


def test_edge_tts_clip_live(live_settings: Settings, live_ledger: Ledger) -> None:
    s = live_settings
    provider = EdgeTTSProvider(
        live_ledger,
        RetryPolicy.from_settings(s),
        voice=s.edge_voice,
        timeout_s=s.provider_timeout_s,
    )
    last_id = _last_id(live_ledger)
    result = provider.synthesize(SENTENCE)
    rows = _new_rows(live_ledger, last_id)
    for row in rows:
        print("ledger row:", row)
    _check_clip(result, s)
    assert [(r["provider"], r["cost_source"], r["usd"]) for r in rows][-1] == ("edge", "free", 0.0)
