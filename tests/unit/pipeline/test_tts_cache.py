from dataclasses import replace
from pathlib import Path

from app.pipeline.tts_cache import TTSCache, tts_cache_key
from app.providers.tts.protocol import SynthesisResult, VoiceSpec

SPEC = VoiceSpec("gemini", "Kore", "teacher-v1", "gemini-2.5-flash-preview-tts")
TEXT = "The P H scale runs from zero to fourteen."


def test_key_changes_with_each_component() -> None:
    base = tts_cache_key(SPEC, TEXT)
    assert base == tts_cache_key(VoiceSpec(*SPEC.__dict__.values()), TEXT)  # deterministic
    assert len(base) == 64 and all(c in "0123456789abcdef" for c in base)
    variants = [
        tts_cache_key(replace(SPEC, provider="edge"), TEXT),
        tts_cache_key(replace(SPEC, model=None), TEXT),
        tts_cache_key(replace(SPEC, voice="Puck"), TEXT),
        tts_cache_key(replace(SPEC, style_version="teacher-v2"), TEXT),
        tts_cache_key(SPEC, TEXT + " "),
    ]
    assert len({base, *variants}) == 6


def test_roundtrip(tmp_path: Path) -> None:
    cache = TTSCache(tmp_path)
    key = tts_cache_key(SPEC, TEXT)
    assert cache.get(key) is None
    result = SynthesisResult(b"\x01\x00\xff\x7f" * 12_000, 24_000, 1, 1.0)
    cache.put(key, result)
    path = tmp_path / "tts" / key[:2] / f"{key}.wav"
    assert path.is_file()
    assert list(path.parent.iterdir()) == [path]  # no temp file left behind
    assert cache.get(key) == result
    cache.put(key, result)  # overwriting is safe
    assert cache.get(key) == result


def test_corrupt_file_treated_as_miss(tmp_path: Path) -> None:
    cache = TTSCache(tmp_path)
    key = tts_cache_key(SPEC, TEXT)
    path = tmp_path / "tts" / key[:2] / f"{key}.wav"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"RIFF\x00\x00garbage")
    assert cache.get(key) is None
    path.write_bytes(b"")
    assert cache.get(key) is None
