"""Content-addressed TTS cache (spec §7.2). Callers re-validate cached clips with A1-A4."""

import hashlib
import json
import os
import tempfile
import wave
from pathlib import Path

from app.pipeline.audio_checks import write_wav
from app.providers.tts.protocol import SynthesisResult, VoiceSpec, pcm_duration_s


def tts_cache_key(spec: VoiceSpec, spoken_text: str) -> str:
    payload = [spec.provider, spec.model, spec.voice, spec.style_version, spoken_text]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


class TTSCache:
    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, key: str) -> Path:
        return self._root / "tts" / key[:2] / f"{key}.wav"

    def get(self, key: str) -> SynthesisResult | None:
        path = self._path(key)
        try:
            with wave.open(str(path)) as w:
                if w.getsampwidth() != 2:
                    return None
                pcm = w.readframes(w.getnframes())
                rate, channels = w.getframerate(), w.getnchannels()
        except (OSError, EOFError, wave.Error):
            return None
        return SynthesisResult(pcm, rate, channels, pcm_duration_s(pcm, rate, channels))

    def put(self, key: str, result: SynthesisResult) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp-wav")
        os.close(fd)
        try:
            write_wav(Path(tmp), result.pcm, result.sample_rate, result.channels)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
