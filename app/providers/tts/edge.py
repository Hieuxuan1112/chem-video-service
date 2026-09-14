"""edge-tts fallback adapter (spec §7.1): a different vendor, free, MP3 decoded to PCM by ffmpeg."""

import asyncio
import subprocess
import time

import aiohttp
import edge_tts

from app.cost.ledger import Ledger
from app.observability.context import attempt_var, job_id_var, stage_var
from app.providers.errors import ProviderError
from app.providers.retry import RetryPolicy, call_with_retry
from app.providers.tts.protocol import SynthesisResult, VoiceSpec, pcm_duration_s

PROVIDER = "edge"
SAMPLE_RATE = 24_000
_STDERR_TAIL = 500


def decode_mp3_to_pcm(mp3: bytes, *, timeout_s: float) -> bytes:
    """MP3 -> PCM s16le 24 kHz mono through an ffmpeg pipe."""
    args = [
        "ffmpeg", "-v", "error", "-i", "pipe:0",
        "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1",
    ]  # fmt: skip
    try:
        proc = subprocess.run(args, input=mp3, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise ProviderError("timeout", "ffmpeg MP3 decode timed out", provider=PROVIDER) from exc
    if proc.returncode != 0 or not proc.stdout:
        tail = proc.stderr.decode("utf-8", "replace")[-_STDERR_TAIL:]
        raise ProviderError("unknown", f"ffmpeg MP3 decode failed: {tail}", provider=PROVIDER)
    return proc.stdout


def _map_error(exc: Exception) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    message = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, TimeoutError):
        return ProviderError("timeout", message, provider=PROVIDER)
    if isinstance(exc, edge_tts.exceptions.EdgeTTSException | aiohttp.ClientError):
        return ProviderError("server", message, provider=PROVIDER)
    return ProviderError("unknown", message, provider=PROVIDER)


class EdgeTTSProvider:
    name = "edge"
    paid = False

    def __init__(self, ledger: Ledger, retry: RetryPolicy, *, voice: str, timeout_s: float) -> None:
        self._ledger = ledger
        self._retry = retry
        self._voice = voice
        self._timeout_s = timeout_s

    def voice_spec(self) -> VoiceSpec:
        return VoiceSpec(PROVIDER, self._voice, "none", None)

    async def _stream(self, text: str) -> bytes:
        # stream() rather than stream_sync(): edge-tts 7.2.8 stream_sync hangs if the stream raises.
        communicate = edge_tts.Communicate(text, self._voice)
        chunks = [c["data"] async for c in communicate.stream() if c["type"] == "audio"]
        return b"".join(chunks)

    def _attempt(self, text: str) -> SynthesisResult:
        units: dict[str, object] = {"characters": len(text)}
        try:
            mp3 = asyncio.run(asyncio.wait_for(self._stream(text), self._timeout_s))
            if not mp3:
                raise ProviderError("server", "edge-tts returned no audio", provider=PROVIDER)
            pcm = decode_mp3_to_pcm(mp3, timeout_s=self._timeout_s)
        except Exception as exc:
            err = _map_error(exc)
            self._record(units, "error", err)
            raise err from exc
        result = SynthesisResult(pcm, SAMPLE_RATE, 1, pcm_duration_s(pcm, SAMPLE_RATE))
        units["audio_seconds"] = result.duration_s
        self._record(units, "ok", None)
        return result

    def _record(self, units: dict[str, object], status: str, err: ProviderError | None) -> None:
        self._ledger.record_free(
            operation="tts",
            provider=PROVIDER,
            model=None,
            units=units,
            job_id=job_id_var.get(),
            attempt=attempt_var.get(),
            stage=stage_var.get(),
            status=status,
            error_code=None if err is None else err.kind,
            error_message=None if err is None else err.message[:1000],
        )

    def synthesize(self, text: str) -> SynthesisResult:
        return call_with_retry(lambda: self._attempt(text), policy=self._retry, sleep=time.sleep)
