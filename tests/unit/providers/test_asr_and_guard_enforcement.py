import inspect
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import Settings
from app.cost.budget import SpendGuard
from app.cost.ledger import Ledger
from app.cost.pricing import project_asr_call
from app.jobs.clock import FakeClock
from app.providers.asr import gemini
from app.providers.asr.fake import FakeASR
from app.providers.asr.gemini import ASR_PROMPT, GeminiASRProvider, asr_max_output_tokens
from app.providers.errors import ProviderError
from app.providers.llm.gemini import GeminiLLMProvider
from app.providers.paid import PAID_ADAPTERS
from app.providers.retry import RetryPolicy
from app.providers.tts.gemini import GeminiTTSProvider

REPO = Path(__file__).resolve().parents[3]
SETTINGS = Settings(_env_file=None)  # type: ignore[call-arg]
_GENAI_IMPORT = re.compile(r"^\s*(from\s+google(\.genai)?(\.\w+)*\s+import|import\s+google)", re.M)


def test_every_paid_adapter_requires_guard() -> None:
    assert set(PAID_ADAPTERS) == {GeminiLLMProvider, GeminiTTSProvider, GeminiASRProvider}
    for cls in PAID_ADAPTERS:
        param = inspect.signature(cls.__init__).parameters["guard"]
        assert param.default is inspect.Parameter.empty, cls
        assert param.annotation is SpendGuard, cls
        # the paid call itself goes through the guard
        assert "self._guard.run(" in inspect.getsource(cls), cls


def test_no_module_imports_genai_outside_adapters() -> None:
    allowed = {
        Path("app/providers/llm/gemini.py"),
        Path("app/providers/tts/gemini.py"),
        Path("app/providers/asr/gemini.py"),
    }
    importing = {
        path.relative_to(REPO)
        for path in (REPO / "app").rglob("*.py")
        if _GENAI_IMPORT.search(path.read_text(encoding="utf-8"))
        and "genai" in path.read_text(encoding="utf-8")
    }
    assert importing == allowed


def test_fake_asr() -> None:
    err = ProviderError("timeout", "slow")
    asr = FakeASR(["the ph scale", err])
    assert asr.name == "fake_asr"
    assert asr.transcribe(b"RIFF", audio_s=2.0) == "the ph scale"
    with pytest.raises(ProviderError):
        asr.transcribe(b"RIFF", audio_s=1.0)
    assert asr.calls == [2.0, 1.0]
    with pytest.raises(AssertionError):
        asr.transcribe(b"RIFF", audio_s=1.0)


def test_gemini_asr_adapter_guarded_and_estimated(
    monkeypatch: pytest.MonkeyPatch, db_path: Path, clock: FakeClock
) -> None:
    calls: list[dict[str, Any]] = []

    class StubModels:
        def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
            calls.append({"model": model, "contents": contents, "config": config})
            usage = SimpleNamespace(
                prompt_token_count=180, candidates_token_count=12, thoughts_token_count=None
            )
            return SimpleNamespace(text=" The pH scale runs from 0 to 14. ", usage_metadata=usage)

    class StubClient:
        def __init__(self, **_: Any) -> None:
            self.models = StubModels()

    monkeypatch.setattr(gemini.genai, "Client", StubClient)
    ledger = Ledger(db_path, clock)
    asr = GeminiASRProvider(
        "k",
        SpendGuard(ledger, SETTINGS, clock),
        RetryPolicy(),
        model=SETTINGS.asr_model,
        timeout_s=9,
    )

    assert asr.transcribe(b"RIFFwav", audio_s=5.5) == "The pH scale runs from 0 to 14."

    [call] = calls
    audio, prompt = call["contents"]
    assert (audio.inline_data.mime_type, audio.inline_data.data) == ("audio/wav", b"RIFFwav")
    assert prompt == ASR_PROMPT
    config = call["config"]
    assert config.temperature == 0.0
    assert config.max_output_tokens == asr_max_output_tokens(5.5)
    assert config.thinking_config.thinking_budget == 0
    assert config.automatic_function_calling.disable is True
    [row] = [dict(r) for r in ledger._conn().execute("SELECT * FROM cost_entries")]
    assert (row["operation"], row["model"], row["status"]) == ("asr", SETTINGS.asr_model, "ok")
    assert row["projected_usd"] == pytest.approx(
        project_asr_call(SETTINGS.asr_model, 5.5, asr_max_output_tokens(5.5))
    )
    assert row["cost_source"] == "estimated"  # Flash-Lite audio input price is still a placeholder
