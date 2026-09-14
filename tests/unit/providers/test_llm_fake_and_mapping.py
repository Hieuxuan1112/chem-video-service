from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.genai import errors
from google.genai._extra_utils import should_disable_afc

from app.config import Settings
from app.cost.budget import SpendGuard
from app.cost.ledger import Ledger
from app.jobs.clock import FakeClock
from app.providers.errors import ProviderError
from app.providers.llm import gemini
from app.providers.llm.fake import FakeLLM
from app.providers.llm.gemini import GeminiLLMProvider, map_google_error
from app.providers.llm.protocol import LLMRequest, LLMResponse, Usage
from app.providers.retry import RetryPolicy


def _req(**overrides: Any) -> LLMRequest:
    fields: dict[str, Any] = {
        "operation": "writer",
        "model": "gemini-2.5-flash",
        "system": "You write storyboards.",
        "prompt": "Explain the pH scale.",
        "json_schema": {"type": "object"},
        "temperature": 0.4,
        "max_output_tokens": 4096,
        "thinking_budget": 1024,
    }
    fields.update(overrides)
    return LLMRequest(**fields)


def test_fake_llm_script_order() -> None:
    err = ProviderError("server", "boom", status=503)
    fake = FakeLLM(["first", err, lambda req: f"echo:{req.prompt}"])
    assert fake.name == "fake"

    r1 = fake.generate_json(_req(prompt="a"))
    assert r1 == LLMResponse(text="first", usage=Usage(1000, 500), model="gemini-2.5-flash")
    with pytest.raises(ProviderError) as info:
        fake.generate_json(_req(prompt="b"))
    assert info.value is err
    assert fake.generate_json(_req(prompt="c")).text == "echo:c"
    assert [r.prompt for r in fake.requests] == ["a", "b", "c"]
    with pytest.raises(AssertionError):
        fake.generate_json(_req())

    assert FakeLLM(["x"], usage=None).generate_json(_req()).usage is None


@pytest.mark.parametrize(
    ("exc", "kind", "status"),
    [
        (errors.ClientError(429, {"error": {"message": "quota"}}), "rate_limited", 429),
        (errors.ServerError(500, {"error": {"message": "internal"}}), "server", 500),
        (errors.ServerError(503, {"error": {"message": "unavailable"}}), "server", 503),
        (errors.ClientError(401, {"error": {"message": "key"}}), "auth", 401),
        (errors.ClientError(403, {"error": {"message": "denied"}}), "auth", 403),
        (errors.ClientError(400, {"error": {"message": "schema"}}), "invalid_request", 400),
        (errors.ClientError(404, {"error": {"message": "model"}}), "unknown", 404),
        (httpx.ReadTimeout("read timed out"), "timeout", None),
        (TimeoutError("timed out"), "timeout", None),
        (ValueError("weird"), "unknown", None),
    ],
)
def test_error_mapping_table(exc: Exception, kind: str, status: int | None) -> None:
    mapped = map_google_error(exc)
    assert isinstance(mapped, ProviderError)
    assert (mapped.kind, mapped.status, mapped.provider) == (kind, status, "gemini")


TOO_MANY_STATES = "The specified schema produces a constraint that has too many states for serving."


def test_provider_message_preserved_and_key_redacted() -> None:
    exc = errors.ClientError(
        400, {"error": {"code": 400, "message": TOO_MANY_STATES, "status": "INVALID_ARGUMENT"}}
    )
    mapped = map_google_error(exc)
    assert mapped.message == f"400 INVALID_ARGUMENT: {TOO_MANY_STATES}"

    leaky = errors.ClientError(403, {"error": {"message": "key AQ.secret-123 is not valid"}})
    redacted = map_google_error(leaky, secrets=("AQ.secret-123",))
    assert "AQ.secret-123" not in redacted.message
    assert "[REDACTED]" in redacted.message
    assert "AQ.secret-123" not in str(redacted)


class StubModels:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_adapter_uses_guard_per_attempt(
    monkeypatch: pytest.MonkeyPatch, db_path: Path, clock: FakeClock
) -> None:
    ok = SimpleNamespace(
        text='{"ok": true}',
        usage_metadata=SimpleNamespace(
            prompt_token_count=12, candidates_token_count=7, thoughts_token_count=30
        ),
    )
    models = StubModels([errors.ClientError(429, {"error": {"message": "quota"}}), ok])
    clients: list[dict[str, Any]] = []

    class StubClient:
        def __init__(self, **kwargs: Any) -> None:
            clients.append(kwargs)
            self.models = models

    monkeypatch.setattr(gemini.genai, "Client", StubClient)
    ledger = Ledger(db_path, clock)
    guard = SpendGuard(ledger, Settings(_env_file=None), clock)  # type: ignore[call-arg]
    provider = GeminiLLMProvider("test-key", guard, RetryPolicy(base_s=0.0), timeout_s=30.0)

    resp = provider.generate_json(_req())

    assert resp == LLMResponse(
        text='{"ok": true}', usage=Usage(12, 7, 30), model="gemini-2.5-flash"
    )
    rows = [dict(r) for r in ledger._conn().execute("SELECT * FROM cost_entries ORDER BY id")]
    assert [(r["status"], r["operation"], r["error_code"]) for r in rows] == [
        ("error", "writer", "rate_limited"),
        ("ok", "writer", None),
    ]
    assert rows[0]["usd"] == 0.0
    assert "quota" in rows[0]["error_message"]
    assert rows[1]["cost_source"] == "actual"

    http_options = clients[0]["http_options"]
    assert clients[0]["api_key"] == "test-key"
    assert http_options.timeout == 30_000  # milliseconds
    assert http_options.retry_options.attempts == 1  # our retry layer is the only one
    config = models.calls[1]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == {"type": "object"}
    assert config.max_output_tokens == 4096
    assert config.thinking_config.thinking_budget == 1024
    assert config.temperature == 0.4
    assert config.system_instruction == "You write storyboards."
    assert config.automatic_function_calling.disable is True
    assert should_disable_afc(config)
    assert models.calls[1]["contents"] == "Explain the pH scale."
