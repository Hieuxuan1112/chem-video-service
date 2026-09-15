import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors

from app.config import Settings
from app.cost.budget import SpendGuard
from app.cost.ledger import Ledger
from app.domain.errors import AppError, BudgetRefused, ErrorCode
from app.jobs.clock import FakeClock
from app.pipeline.critic import Critic
from app.pipeline.storyboard import Storyboard, provider_json_schema
from app.pipeline.writer import StoryboardWriter
from app.providers.errors import ProviderError
from app.providers.llm import gemini
from app.providers.llm.fake import FakeLLM
from app.providers.llm.gemini import GeminiLLMProvider
from app.providers.llm.protocol import LLMRequest, LLMResponse
from app.providers.retry import RetryPolicy, call_with_retry
from app.render.layout import FontSet
from app.topics.registry import TOPICS_ROOT, TopicRegistry

REGISTRY = TopicRegistry.load(TOPICS_ROOT)
SETTINGS = Settings(_env_file=None)  # type: ignore[call-arg]
TOPIC = "ph_scale"
QUERY = "How does the pH scale work?"
GOOD = REGISTRY.curated_storyboard(TOPIC).model_dump_json()
G1_FIVE_POINTS = "Scene 5: visual.points: List should have at most 4 items after validation, not 5"


def _five_points() -> str:
    """The G1 failure observed live on 2026-09-14: a key_points visual with 5 points."""
    raw = json.loads(GOOD)
    raw["scenes"][5]["visual"]["points"].append("Water is neutral")
    return json.dumps(raw)


def _misconception() -> str:
    """Passes G1, fails G2a (misconception) and G2c (pH 7 classified as acidic)."""
    raw = json.loads(GOOD)
    raw["scenes"][2]["narration"] += " A pH of 7 is acidic."
    return json.dumps(raw)


def _verdict(score: int = 5) -> str:
    return json.dumps(
        {"accuracy": score, "relevance": 5, "clarity": 5, "alignment": 5, "issues": []}
    )


class Events:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, type: str, data: dict[str, Any]) -> None:
        self.items.append((type, data))

    def types(self) -> list[str]:
        return [t for t, _ in self.items]


def _writer(
    llm: Any, critic_llm: Any, fonts: FontSet, events: Events | None = None, registry: Any = None
) -> StoryboardWriter:
    return StoryboardWriter(
        llm,
        Critic(critic_llm, SETTINGS),
        registry or REGISTRY,
        SETTINGS,
        fonts,
        emit=events or Events(),
    )


def test_first_attempt_valid_is_llm(fonts: FontSet) -> None:
    llm, critic_llm, events = FakeLLM([GOOD]), FakeLLM([_verdict()]), Events()
    outcome = _writer(llm, critic_llm, fonts, events).write(topic_id=TOPIC, query=QUERY)

    assert outcome.generation_source == "llm"
    assert outcome.fallback_reason is None
    assert outcome.storyboard == Storyboard.model_validate_json(GOOD)
    assert [(a.attempt, a.passed) for a in outcome.attempts] == [(1, True)]
    assert outcome.critic_scores == {"accuracy": 5, "relevance": 5, "clarity": 5, "alignment": 5}
    assert outcome.critic_skipped is False
    [req] = llm.requests
    assert (req.operation, req.model) == ("writer", SETTINGS.writer_model)
    assert req.json_schema == provider_json_schema()
    assert req.max_output_tokens == SETTINGS.writer_max_output_tokens
    assert req.thinking_budget == SETTINGS.writer_thinking_budget
    assert "writer_attempt" in events.types() and "gate_result" in events.types()
    report = outcome.report()
    assert report["generation_source"] == "llm"
    assert report["prompt_versions"]["writer"] == "writer_v1"
    json.dumps(report)


def test_invalid_then_valid_rewrite_contains_exact_violations(fonts: FontSet) -> None:
    bad = _five_points()
    llm, critic_llm = FakeLLM([bad, GOOD]), FakeLLM([_verdict()])
    outcome = _writer(llm, critic_llm, fonts).write(topic_id=TOPIC, query=QUERY)

    assert outcome.generation_source == "llm"
    first, second = outcome.attempts
    assert (first.passed, second.passed) == (False, True)
    assert first.g1["passed"] is False
    assert [v["message"] for v in first.g1["violations"]] == [G1_FIVE_POINTS]
    rewrite = llm.requests[1]
    assert f"1. {G1_FIVE_POINTS}" in rewrite.prompt
    assert bad in rewrite.prompt  # the previous storyboard is shown, not the original prompt
    assert rewrite.prompt != llm.requests[0].prompt
    assert rewrite.json_schema == llm.requests[0].json_schema == provider_json_schema()
    assert len(critic_llm.requests) == 1


def test_g2_failure_is_fed_back(fonts: FontSet) -> None:
    llm, critic_llm = FakeLLM([_misconception(), GOOD]), FakeLLM([_verdict()])
    outcome = _writer(llm, critic_llm, fonts).write(topic_id=TOPIC, query=QUERY)
    assert outcome.generation_source == "llm"
    assert (
        "Scene 2: incorrect statement. A pH of 7 is neutral, not acidic." in llm.requests[1].prompt
    )
    assert outcome.attempts[0].g1["passed"] is True
    assert any(not r["passed"] for r in outcome.attempts[0].g2)


def test_three_invalid_fall_back_with_reason_codes(fonts: FontSet) -> None:
    llm = FakeLLM([_five_points(), "not json", _misconception()])
    critic_llm, events = FakeLLM([]), Events()
    outcome = _writer(llm, critic_llm, fonts, events).write(topic_id=TOPIC, query=QUERY)

    assert outcome.generation_source == "curated_fallback"
    assert outcome.fallback_reason == "LLM_VALIDATION_FAILED:misconception,claim_ph_classification"
    assert outcome.storyboard == REGISTRY.curated_storyboard(TOPIC)
    assert len(llm.requests) == SETTINGS.max_storyboard_attempts == 3
    assert [a.passed for a in outcome.attempts] == [False, False, False]
    assert critic_llm.requests == []
    assert ("fallback_used", {"reason": outcome.fallback_reason}) in events.items


def test_g1_failure_skips_g2_and_critic(fonts: FontSet) -> None:
    llm, critic_llm = FakeLLM([_five_points(), GOOD]), FakeLLM([_verdict()])
    outcome = _writer(llm, critic_llm, fonts).write(topic_id=TOPIC, query=QUERY)
    first = outcome.attempts[0]
    assert first.g2 == [] and first.g3 is None
    assert len(critic_llm.requests) == 1  # only for the second, valid attempt


@pytest.mark.parametrize(
    "error",
    [
        ProviderError("auth", "403 PERMISSION_DENIED: key", status=403),
        ProviderError(
            "invalid_request",
            "400 INVALID_ARGUMENT: The specified schema produces a constraint that has too many"
            " states for serving.",
            status=400,
        ),
    ],
)
def test_auth_error_immediate_fallback_no_retry(fonts: FontSet, error: ProviderError) -> None:
    llm, critic_llm, events = FakeLLM([error, GOOD]), FakeLLM([_verdict()]), Events()
    outcome = _writer(llm, critic_llm, fonts, events).write(topic_id=TOPIC, query=QUERY)
    assert outcome.generation_source == "curated_fallback"
    assert outcome.fallback_reason == "LLM_PROVIDER_ERROR"
    assert len(llm.requests) == 1  # a provider error is not a content problem: no rewrite
    assert critic_llm.requests == []


class RetryingLLM:
    """Stands in for an adapter: call_with_retry around each provider call."""

    name = "retrying"

    def __init__(self, inner: FakeLLM) -> None:
        self.inner = inner

    def generate_json(self, req: LLMRequest) -> LLMResponse:
        return call_with_retry(
            lambda: self.inner.generate_json(req), policy=RetryPolicy(), sleep=lambda _: None
        )


def test_rate_limit_twice_then_success(fonts: FontSet) -> None:
    limited = ProviderError("rate_limited", "429", status=429)
    inner = FakeLLM([limited, limited, GOOD])
    outcome = _writer(RetryingLLM(inner), FakeLLM([_verdict()]), fonts).write(
        topic_id=TOPIC, query=QUERY
    )
    assert outcome.generation_source == "llm"
    assert len(inner.requests) == 3
    assert len(outcome.attempts) == 1


def test_retries_exhausted_is_provider_error(fonts: FontSet) -> None:
    limited = ProviderError("rate_limited", "429", status=429)
    inner = FakeLLM([limited, limited, limited, GOOD])
    outcome = _writer(RetryingLLM(inner), FakeLLM([_verdict()]), fonts).write(
        topic_id=TOPIC, query=QUERY
    )
    assert outcome.fallback_reason == "LLM_PROVIDER_ERROR"
    assert len(inner.requests) == 3


def test_critic_error_sets_skipped_and_passes(fonts: FontSet) -> None:
    critic_llm, events = FakeLLM([ProviderError("timeout", "slow")]), Events()
    outcome = _writer(FakeLLM([GOOD]), critic_llm, fonts, events).write(topic_id=TOPIC, query=QUERY)
    assert outcome.generation_source == "llm"
    assert outcome.critic_skipped is True
    assert outcome.attempts[0].critic_skipped is True
    assert outcome.critic_scores is None
    assert ("critic_skipped", {"attempt": 1, "reason": "provider_error:timeout"}) in events.items


def test_critic_low_score_triggers_rewrite(fonts: FontSet) -> None:
    llm, critic_llm = FakeLLM([GOOD, GOOD]), FakeLLM([_verdict(score=2), _verdict()])
    outcome = _writer(llm, critic_llm, fonts).write(topic_id=TOPIC, query=QUERY)
    assert outcome.generation_source == "llm"
    assert len(outcome.attempts) == 2
    assert outcome.attempts[0].g3 is not None
    assert "1. Critic scored accuracy 2/5; the minimum is 4." in llm.requests[1].prompt


def _refusing(scope: str) -> Any:
    def refuse(_: LLMRequest) -> str:
        raise BudgetRefused(scope, 0.013, 0.20, 0.19)  # type: ignore[arg-type]

    return refuse


def test_job_budget_refusal_stops_llm_calls(fonts: FontSet) -> None:
    llm, critic_llm, events = FakeLLM([_refusing("job"), GOOD]), FakeLLM([_verdict()]), Events()
    outcome = _writer(llm, critic_llm, fonts, events).write(topic_id=TOPIC, query=QUERY)
    assert outcome.generation_source == "curated_fallback"
    assert outcome.fallback_reason == "BUDGET_EXCEEDED_JOB"
    assert len(llm.requests) == 1 and critic_llm.requests == []
    assert "budget_degraded" in events.types()


@pytest.mark.parametrize(
    ("scope", "reason"), [("project", "PROJECT_CAP_REACHED"), ("daily", "BUDGET_EXCEEDED_DAILY")]
)
def test_project_cap_refusal_reason(fonts: FontSet, scope: str, reason: str) -> None:
    critic_llm = FakeLLM([_refusing(scope)])
    outcome = _writer(FakeLLM([GOOD]), critic_llm, fonts).write(topic_id=TOPIC, query=QUERY)
    assert outcome.fallback_reason == reason


class BrokenCuratedRegistry:
    def __init__(self, inner: TopicRegistry) -> None:
        self.inner = inner

    def get(self, topic_id: str) -> Any:
        return self.inner.get(topic_id)

    def curated_storyboard(self, topic_id: str) -> Storyboard:
        return Storyboard.model_validate_json(_misconception())


def test_curated_invalid_raises_non_retryable(fonts: FontSet) -> None:
    writer = _writer(
        FakeLLM([ProviderError("auth", "403", status=403)]),
        FakeLLM([]),
        fonts,
        registry=BrokenCuratedRegistry(REGISTRY),
    )
    with pytest.raises(AppError) as info:
        writer.write(topic_id=TOPIC, query=QUERY)
    assert info.value.code == ErrorCode.CURATED_FALLBACK_INVALID
    assert info.value.retryable is False


class StubModels:
    def __init__(self, texts: Sequence[object]) -> None:
        self.texts = list(texts)

    def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        item = self.texts.pop(0)
        if isinstance(item, BaseException):
            raise item
        usage = SimpleNamespace(
            prompt_token_count=1000, candidates_token_count=500, thoughts_token_count=100
        )
        return SimpleNamespace(text=item, usage_metadata=usage)


def test_every_provider_call_is_in_the_ledger(
    monkeypatch: pytest.MonkeyPatch, db_path: Path, clock: FakeClock, fonts: FontSet
) -> None:
    rejected = errors.ClientError(
        429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}}
    )
    writer_models = StubModels([_five_points(), rejected, GOOD])
    critic_models = StubModels([_verdict()])
    queue = [writer_models, critic_models]

    class StubClient:
        def __init__(self, **_: Any) -> None:
            self.models = queue.pop(0)

    monkeypatch.setattr(gemini.genai, "Client", StubClient)
    ledger = Ledger(db_path, clock)
    guard = SpendGuard(ledger, SETTINGS, clock)
    policy = RetryPolicy(base_s=0.0)
    writer_llm = GeminiLLMProvider("k", guard, policy, timeout_s=5)
    critic_llm = GeminiLLMProvider("k", guard, policy, timeout_s=5)

    outcome = _writer(writer_llm, critic_llm, fonts).write(topic_id=TOPIC, query=QUERY)

    assert outcome.generation_source == "llm"
    rows = [dict(r) for r in ledger._conn().execute("SELECT * FROM cost_entries ORDER BY id")]
    assert [(r["operation"], r["model"], r["status"]) for r in rows] == [
        ("writer", SETTINGS.writer_model, "ok"),  # attempt 1: G1 failure is still a paid call
        ("writer", SETTINGS.writer_model, "error"),  # attempt 2, provider retry 1: 429
        ("writer", SETTINGS.writer_model, "ok"),  # attempt 2, provider retry 2
        ("critic", SETTINGS.critic_model, "ok"),
    ]
    assert ledger.unsettled() == []
