import json

import pytest

from app.config import Settings
from app.domain.errors import BudgetRefused
from app.pipeline.critic import Critic, CriticOutcome, CriticVerdict, critic_provider_schema
from app.providers.errors import ProviderError
from app.providers.llm.fake import FakeLLM
from app.topics.registry import TOPICS_ROOT, TopicRegistry

REGISTRY = TopicRegistry.load(TOPICS_ROOT)
SETTINGS = Settings(_env_file=None)  # type: ignore[call-arg]
PACK = REGISTRY.get("ph_scale")
STORYBOARD = REGISTRY.curated_storyboard("ph_scale")


def _verdict(score: int = 5, issues: list[dict[str, object]] | None = None) -> str:
    return json.dumps(
        {
            "accuracy": score,
            "relevance": 5,
            "clarity": 5,
            "alignment": 5,
            "issues": issues or [],
        }
    )


def _review(llm: FakeLLM) -> CriticOutcome:
    critic = Critic(llm, SETTINGS)
    return critic.review(pack=PACK, query="How does pH work?", storyboard=STORYBOARD, unverified=[])


def test_pass() -> None:
    llm = FakeLLM([_verdict(issues=[{"scene_index": 2, "severity": "minor", "message": "wordy"}])])
    outcome = _review(llm)
    assert outcome.skipped is False
    verdict = outcome.verdict
    assert verdict.passed(SETTINGS.critic_min_score)
    [req] = llm.requests
    assert (req.operation, req.model, req.temperature) == ("critic", "gemini-2.5-flash-lite", 0.0)
    assert req.max_output_tokens == SETTINGS.critic_max_output_tokens
    assert req.json_schema == critic_provider_schema()
    assert STORYBOARD.model_dump_json() in req.prompt


def test_low_score_fails_with_violation_message() -> None:
    verdict = CriticVerdict.model_validate_json(_verdict(score=3))
    assert not verdict.passed(4)
    violations = verdict.as_violations(4)
    assert [(v.gate, v.code) for v in violations] == [("G3", "critic_low_score")]
    assert violations[0].message == "Critic scored accuracy 3/5; the minimum is 4."


def test_blocker_fails() -> None:
    issue = {"scene_index": 1, "severity": "blocker", "message": "Says pH 7 is acidic."}
    verdict = CriticVerdict.model_validate_json(_verdict(issues=[issue]))
    assert not verdict.passed(4)
    [v] = verdict.as_violations(4)
    assert (v.gate, v.code, v.scene_index) == ("G3", "critic_blocker", 1)
    assert v.message == "Scene 1: Says pH 7 is acidic."
    whole = CriticVerdict.model_validate_json(
        _verdict(issues=[{"severity": "major", "message": "Off topic."}])
    )
    assert [(v.code, v.scene_index, v.message) for v in whole.as_violations(4)] == [
        ("critic_major", None, "Off topic.")
    ]


def test_provider_error_skips() -> None:
    outcome = _review(FakeLLM([ProviderError("server", "503", status=503)]))
    assert outcome.skipped is True
    assert outcome.verdict is None
    assert outcome.skip_reason == "provider_error:server"


@pytest.mark.parametrize("text", ["not json", json.dumps({"accuracy": 9})])
def test_invalid_json_skips(text: str) -> None:
    outcome = _review(FakeLLM([text]))
    assert outcome.skipped is True
    assert outcome.skip_reason == "invalid_output"


def test_budget_refused_propagates() -> None:
    def refuse(_: object) -> str:
        raise BudgetRefused("job", 0.001, 0.20, 0.2)

    with pytest.raises(BudgetRefused):
        _review(FakeLLM([refuse]))


def test_critic_provider_schema_has_no_bounds() -> None:
    text = json.dumps(critic_provider_schema())
    for key in ("minimum", "maximum", "minItems", "maxItems", "maxLength", "minLength"):
        assert key not in text
