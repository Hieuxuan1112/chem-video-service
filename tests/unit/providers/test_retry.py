import random

import pytest

from app.config import Settings
from app.domain.errors import BudgetRefused
from app.providers.errors import ProviderError
from app.providers.retry import RetryPolicy, backoff_delay, call_with_retry


class Script:
    """Raises the scripted errors in order, then returns 'ok'."""

    def __init__(self, *outcomes: BaseException) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.outcomes:
            raise self.outcomes.pop(0)
        return "ok"


def test_transient_retried_then_success() -> None:
    fn = Script(ProviderError("rate_limited", "slow", status=429), ProviderError("server", "x"))
    delays: list[float] = []
    retries: list[tuple[int, str]] = []

    result = call_with_retry(
        fn,
        policy=RetryPolicy(),
        sleep=delays.append,
        rng=random.Random(1),
        on_retry=lambda i, err, d: retries.append((i, err.kind)),
    )

    assert result == "ok"
    assert fn.calls == 3
    assert len(delays) == 2
    assert retries == [(0, "rate_limited"), (1, "server")]


def test_non_transient_not_retried() -> None:
    for kind in ("auth", "invalid_request", "unknown"):
        fn = Script(ProviderError(kind, "no"))  # type: ignore[arg-type]
        delays: list[float] = []
        with pytest.raises(ProviderError):
            call_with_retry(fn, policy=RetryPolicy(), sleep=delays.append)
        assert fn.calls == 1
        assert delays == []


def test_gives_up_after_max_attempts_raises_last() -> None:
    errors = [ProviderError("timeout", f"t{i}") for i in range(5)]
    fn = Script(*errors)
    delays: list[float] = []
    with pytest.raises(ProviderError) as info:
        call_with_retry(fn, policy=RetryPolicy(max_attempts=3), sleep=delays.append)
    assert info.value is errors[2]
    assert fn.calls == 3
    assert len(delays) == 2


def test_delays_bounded_by_cap_with_jitter() -> None:
    policy = RetryPolicy(base_s=1.0, factor=2.0, cap_s=20.0)
    rng = random.Random(42)
    for i in range(10):
        ceiling = min(20.0, 1.0 * 2.0**i)
        samples = [backoff_delay(policy, i, rng) for _ in range(200)]
        assert all(0.0 <= d <= ceiling for d in samples)
        # full jitter: spread over the interval, not pinned to the ceiling
        assert min(samples) < ceiling / 2 < max(samples)
    assert backoff_delay(policy, 3, random.Random(7)) == backoff_delay(policy, 3, random.Random(7))


def test_budget_refused_not_retried() -> None:
    fn = Script(BudgetRefused("project", 0.01, 5.0, 4.995))
    delays: list[float] = []
    with pytest.raises(BudgetRefused):
        call_with_retry(fn, policy=RetryPolicy(), sleep=delays.append)
    assert fn.calls == 1
    assert delays == []


def test_error_flags_and_policy_from_settings() -> None:
    assert ProviderError("rate_limited", "x", status=429).transient
    assert ProviderError("rate_limited", "x", status=429).rejected_before_processing
    assert not ProviderError("server", "x", status=503).rejected_before_processing
    assert not ProviderError("timeout", "x").rejected_before_processing
    assert ProviderError("auth", "x", status=403).rejected_before_processing
    assert not ProviderError("auth", "x", status=403).transient

    s = Settings(retry_max_attempts=4, retry_base_s=0.5, retry_factor=3.0, retry_cap_s=9.0)
    assert RetryPolicy.from_settings(s) == RetryPolicy(4, 0.5, 3.0, 9.0)
