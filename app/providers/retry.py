"""Retry layer 1: provider calls, transient errors only, capped full-jitter backoff (spec §4.6)."""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.config import Settings
from app.providers.errors import ProviderError


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_s: float = 1.0
    factor: float = 2.0
    cap_s: float = 20.0

    @classmethod
    def from_settings(cls, s: Settings) -> "RetryPolicy":
        return cls(
            max_attempts=s.retry_max_attempts,
            base_s=s.retry_base_s,
            factor=s.retry_factor,
            cap_s=s.retry_cap_s,
        )


def backoff_delay(policy: RetryPolicy, retry_index: int, rng: random.Random) -> float:
    return rng.uniform(0.0, min(policy.cap_s, policy.base_s * policy.factor**retry_index))


def call_with_retry[T](
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    on_retry: Callable[[int, ProviderError, float], None] | None = None,
) -> T:
    """Retries transient ProviderErrors only; anything else, BudgetRefused included, propagates."""
    rng = rng or random.Random()
    retry_index = 0
    while True:
        try:
            return fn()
        except ProviderError as err:
            if not err.transient or retry_index + 1 >= policy.max_attempts:
                raise
            delay = backoff_delay(policy, retry_index, rng)
            if on_retry is not None:
                on_retry(retry_index, err, delay)
            sleep(delay)
            retry_index += 1
