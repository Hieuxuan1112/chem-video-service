"""Provider-neutral errors classified by the retry layer and cost settlement (spec §4.6, §9.3)."""

from typing import Literal

ProviderErrorKind = Literal[
    "rate_limited", "server", "timeout", "auth", "invalid_request", "unknown"
]

_TRANSIENT: frozenset[str] = frozenset({"rate_limited", "server", "timeout"})
_REJECTED_STATUSES: frozenset[int] = frozenset({400, 401, 403, 429})


class ProviderError(Exception):
    def __init__(
        self,
        kind: ProviderErrorKind,
        message: str,
        *,
        status: int | None = None,
        provider: str = "",
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind: ProviderErrorKind = kind
        self.message = message
        self.status = status
        self.provider = provider
        self.model = model

    @property
    def transient(self) -> bool:
        return self.kind in _TRANSIENT

    @property
    def rejected_before_processing(self) -> bool:
        return self.status in _REJECTED_STATUSES
