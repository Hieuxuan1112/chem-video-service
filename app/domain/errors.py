"""Failure classification (spec §4.6) and domain exception types."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal


class ErrorCode(StrEnum):
    WORKER_LOST = "WORKER_LOST"
    CURATED_FALLBACK_INVALID = "CURATED_FALLBACK_INVALID"
    TTS_PROVIDER_UNAVAILABLE = "TTS_PROVIDER_UNAVAILABLE"
    TTS_VALIDATION_FAILED = "TTS_VALIDATION_FAILED"
    RENDER_FAILED = "RENDER_FAILED"
    RENDER_TIMEOUT = "RENDER_TIMEOUT"
    RENDER_TEMPLATE_ERROR = "RENDER_TEMPLATE_ERROR"
    RENDER_LAYOUT_ERROR = "RENDER_LAYOUT_ERROR"
    ASSEMBLY_FAILED = "ASSEMBLY_FAILED"
    QUALITY_CHECK_FAILED = "QUALITY_CHECK_FAILED"
    ARTIFACT_STORE_ERROR = "ARTIFACT_STORE_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


RETRYABILITY: Mapping[tuple[ErrorCode, str], bool] = {
    (ErrorCode.WORKER_LOST, "lease_expired_max_claims"): True,
    (ErrorCode.CURATED_FALLBACK_INVALID, "deterministic_bug"): False,
    (ErrorCode.TTS_PROVIDER_UNAVAILABLE, "provider_outage"): True,
    (ErrorCode.TTS_VALIDATION_FAILED, "output_invalid_after_retries"): True,
    (ErrorCode.RENDER_FAILED, "subprocess_crash"): True,
    (ErrorCode.RENDER_TIMEOUT, "timeout"): True,
    (ErrorCode.RENDER_TEMPLATE_ERROR, "deterministic_bug"): False,
    (ErrorCode.RENDER_LAYOUT_ERROR, "deterministic_bug"): False,
    (ErrorCode.ASSEMBLY_FAILED, "ffmpeg_transient"): True,
    (ErrorCode.ASSEMBLY_FAILED, "invalid_input"): False,
    (ErrorCode.QUALITY_CHECK_FAILED, "io_error"): True,
    (ErrorCode.QUALITY_CHECK_FAILED, "deterministic_output_bug"): False,
    (ErrorCode.ARTIFACT_STORE_ERROR, "io_error"): True,
    (ErrorCode.INTERNAL_ERROR, "unclassified"): False,
}


class AppError(Exception):
    """A classified failure. Retryability comes only from RETRYABILITY."""

    def __init__(
        self, code: ErrorCode, cause: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        key = (ErrorCode(code), cause)
        if key not in RETRYABILITY:
            raise ValueError(f"unclassified error pair: {code}/{cause}")
        super().__init__(message)
        self.code = ErrorCode(code)
        self.cause = cause
        self.message = message
        self.details = details or {}
        self.retryable: bool = RETRYABILITY[key]


def classify(exc: BaseException) -> AppError:
    if isinstance(exc, AppError):
        return exc
    return AppError(ErrorCode.INTERNAL_ERROR, "unclassified", f"{type(exc).__name__}: {exc}")


class IllegalTransition(Exception):
    pass


class StaleLeaseError(Exception):
    pass


class IdempotencyConflict(Exception):
    pass


class NotRetryable(Exception):
    pass


class BudgetRefused(Exception):
    def __init__(
        self,
        scope: Literal["job", "project", "daily"],
        needed_usd: float,
        limit_usd: float,
        current_usd: float,
    ) -> None:
        super().__init__(
            f"{scope} budget refused: current {current_usd:.6f} + needed {needed_usd:.6f}"
            f" > limit {limit_usd:.6f} USD"
        )
        self.scope = scope
        self.needed_usd = needed_usd
        self.limit_usd = limit_usd
        self.current_usd = current_usd
