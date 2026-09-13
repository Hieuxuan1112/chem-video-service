import pytest

from app.domain.errors import RETRYABILITY, AppError, ErrorCode, classify

SPEC_TABLE = {
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


def test_table_matches_spec() -> None:
    assert len(SPEC_TABLE) == 14
    assert dict(RETRYABILITY) == SPEC_TABLE


@pytest.mark.parametrize(("pair", "retryable"), SPEC_TABLE.items())
def test_app_error_retryable_from_table(pair: tuple[ErrorCode, str], retryable: bool) -> None:
    err = AppError(pair[0], pair[1], "msg")
    assert err.retryable is retryable


def test_unknown_pair_rejected() -> None:
    with pytest.raises(ValueError):
        AppError(ErrorCode.RENDER_FAILED, "timeout", "not in the table")


def test_classify_passthrough() -> None:
    err = AppError(ErrorCode.RENDER_TIMEOUT, "timeout", "slow", {"scene": 3})
    assert classify(err) is err


def test_classify_unknown_exception_is_internal_non_retryable() -> None:
    err = classify(KeyError("boom"))
    assert err.code is ErrorCode.INTERNAL_ERROR
    assert err.cause == "unclassified"
    assert err.retryable is False
    assert "KeyError" in err.message
