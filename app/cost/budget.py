"""SpendGuard: reserve -> call -> settle around every paid provider call (spec §9.3)."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from app.config import Settings
from app.cost.ledger import Ledger, Limits, ReservationRequest
from app.cost.pricing import (
    AUDIO_INPUT_TOKENS_PER_S,
    TTS_AUDIO_TOKENS_PER_S,
    asr_call_cost,
    estimate_text_tokens,
    text_call_cost,
    tts_call_cost,
)
from app.jobs.clock import Clock
from app.observability.context import attempt_var, job_id_var, stage_var
from app.providers.errors import ProviderError

log = logging.getLogger(__name__)

WARNING_MARK = "project_warning_emitted"
MAX_ERROR_MESSAGE_CHARS = 1000


class EventSink(Protocol):
    def __call__(self, type: str, data: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class Settlement:
    status: Literal["ok", "error"]
    usd: float
    cost_source: Literal["actual", "estimated", "free"]
    units: dict[str, Any]
    error_code: str | None = None
    error_message: str | None = None


SettleFn = Callable[[Any, BaseException | None], Settlement]


class SpendGuard:
    def __init__(
        self, ledger: Ledger, settings: Settings, clock: Clock, *, events: EventSink | None = None
    ) -> None:
        self._ledger = ledger
        self._settings = settings
        self._clock = clock
        self._events = events

    def _day_start(self) -> float:
        now = self._clock.now()
        return now - now % 86400  # current UTC day

    def _limits(self) -> Limits:
        s = self._settings
        return Limits(
            hard_project_cap_usd=s.hard_project_cap_usd,
            job_budget_usd=s.cost_job_budget_usd,
            daily_budget_usd=s.cost_daily_budget_usd,
            day_start_ts=self._day_start(),
        )

    def run[T](
        self,
        *,
        operation: str,
        provider: str,
        model: str,
        projected_usd: float,
        call: Callable[[], T],
        settle: Callable[[T | None, BaseException | None], Settlement],
    ) -> T:
        """Raises BudgetRefused before the call when the reservation does not fit."""
        entry_id = self._ledger.try_reserve(
            ReservationRequest(
                operation=operation,
                provider=provider,
                model=model,
                projected_usd=projected_usd,
                job_id=job_id_var.get(),
                attempt=attempt_var.get(),
                stage=stage_var.get(),
            ),
            self._limits(),
        )
        result: T | None = None
        error: BaseException | None = None
        try:
            result = call()
            return result
        except BaseException as exc:
            error = exc
            raise
        finally:
            s = settle(result, error)
            self._ledger.settle(
                entry_id,
                status=s.status,
                usd=s.usd,
                cost_source=s.cost_source,
                units=s.units,
                error_code=s.error_code,
                error_message=s.error_message,
            )
            if error is not None:
                log.warning(
                    "provider_call_failed",
                    extra={
                        "data": {
                            "operation": operation,
                            "provider": provider,
                            "model": model,
                            "error_code": s.error_code,
                            "status": getattr(error, "status", None),
                            "error_message": s.error_message,
                        }
                    },
                )
            if s.usd > projected_usd:
                log.warning(
                    "projection_underestimate",
                    extra={
                        "data": {
                            "operation": operation,
                            "projected_usd": projected_usd,
                            "usd": s.usd,
                        }
                    },
                )
            self._check_warning()

    def _check_warning(self) -> None:
        billable = self._ledger.project_totals().billable_settled_usd
        warning = self._settings.warning_budget_usd
        if billable < warning or not self._ledger.mark_once(WARNING_MARK, f"{billable:.6f}"):
            return
        data = {"billable_usd": billable, "warning_usd": warning}
        log.warning("project_budget_warning", extra={"data": data})
        if self._events is not None:
            self._events("project_budget_warning", data)

    def headroom(self, *, job_id: str | None) -> dict[str, float | None]:
        s = self._settings
        daily = s.cost_daily_budget_usd
        return {
            "job_usd": None
            if job_id is None
            else s.cost_job_budget_usd - self._ledger.job_totals(job_id).committed_usd,
            "project_usd": s.hard_project_cap_usd - self._ledger.project_totals().committed_usd,
            "daily_usd": None
            if daily is None
            else daily - self._ledger.day_totals(self._day_start()).committed_usd,
        }

    def state(self) -> dict[str, Any]:
        s = self._settings
        totals = self._ledger.project_totals()
        if totals.committed_usd >= s.hard_project_cap_usd:
            state = "cap_reached"
        elif (
            self._ledger.has_mark(WARNING_MARK)
            or totals.billable_settled_usd >= s.warning_budget_usd
        ):
            state = "warning"
        else:
            state = "ok"
        return {
            "billable_usd": totals.billable_settled_usd,
            "reserved_usd": totals.reserved_usd,
            "warning_usd": s.warning_budget_usd,
            "hard_cap_usd": s.hard_project_cap_usd,
            "state": state,
        }


def assert_project_ledger(settings: Settings) -> None:
    """Paid-call commands refuse a scratch ledger, so the hard cap cannot be bypassed."""
    if not settings.is_project_ledger():
        raise RuntimeError(
            f"paid calls require the project ledger {settings.project_ledger_path},"
            f" got {settings.ledger_path()}"
        )


# --- settle rules ------------------------------------------------------------------------------
# Results are duck-typed: `usage` (input_tokens, output_tokens[, thinking_tokens]) or None, plus
# `text` for text/ASR results and `duration_s` for TTS results.


def _error_settlement(exc: BaseException, input_estimate_usd: float) -> Settlement:
    """Rejected before processing (400/401/403/429) costs 0; timeouts and 5xx may have been
    processed, so the input side is charged. Both are marked for billing-docs verification."""
    message = str(exc)[:MAX_ERROR_MESSAGE_CHARS] or None
    if isinstance(exc, ProviderError):
        usd = 0.0 if exc.rejected_before_processing else input_estimate_usd
        return Settlement("error", usd, "estimated", {}, exc.kind, message)
    return Settlement("error", input_estimate_usd, "estimated", {}, type(exc).__name__, message)


def settle_text(model: str, prompt_chars: int) -> SettleFn:
    input_tokens = estimate_text_tokens(prompt_chars)

    def settle(result: Any, exc: BaseException | None) -> Settlement:
        if exc is not None or result is None:
            assert exc is not None
            return _error_settlement(exc, text_call_cost(model, input_tokens, 0, 0))
        usage = result.usage
        if usage is None:
            out = estimate_text_tokens(len(result.text))
            units = {"input_tokens": input_tokens, "output_tokens": out, "thinking_tokens": 0}
            return Settlement("ok", text_call_cost(model, input_tokens, out, 0), "estimated", units)
        thinking = getattr(usage, "thinking_tokens", 0)
        units = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "thinking_tokens": thinking,
        }
        usd = text_call_cost(model, usage.input_tokens, usage.output_tokens, thinking)
        return Settlement("ok", usd, "actual", units)

    return settle


def settle_tts(model: str, text_chars: int) -> SettleFn:
    text_tokens = estimate_text_tokens(text_chars)

    def settle(result: Any, exc: BaseException | None) -> Settlement:
        if exc is not None or result is None:
            assert exc is not None
            return _error_settlement(exc, tts_call_cost(model, text_tokens, 0))
        usage = result.usage
        if usage is None:
            audio = round(result.duration_s * TTS_AUDIO_TOKENS_PER_S)
            units = {"text_tokens": text_tokens, "audio_tokens": audio}
            units["audio_seconds"] = result.duration_s
            return Settlement("ok", tts_call_cost(model, text_tokens, audio), "estimated", units)
        units = {
            "text_tokens": usage.input_tokens,
            "audio_tokens": usage.output_tokens,
            "audio_seconds": result.duration_s,
        }
        usd = tts_call_cost(model, usage.input_tokens, usage.output_tokens)
        return Settlement("ok", usd, "actual", units)

    return settle


def settle_asr(model: str, audio_s: float) -> SettleFn:
    """All reported input tokens are priced as audio: conservative, since audio >= text price."""
    audio_tokens = round(audio_s * AUDIO_INPUT_TOKENS_PER_S)

    def settle(result: Any, exc: BaseException | None) -> Settlement:
        if exc is not None or result is None:
            assert exc is not None
            usd, _ = asr_call_cost(model, audio_tokens, 0, 0)
            return _error_settlement(exc, usd)
        usage = result.usage
        if usage is None:
            out = estimate_text_tokens(len(result.text))
            usd, _ = asr_call_cost(model, audio_tokens, 0, out)
            units = {"audio_tokens": audio_tokens, "output_tokens": out, "audio_seconds": audio_s}
            return Settlement("ok", usd, "estimated", units)
        usd, source = asr_call_cost(model, usage.input_tokens, 0, usage.output_tokens)
        units = {
            "audio_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "audio_seconds": audio_s,
        }
        return Settlement("ok", usd, source, units)

    return settle
