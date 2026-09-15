"""Bounded writer/critic loop with located feedback and a validated curated fallback (spec §5.6)."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from app.config import Settings
from app.domain.errors import AppError, BudgetRefused, ErrorCode
from app.pipeline.critic import Critic
from app.pipeline.gates.g1_schema import g1_validate
from app.pipeline.gates.g2 import run_g2
from app.pipeline.gates.types import Violation, format_violations
from app.pipeline.prompts import PROMPT_VERSIONS, render_rewrite_prompt, render_writer_prompt
from app.pipeline.storyboard import Storyboard, provider_json_schema
from app.providers.errors import ProviderError
from app.providers.llm.protocol import LLMProvider, LLMRequest
from app.render.layout import FontSet
from app.topics.pack import TopicPack

BUDGET_REASONS = {
    "job": "BUDGET_EXCEEDED_JOB",
    "project": "PROJECT_CAP_REACHED",
    "daily": "BUDGET_EXCEEDED_DAILY",
}


class RegistryLike(Protocol):
    def get(self, topic_id: str) -> TopicPack: ...

    def curated_storyboard(self, topic_id: str) -> Storyboard: ...


@dataclass
class AttemptRecord:
    attempt: int
    g1: dict[str, Any]
    g2: list[dict[str, Any]]
    g3: dict[str, Any] | None
    critic_skipped: bool
    passed: bool


@dataclass
class WriterOutcome:
    storyboard: Storyboard
    generation_source: Literal["llm", "curated_fallback"]
    fallback_reason: str | None
    attempts: list[AttemptRecord]
    unverified_claims: list[str]
    critic_scores: dict[str, int] | None
    critic_skipped: bool

    def report(self) -> dict[str, Any]:
        return {
            "generation_source": self.generation_source,
            "fallback_reason": self.fallback_reason,
            "attempts": [asdict(a) for a in self.attempts],
            "unverified_claims": self.unverified_claims,
            "critic_scores": self.critic_scores,
            "critic_skipped": self.critic_skipped,
            "prompt_versions": dict(PROMPT_VERSIONS),
        }


class _Fallback(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _codes(violations: list[Violation]) -> str:
    return ",".join(dict.fromkeys(v.code for v in violations))


class StoryboardWriter:
    def __init__(
        self,
        llm: LLMProvider,
        critic: Critic,
        registry: RegistryLike,
        settings: Settings,
        fonts: FontSet,
        *,
        emit: Callable[[str, dict[str, Any]], None] = lambda t, d: None,
    ) -> None:
        self._llm = llm
        self._critic = critic
        self._registry = registry
        self._settings = settings
        self._fonts = fonts
        self._emit = emit

    def write(self, *, topic_id: str, query: str) -> WriterOutcome:
        pack = self._registry.get(topic_id)
        attempts: list[AttemptRecord] = []
        try:
            return self._loop(pack, query, attempts)
        except _Fallback as fb:
            return self._fallback(pack, fb.reason, attempts)

    def _request(
        self, pack: TopicPack, query: str, previous: str | None, feedback: str
    ) -> LLMRequest:
        s = self._settings
        if previous is None:
            system, prompt = render_writer_prompt(pack, query, s)
        else:
            system, prompt = render_rewrite_prompt(pack, query, previous, feedback, s)
        return LLMRequest(
            operation="writer",
            model=s.writer_model,
            system=system,
            prompt=prompt,
            json_schema=provider_json_schema(),
            temperature=s.writer_temperature,
            max_output_tokens=s.writer_max_output_tokens,
            thinking_budget=s.writer_thinking_budget,
        )

    def _budget(self, exc: BudgetRefused) -> _Fallback:
        reason = BUDGET_REASONS[exc.scope]
        self._emit("budget_degraded", {"scope": exc.scope, "reason": reason})
        return _Fallback(reason)

    def _loop(self, pack: TopicPack, query: str, attempts: list[AttemptRecord]) -> WriterOutcome:
        s = self._settings
        previous: str | None = None
        violations: list[Violation] = []
        for attempt in range(1, s.max_storyboard_attempts + 1):
            kind = "writer" if previous is None else "rewrite"
            self._emit("writer_attempt", {"attempt": attempt, "prompt": PROMPT_VERSIONS[kind]})
            req = self._request(pack, query, previous, format_violations(violations))
            try:
                response = self._llm.generate_json(req)
            except BudgetRefused as exc:
                raise self._budget(exc) from exc
            except ProviderError as exc:
                # Provider retries already happened in the adapter; a provider error is not a
                # content problem, so it is never turned into a rewrite.
                raise _Fallback("LLM_PROVIDER_ERROR") from exc
            previous = response.text

            storyboard, g1 = g1_validate(response.text)
            record = AttemptRecord(attempt, g1.to_dict(), [], None, False, False)
            attempts.append(record)
            self._emit("gate_result", {"attempt": attempt, **g1.to_dict()})
            if storyboard is None:
                violations = g1.violations
                continue

            g2 = run_g2(storyboard, pack, s, self._fonts)
            record.g2 = [r.to_dict() for r in g2.results]
            for result in g2.results:
                self._emit("gate_result", {"attempt": attempt, **result.to_dict()})
            if not g2.passed:
                violations = g2.violations
                continue

            try:
                review = self._critic.review(
                    pack=pack, query=query, storyboard=storyboard, unverified=g2.unverified_claims
                )
            except BudgetRefused as exc:
                raise self._budget(exc) from exc
            if review.skipped:
                record.critic_skipped = True
                self._emit("critic_skipped", {"attempt": attempt, "reason": review.skip_reason})
            else:
                assert review.verdict is not None
                record.g3 = review.verdict.model_dump()
                if not review.verdict.passed(s.critic_min_score):
                    violations = review.verdict.as_violations(s.critic_min_score)
                    self._emit("gate_result", {"attempt": attempt, "gate": "G3", "passed": False})
                    continue
            record.passed = True
            return WriterOutcome(
                storyboard=storyboard,
                generation_source="llm",
                fallback_reason=None,
                attempts=attempts,
                unverified_claims=g2.unverified_claims,
                critic_scores=None if review.verdict is None else review.verdict.scores(),
                critic_skipped=review.skipped,
            )
        raise _Fallback(f"LLM_VALIDATION_FAILED:{_codes(violations)}")

    def _fallback(
        self, pack: TopicPack, reason: str, attempts: list[AttemptRecord]
    ) -> WriterOutcome:
        curated = self._registry.curated_storyboard(pack.id)
        storyboard, g1 = g1_validate(curated.model_dump(mode="json"))
        problems = list(g1.violations)
        unverified: list[str] = []
        if storyboard is not None:
            g2 = run_g2(storyboard, pack, self._settings, self._fonts)
            problems += g2.violations
            unverified = g2.unverified_claims
        if storyboard is None or problems:
            raise AppError(
                ErrorCode.CURATED_FALLBACK_INVALID,
                "deterministic_bug",
                f"curated storyboard for {pack.id} fails validation: {_codes(problems)}",
                {"violations": [v.message for v in problems]},
            )
        self._emit("fallback_used", {"reason": reason})
        return WriterOutcome(
            storyboard=storyboard,
            generation_source="curated_fallback",
            fallback_reason=reason,
            attempts=attempts,
            unverified_claims=unverified,
            critic_scores=None,
            critic_skipped=False,
        )
