"""G3 critic (spec §5.4): a rubric judge that adds signal but never blocks when unavailable."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.config import Settings
from app.pipeline.gates.types import Violation
from app.pipeline.prompts import render_critic_prompt
from app.pipeline.storyboard import Storyboard
from app.providers.errors import ProviderError
from app.providers.llm.protocol import LLMProvider, LLMRequest
from app.topics.pack import TopicPack

GATE = "G3"
_SCORES = ("accuracy", "relevance", "clarity", "alignment")


class CriticIssue(BaseModel):
    scene_index: int | None = None  # omitted for storyboard-wide issues
    severity: Literal["blocker", "major", "minor"]
    message: str


class CriticVerdict(BaseModel):
    accuracy: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    alignment: int = Field(ge=1, le=5)
    issues: list[CriticIssue] = []

    def scores(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in _SCORES}

    def passed(self, min_score: int) -> bool:
        return all(s >= min_score for s in self.scores().values()) and not any(
            i.severity == "blocker" for i in self.issues
        )

    def as_violations(self, min_score: int) -> list[Violation]:
        out = [
            Violation(
                GATE,
                "critic_low_score",
                f"Critic scored {name} {score}/5; the minimum is {min_score}.",
            )
            for name, score in self.scores().items()
            if score < min_score
        ]
        for issue in self.issues:
            prefix = f"Scene {issue.scene_index}: " if issue.scene_index is not None else ""
            out.append(
                Violation(
                    GATE, f"critic_{issue.severity}", prefix + issue.message, issue.scene_index
                )
            )
        return out


@dataclass
class CriticOutcome:
    verdict: CriticVerdict | None
    skipped: bool
    skip_reason: str | None


def critic_provider_schema() -> dict[str, Any]:
    """Types, enums and required fields only; score bounds are checked by CriticVerdict (§5.3)."""
    score = {"type": "integer"}
    return {
        "type": "object",
        "properties": {
            **{name: dict(score) for name in _SCORES},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "scene_index": {"type": "integer"},
                        "severity": {"type": "string", "enum": ["blocker", "major", "minor"]},
                        "message": {"type": "string"},
                    },
                    "required": ["severity", "message"],
                },
            },
        },
        "required": [*_SCORES, "issues"],
    }


class Critic:
    def __init__(self, llm: LLMProvider, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings

    def review(
        self, *, pack: TopicPack, query: str, storyboard: Storyboard, unverified: Sequence[str]
    ) -> CriticOutcome:
        """BudgetRefused propagates; any other failure skips the critic."""
        system, prompt = render_critic_prompt(pack, query, storyboard.model_dump_json(), unverified)
        req = LLMRequest(
            operation="critic",
            model=self._settings.critic_model,
            system=system,
            prompt=prompt,
            json_schema=critic_provider_schema(),
            temperature=0.0,
            max_output_tokens=self._settings.critic_max_output_tokens,
        )
        try:
            response = self._llm.generate_json(req)
        except ProviderError as exc:
            return CriticOutcome(None, True, f"provider_error:{exc.kind}")
        try:
            verdict = CriticVerdict.model_validate_json(response.text)
        except ValidationError:
            return CriticOutcome(None, True, "invalid_output")
        return CriticOutcome(verdict, False, None)
