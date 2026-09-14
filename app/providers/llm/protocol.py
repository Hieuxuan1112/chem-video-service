"""LLM provider protocol: the pipeline depends on this, never on an SDK."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    thinking_tokens: int = 0


@dataclass(frozen=True)
class LLMRequest:
    operation: Literal["writer", "critic", "calibration_critic"]
    model: str
    system: str
    prompt: str
    json_schema: dict[str, Any]
    temperature: float
    max_output_tokens: int
    thinking_budget: int = 0


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Usage | None
    model: str


class LLMProvider(Protocol):
    name: str

    def generate_json(self, req: LLMRequest) -> LLMResponse: ...
