"""Gate result types shared by G1, G2 and G3."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Violation:
    gate: str
    code: str
    message: str
    scene_index: int | None = None


@dataclass
class GateResult:
    gate: str
    passed: bool
    violations: list[Violation]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "violations": [asdict(v) for v in self.violations],
            "details": self.details,
        }


def format_violations(violations: Sequence[Violation]) -> str:
    """Numbered lines, used verbatim in the rewrite prompt."""
    return "\n".join(f"{i}. {v.message}" for i, v in enumerate(violations, start=1))
