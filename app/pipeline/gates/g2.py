"""G2 aggregate: rules and chemistry consistency, claims, then layout (cheapest first)."""

from dataclasses import dataclass

from app.config import Settings
from app.pipeline.gates.g2_claims import g2_claims
from app.pipeline.gates.g2_layout import g2_layout
from app.pipeline.gates.g2_rules import g2_rules
from app.pipeline.gates.types import GateResult, Violation
from app.pipeline.storyboard import Storyboard
from app.render.layout import FontSet
from app.topics.pack import TopicPack


@dataclass
class G2Outcome:
    results: list[GateResult]
    unverified_claims: list[str]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def violations(self) -> list[Violation]:
        return [v for r in self.results for v in r.violations]


def run_g2(sb: Storyboard, pack: TopicPack, settings: Settings, fonts: FontSet) -> G2Outcome:
    rules = g2_rules(sb, pack, settings)
    claims, unverified = g2_claims(sb, pack)
    layout = g2_layout(sb, fonts)
    return G2Outcome(results=[rules, claims, layout], unverified_claims=unverified)
