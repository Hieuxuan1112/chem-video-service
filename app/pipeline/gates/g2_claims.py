"""G2c: numeric claims in narration and on-screen text are verified against code data."""

from app.pipeline.gates.g2_rules import on_screen_texts
from app.pipeline.gates.types import GateResult, Violation
from app.pipeline.storyboard import Storyboard
from app.topics.claims import check_texts
from app.topics.pack import TopicPack

GATE = "G2_claims"


def g2_claims(sb: Storyboard, pack: TopicPack) -> tuple[GateResult, list[str]]:
    violations: list[Violation] = []
    unverified: list[str] = []
    checked = 0
    for i, scene in enumerate(sb.scenes):
        report = check_texts([scene.narration, *on_screen_texts(scene)], pack.claim_checks)
        checked += len(report.findings)
        for finding in report.violations:
            violations.append(
                Violation(
                    "G2c",
                    f"claim_{finding.checker}",
                    f'Scene {i}: "{finding.clause}" is wrong: {finding.message}',
                    i,
                )
            )
        unverified.extend(c for c in report.unverified if c not in unverified)
    details = {"claims_checked": checked, "unverified_numeric_claims": unverified}
    return GateResult(GATE, not violations, violations, details), unverified
