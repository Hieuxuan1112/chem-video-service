import json
import sqlite3

import pytest

from app.config import Settings
from app.cost.budget import SpendGuard
from app.cost.ledger import Ledger
from app.pipeline.critic import Critic
from app.pipeline.writer import StoryboardWriter
from app.providers.llm.gemini import GeminiLLMProvider
from app.providers.llm.protocol import LLMRequest, LLMResponse
from app.providers.retry import RetryPolicy
from app.render.layout import FontSet
from app.topics.registry import TOPICS_ROOT, TopicRegistry

pytestmark = pytest.mark.live

TOPICS = ["ph_scale", "covalent_bonds", "ionic_vs_covalent"]
_FEEDBACK_MARKER = "It was rejected for these problems:"


class Recording:
    """Keeps every request so the rewrite feedback sent to Gemini can be printed."""

    def __init__(self, inner: GeminiLLMProvider) -> None:
        self.inner = inner
        self.name = inner.name
        self.requests: list[LLMRequest] = []

    def generate_json(self, req: LLMRequest) -> LLMResponse:
        self.requests.append(req)
        return self.inner.generate_json(req)


@pytest.mark.parametrize("topic_id", TOPICS)
def test_writer_loop_live_each_topic(
    topic_id: str,
    live_settings: Settings,
    live_guard: SpendGuard,
    live_ledger: Ledger,
    fonts: FontSet,
) -> None:
    s = live_settings
    assert s.google_api_key is not None
    key = s.google_api_key.get_secret_value()
    policy = RetryPolicy.from_settings(s)
    writer_llm = Recording(
        GeminiLLMProvider(key, live_guard, policy, timeout_s=s.provider_timeout_s)
    )
    critic_llm = GeminiLLMProvider(key, live_guard, policy, timeout_s=s.provider_timeout_s)
    registry = TopicRegistry.load(TOPICS_ROOT)
    events: list[tuple[str, dict[str, object]]] = []
    writer = StoryboardWriter(
        writer_llm,
        Critic(critic_llm, s),
        registry,
        s,
        fonts,
        emit=lambda t, d: events.append((t, d)),
    )
    conn: sqlite3.Connection = live_ledger._conn()
    last_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM cost_entries").fetchone()[0]
    before = live_ledger.project_totals().committed_usd

    outcome = writer.write(topic_id=topic_id, query=registry.get(topic_id).canonical_query)

    for record in outcome.attempts:
        failed = [r for r in [record.g1, *record.g2] if not r["passed"]]
        print(
            f"attempt {record.attempt}: passed={record.passed} g1={record.g1['passed']}"
            f" g2={[r['passed'] for r in record.g2]} g3={record.g3}"
            f" critic_skipped={record.critic_skipped}"
        )
        for result in failed:
            for v in result["violations"]:
                print(f"  {result['gate']} {v['code']}: {v['message']}")
    for i, req in enumerate(writer_llm.requests, start=1):
        if _FEEDBACK_MARKER in req.prompt:
            print(f"feedback sent with writer call {i}:")
            print(req.prompt.split(_FEEDBACK_MARKER, 1)[1].split("Return the full", 1)[0].strip())
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT id, operation, model, status, projected_usd, usd, cost_source, units_json,"
            " error_code, error_message FROM cost_entries WHERE id > ? ORDER BY id",
            (last_id,),
        )
    ]
    for row in rows:
        print("ledger row:", json.dumps(row))
    delta = live_ledger.project_totals().committed_usd - before
    print(
        f"generation_source={outcome.generation_source} fallback_reason={outcome.fallback_reason}"
        f" critic_scores={outcome.critic_scores} unverified={outcome.unverified_claims}"
        f" provider_calls={len(rows)} ledger_delta=${delta:.6f}"
    )
    print("events:", [t for t, _ in events])

    assert outcome.storyboard is not None
    assert live_ledger.unsettled() == []
    assert delta <= 3 * 0.02
