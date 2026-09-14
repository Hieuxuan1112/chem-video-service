import pytest

from app.config import Settings
from app.cost.budget import SpendGuard
from app.cost.ledger import Ledger
from app.domain.errors import BudgetRefused
from app.pipeline.gates.g1_schema import g1_validate
from app.pipeline.storyboard import provider_json_schema, strict_json_schema
from app.providers.errors import ProviderError
from app.providers.llm.gemini import GeminiLLMProvider
from app.providers.llm.protocol import LLMRequest
from app.providers.retry import RetryPolicy
from app.topics.registry import TOPICS_ROOT, TopicRegistry

pytestmark = pytest.mark.live

SYSTEM = (
    "You write short chemistry explainer storyboards for students as JSON that matches the"
    " response schema exactly."
)
# The provider schema carries no bounds (spec §5.3), so the prompt states them; G1 enforces them.
LIMITS = (
    "Limits, checked after generation (a storyboard that breaks any of them is rejected):"
    " 5 to 7 scenes; title at most 60 characters; learning_objective at most 160 characters;"
    " on_screen_title 1 to 40 characters; narration 25 to 60 words per scene and at most 600"
    " characters; title_card subtitle at most 60 characters; key_points 2 to 4 points of at most"
    " 60 characters each; ph_scale 2 to 5 markers; log_ratio from_ph and to_ph are integers from"
    " 0 to 14; atom_shells 1 or 2 elements; comparison_table headers at most 16 characters, 2 to"
    " 4 rows, aspect at most 20 characters, left and right at most 28 characters. Include only"
    " the fields of the chosen visual type."
)


def test_writer_schema_roundtrip_live(
    live_settings: Settings, live_guard: SpendGuard, live_ledger: Ledger
) -> None:
    s = live_settings
    assert s.google_api_key is not None
    example = TopicRegistry.load(TOPICS_ROOT).curated_storyboard("ph_scale")
    prompt = (
        "Topic id: ph_scale. Write a storyboard that explains the pH scale. Keep the same"
        " structure, scene count and visual types as this valid example, but reword the"
        f" narration.\n{LIMITS}\nExample:\n{example.model_dump_json()}"
    )
    provider = GeminiLLMProvider(
        s.google_api_key.get_secret_value(),
        live_guard,
        RetryPolicy.from_settings(s),
        timeout_s=s.provider_timeout_s,
    )

    def request(schema: dict[str, object]) -> LLMRequest:
        return LLMRequest(
            operation="writer",
            model=s.writer_model,
            system=SYSTEM,
            prompt=prompt,
            json_schema=schema,
            temperature=s.writer_temperature,
            max_output_tokens=s.writer_max_output_tokens,
            thinking_budget=s.writer_thinking_budget,
        )

    before = live_ledger.project_totals().committed_usd
    last_id = (
        live_ledger._conn().execute("SELECT COALESCE(MAX(id), 0) FROM cost_entries").fetchone()[0]
    )
    path = "strict"
    try:
        try:
            response = provider.generate_json(request(strict_json_schema()))
        except ProviderError as exc:
            if exc.kind != "invalid_request" or "schema" not in exc.message.lower():
                raise
            print(f"strict schema rejected: {exc.message}")
            path = "flattened"
            response = provider.generate_json(request(provider_json_schema()))
    except BudgetRefused as exc:
        pytest.skip(str(exc))

    delta = live_ledger.project_totals().committed_usd - before
    for row in live_ledger._conn().execute(
        "SELECT id, status, projected_usd, usd, cost_source, error_code, error_message"
        " FROM cost_entries WHERE id > ? ORDER BY id",
        (last_id,),
    ):
        print("ledger row:", dict(row))
    print(f"schema path={path} usage={response.usage} ledger delta=${delta:.6f}")
    _, result = g1_validate(response.text)
    assert result.passed, [v.message for v in result.violations]
    assert delta < 0.02
