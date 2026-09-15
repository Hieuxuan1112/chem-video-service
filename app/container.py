"""Dependency container: plain constructor wiring, no registry."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from app.config import Settings
from app.jobs.clock import Clock
from app.jobs.repository import JobRepository
from app.jobs.worker import Worker
from app.topics.resolver import TopicResolver

if TYPE_CHECKING:
    from app.providers.llm.protocol import LLMRequest, LLMResponse
    from app.topics.registry import TopicRegistry


class ArtifactStoreLike(Protocol):
    def open(self, job_id: str, name: str) -> Path | None: ...

    def list_refs(self, job_id: str) -> list[dict[str, Any]]: ...

    def manifest(self, job_id: str) -> dict[str, Any] | None: ...


@dataclass
class Container:
    settings: Settings
    clock: Clock
    repo: JobRepository
    resolver: TopicResolver
    artifacts: ArtifactStoreLike | None = None
    spend_state: Callable[[], dict[str, Any]] | None = None
    job_cost: Callable[[str], dict[str, Any]] | None = None
    worker: Worker | None = None


class CuratedFakeLLM:
    """No-network LLM: the writer gets the topic's curated storyboard, the critic a passing verdict.

    Used by build_container(fake_providers=True) for integration tests and chaos evaluation.
    """

    name = "fake"

    def __init__(self, registry: "TopicRegistry") -> None:
        self._registry = registry
        self.requests: list[LLMRequest] = []

    def generate_json(self, req: "LLMRequest") -> "LLMResponse":
        from app.providers.llm.protocol import LLMResponse, Usage

        self.requests.append(req)
        if req.operation == "writer":
            match = re.search(r'topic_id: "([a-z_]+)"', req.prompt)
            if match is None:
                raise AssertionError("writer prompt without topic_id")
            text = self._registry.curated_storyboard(match.group(1)).model_dump_json()
        else:
            scores = dict.fromkeys(("accuracy", "relevance", "clarity", "alignment"), 5)
            text = json.dumps({**scores, "issues": []})
        return LLMResponse(text=text, usage=Usage(1000, 500), model=req.model)


def build_container(settings: Settings | None = None, *, fake_providers: bool = False) -> Container:
    """Plain constructor wiring for the API process, the worker process and tests."""
    from app.artifacts.store import LocalArtifactStore
    from app.cost.budget import SpendGuard, assert_project_ledger
    from app.cost.ledger import Ledger
    from app.jobs.clock import SystemClock
    from app.pipeline.checkpoints import CheckpointStore
    from app.pipeline.critic import Critic
    from app.pipeline.orchestrator import EventRelay, PipelineDeps, VideoPipeline
    from app.pipeline.tts_cache import TTSCache
    from app.pipeline.voicing import Voicer
    from app.pipeline.writer import StoryboardWriter
    from app.providers.retry import RetryPolicy
    from app.providers.tts.edge import EdgeTTSProvider
    from app.providers.tts.fake import FakeTTS
    from app.render.assembler import MediaAssembler
    from app.render.fake import FakeRenderer
    from app.render.layout import FontSet
    from app.render.manim_renderer import ManimRenderer
    from app.render.media_qa import MediaQA
    from app.render.protocol import Renderer
    from app.topics.registry import TOPICS_ROOT, TopicRegistry
    from app.topics.resolver import KeywordResolver

    s = settings or Settings()
    clock = SystemClock()
    for directory in (s.database_path.parent, s.artifacts_dir, s.work_dir, s.cache_dir):
        directory.mkdir(parents=True, exist_ok=True)
    repo = JobRepository(s.database_path, clock)
    registry = TopicRegistry.load(TOPICS_ROOT)
    fonts = FontSet.from_dir(s.fonts_dir)
    ledger = Ledger(s.ledger_path(), clock)
    relay = EventRelay()
    guard = SpendGuard(ledger, s, clock, events=relay.emit)
    policy = RetryPolicy.from_settings(s)

    llm: Any
    primary: Any
    fallback: Any
    asr: Any
    renderer: Renderer
    if fake_providers:
        llm = CuratedFakeLLM(registry)
        primary, fallback, asr = FakeTTS(name="fake_tts"), FakeTTS(name="fake_tts_fallback"), None
        renderer = FakeRenderer()
    else:
        from app.providers.asr.gemini import GeminiASRProvider
        from app.providers.llm.gemini import GeminiLLMProvider
        from app.providers.tts.gemini import GeminiTTSProvider

        if s.google_api_key is None:
            raise RuntimeError("GOOGLE_API_KEY is required unless fake providers are used")
        assert_project_ledger(s)
        key = s.google_api_key.get_secret_value()
        llm = GeminiLLMProvider(key, guard, policy, timeout_s=s.provider_timeout_s)
        primary = GeminiTTSProvider(
            key,
            guard,
            policy,
            model=s.tts_model,
            voice=s.tts_voice,
            style_version=s.tts_style_version,
            timeout_s=s.provider_timeout_s,
            words_per_second=s.words_per_second,
            max_ratio=s.a2_max_ratio,
        )
        fallback = EdgeTTSProvider(
            ledger, policy, voice=s.edge_voice, timeout_s=s.provider_timeout_s
        )
        asr = GeminiASRProvider(
            key, guard, policy, model=s.asr_model, timeout_s=s.provider_timeout_s
        )
        renderer = ManimRenderer(s)

    artifacts = LocalArtifactStore(s.artifacts_dir)
    writer = StoryboardWriter(llm, Critic(llm, s), registry, s, fonts, emit=relay.emit)
    voicer = Voicer(primary, fallback, asr, TTSCache(s.cache_dir), guard, s, emit=relay.emit)
    pipeline = VideoPipeline(
        PipelineDeps(
            settings=s,
            registry=registry,
            writer=writer,
            voicer=voicer,
            renderer=renderer,
            assembler=MediaAssembler(s, fonts),
            qa=MediaQA(s),
            checkpoints=CheckpointStore(s.work_dir, repo),
            artifacts=artifacts,
            ledger=ledger,
            fonts=fonts,
            events=relay,
        )
    )
    return Container(
        settings=s,
        clock=clock,
        repo=repo,
        resolver=KeywordResolver(registry),
        artifacts=artifacts,
        spend_state=guard.state,
        job_cost=ledger.job_cost_summary,
        worker=Worker(repo, pipeline, s),
    )
