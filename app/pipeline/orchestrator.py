"""VideoPipeline: the resumable job handler (spec 4, 6, 8, 9).

Each stage is checkpointed and fenced by the job lease. On a claim, valid checkpoints are reused,
so a retried or reclaimed job converges to one published video without paying twice.
"""

import contextvars
import json
import shutil
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.artifacts.store import ArtifactStore, Manifest
from app.config import Settings
from app.cost.ledger import Ledger
from app.cost.pricing import render_compute_estimate
from app.domain.errors import AppError, ErrorCode
from app.domain.job import Stage
from app.jobs.worker import JobContext
from app.observability.context import stage_context
from app.pipeline.checkpoints import CheckpointStore, Versions, file_sha256, stable_hash
from app.pipeline.prompts import PROMPT_VERSIONS
from app.pipeline.storyboard import SCHEMA_VERSION, Storyboard
from app.pipeline.voicing import VoicedClip, Voicer
from app.pipeline.writer import StoryboardWriter
from app.render.assembler import MediaAssembler
from app.render.layout import FontSet
from app.render.media_qa import MediaQA
from app.render.plan import build_render_plan
from app.render.protocol import Renderer, SceneRenderResult
from app.topics.registry import TopicRegistry

RENDER_FILES = ("video.mp4", "narration.wav", "captions.vtt", "thumbnail.jpg", "contact_sheet.jpg")


class EventRelay:
    """Writer and voicer are shared across jobs; their events go to the job running in this context
    (pool threads inherit it through contextvars.copy_context)."""

    def __init__(self) -> None:
        self._ctx: contextvars.ContextVar[tuple[JobContext, str] | None] = contextvars.ContextVar(
            "pipeline_job", default=None
        )

    @contextmanager
    def bind(self, ctx: JobContext, stage: Stage) -> Iterator[None]:
        token = self._ctx.set((ctx, str(stage)))
        try:
            yield
        finally:
            self._ctx.reset(token)

    def emit(self, type: str, data: dict[str, Any]) -> None:
        bound = self._ctx.get()
        if bound is not None:
            bound[0].emit(type, bound[1], data)


@dataclass
class PipelineDeps:
    settings: Settings
    registry: TopicRegistry
    writer: StoryboardWriter
    voicer: Voicer
    renderer: Renderer
    assembler: MediaAssembler
    qa: MediaQA
    checkpoints: CheckpointStore
    artifacts: ArtifactStore
    ledger: Ledger
    fonts: FontSet
    events: EventRelay = field(default_factory=EventRelay)


def render_error_to_app_error(failed: Sequence[SceneRenderResult]) -> AppError:
    """Non-retryable code bugs win over retryable infrastructure failures."""
    details = {
        "scenes": [
            {"index": s.index, "error_type": s.error_type, "message": s.message} for s in failed
        ]
    }
    types = {s.error_type for s in failed}
    indices = ", ".join(str(s.index) for s in failed)
    if "template_error" in types:
        code, cause = ErrorCode.RENDER_TEMPLATE_ERROR, "deterministic_bug"
    elif "layout_error" in types:
        code, cause = ErrorCode.RENDER_LAYOUT_ERROR, "deterministic_bug"
    elif "timeout" in types:
        code, cause = ErrorCode.RENDER_TIMEOUT, "timeout"
    else:
        code, cause = ErrorCode.RENDER_FAILED, "subprocess_crash"
    return AppError(code, cause, f"rendering failed for scene(s) {indices}", details)


def _clips_from_payload(directory: Path, payload: dict[str, Any]) -> list[VoicedClip]:
    return [
        VoicedClip(**{**c, "wav_path": directory / Path(c["wav_path"]).name})
        for c in payload["clips"]
    ]


class VideoPipeline:
    def __init__(self, deps: PipelineDeps) -> None:
        self._d = deps

    def _versions(self, topic_id: str) -> Versions:
        s = self._d.settings
        return Versions(
            pipeline=s.pipeline_version,
            schema=SCHEMA_VERSION,
            prompts=dict(PROMPT_VERSIONS),
            topic_pack=self._d.registry.get(topic_id).version,
            renderer=self._d.renderer.version,
            voice=f"{s.tts_model}/{s.tts_voice}/{s.tts_style_version}|edge/{s.edge_voice}",
        )

    @contextmanager
    def _stage(self, ctx: JobContext, stage: Stage, durations: dict[str, float]) -> Iterator[None]:
        ctx.check()
        ctx.set_stage(stage)
        started = time.monotonic()
        with stage_context(str(stage)), self._d.events.bind(ctx, stage):
            yield
        durations[str(stage)] = round(
            durations.get(str(stage), 0.0) + time.monotonic() - started, 3
        )

    def handle(self, ctx: JobContext) -> None:
        d = self._d
        job = ctx.job
        if self._recover(ctx):
            return
        versions = self._versions(job.topic_id)
        durations: dict[str, float] = {}

        # --- reviewing -----------------------------------------------------------------------
        review_hash = stable_hash(
            {
                "topic_id": job.topic_id,
                "query": job.query,
                "prompts": dict(PROMPT_VERSIONS),
                "topic_pack": versions.topic_pack,
            }
        )
        reviewed = d.checkpoints.load_valid(
            job.id, stage_key="reviewing", input_hash=review_hash, versions=versions
        )
        if reviewed is None:
            with self._stage(ctx, Stage.SCRIPTING, durations):
                outcome = d.writer.write(topic_id=job.topic_id, query=job.query)
            with self._stage(ctx, Stage.REVIEWING, durations):
                scratch = d.checkpoints.scratch_dir(ctx.lease, "reviewing")
                (scratch / "storyboard.json").write_text(
                    outcome.storyboard.model_dump_json(indent=2), encoding="utf-8"
                )
                writer_report = outcome.report()
                self._set_generation(ctx, writer_report)
                d.checkpoints.promote(
                    ctx.lease,
                    stage_key="reviewing",
                    input_hash=review_hash,
                    versions=versions,
                    files=["storyboard.json"],
                    payload=writer_report,
                )
            review_dir = scratch
        else:
            review_dir, writer_report = reviewed
            ctx.emit("stage_skipped_checkpoint", "reviewing", {"stage_key": "reviewing"})
            self._set_generation(ctx, writer_report)
        storyboard = Storyboard.model_validate_json(
            (review_dir / "storyboard.json").read_text(encoding="utf-8")
        )

        # --- voicing -------------------------------------------------------------------------
        pack = d.registry.get(job.topic_id)
        voice_hash = stable_hash(
            {
                "storyboard": storyboard.model_dump(),
                "voice": versions.voice,
                "pronunciations": pack.pronunciations,
            }
        )
        voiced = d.checkpoints.load_valid(
            job.id, stage_key="voicing", input_hash=voice_hash, versions=versions
        )
        if voiced is None:
            with self._stage(ctx, Stage.VOICING, durations):
                scratch = d.checkpoints.scratch_dir(ctx.lease, "voicing")
                voicing = d.voicer.voice(storyboard=storyboard, pack=pack, out_dir=scratch)
                voicing_report = voicing.report()
                d.checkpoints.promote(
                    ctx.lease,
                    stage_key="voicing",
                    input_hash=voice_hash,
                    versions=versions,
                    files=[c.wav_path.name for c in voicing.clips],
                    payload=voicing_report,
                )
            clips = list(voicing.clips)
        else:
            voice_dir, voicing_report = voiced
            ctx.emit("stage_skipped_checkpoint", "voicing", {"stage_key": "voicing"})
            clips = _clips_from_payload(voice_dir, voicing_report)

        # --- rendering and assembly ------------------------------------------------------------
        plan = build_render_plan(
            job_id=job.id,
            question=job.query,
            storyboard=storyboard,
            clip_durations_s=[c.duration_s for c in clips],
            settings=d.settings,
        )
        clip_hashes = [file_sha256(c.wav_path) for c in clips]
        assembly_hash = stable_hash({"plan": plan.model_dump(), "clips": clip_hashes})
        assembled = d.checkpoints.load_valid(
            job.id, stage_key="rendering", input_hash=assembly_hash, versions=versions
        )
        if assembled is None:
            with self._stage(ctx, Stage.RENDERING, durations):
                scene_videos = self._render_scenes(ctx, plan, versions)
                scratch = d.checkpoints.scratch_dir(ctx.lease, "rendering")
                result = d.assembler.assemble(
                    plan=plan,
                    clips=clips,
                    scene_videos=scene_videos,
                    workdir=scratch,
                    cancel=ctx.lease_lost,
                )
                ctx.check()
                d.checkpoints.promote(
                    ctx.lease,
                    stage_key="rendering",
                    input_hash=assembly_hash,
                    versions=versions,
                    files=list(RENDER_FILES),
                    payload={"frame_fixes": result.frame_fixes},
                )
            render_dir = scratch
        else:
            render_dir, _ = assembled
            ctx.emit("stage_skipped_checkpoint", "rendering", {"stage_key": "rendering"})

        # --- quality check ---------------------------------------------------------------------
        qa_hash = stable_hash(
            {"video": file_sha256(render_dir / "video.mp4"), "plan": plan.model_dump()}
        )
        checked = d.checkpoints.load_valid(
            job.id, stage_key="quality_check", input_hash=qa_hash, versions=versions
        )
        if checked is None:
            with self._stage(ctx, Stage.QUALITY_CHECK, durations):
                report = d.qa.check(video=render_dir / "video.mp4", plan=plan)
                scratch = d.checkpoints.scratch_dir(ctx.lease, "quality_check")
                quality = self._quality_report(writer_report, voicing_report, report.to_dict())
                (scratch / "quality_report.json").write_text(
                    json.dumps(quality, indent=2), encoding="utf-8"
                )
                ctx.emit("media_qa", str(Stage.QUALITY_CHECK), report.to_dict())
                d.qa.raise_for(report)
                d.checkpoints.promote(
                    ctx.lease,
                    stage_key="quality_check",
                    input_hash=qa_hash,
                    versions=versions,
                    files=["quality_report.json"],
                    payload={"passed": True},
                )
            qa_dir = scratch
        else:
            qa_dir, _ = checked
            ctx.emit("stage_skipped_checkpoint", "quality_check", {"stage_key": "quality_check"})

        # --- publishing ------------------------------------------------------------------------
        with self._stage(ctx, Stage.PUBLISHING, durations):
            ctx.renew_lease()
            files = {
                "video.mp4": render_dir / "video.mp4",
                "thumbnail.jpg": render_dir / "thumbnail.jpg",
                "captions.vtt": render_dir / "captions.vtt",
                "contact_sheet.jpg": render_dir / "contact_sheet.jpg",
                "storyboard.json": review_dir / "storyboard.json",
                "quality_report.json": qa_dir / "quality_report.json",
            }
            manifest = Manifest(
                job_id=job.id,
                pipeline_version=versions.pipeline,
                schema_version=versions.schema,
                prompt_versions=dict(versions.prompts),
                models={
                    "writer": d.settings.writer_model,
                    "critic": d.settings.critic_model,
                    "tts": d.settings.tts_model,
                    "asr": d.settings.asr_model,
                },
                voice={
                    "provider": voicing_report["provider"],
                    "voice": voicing_report["voice"],
                    "fallback_reason": voicing_report["voice_fallback_reason"],
                },
                renderer_version=versions.renderer,
                topic_pack_version=versions.topic_pack,
                generation_source=writer_report["generation_source"],
                fallback_reason=writer_report["fallback_reason"],
                checkpoint_attempts=d.checkpoints.attempts_map(job.id),
                cost=d.ledger.job_cost_summary(job.id),
                stage_durations_s=durations,
                files={},
                created_at=time.time(),
            )
            ctx.check()
            d.artifacts.publish(job.id, files, manifest, attempt=ctx.lease.attempt)
            ctx.emit("artifact_published", str(Stage.PUBLISHING), {"files": sorted(files)})

    def _recover(self, ctx: JobContext) -> bool:
        d = self._d
        manifest = d.artifacts.verify(ctx.job.id)
        if (
            manifest is None
            or manifest.pipeline_version != d.settings.pipeline_version
            or manifest.checkpoint_attempts != d.checkpoints.attempts_map(ctx.job.id)
        ):
            return False
        ctx.emit(
            "publish_recovered", str(Stage.PUBLISHING), {"attempts": manifest.checkpoint_attempts}
        )
        return True

    @staticmethod
    def _set_generation(ctx: JobContext, writer_report: dict[str, Any]) -> None:
        ctx.check()
        ctx.repo.set_generation(
            ctx.lease,
            generation_source=writer_report["generation_source"],
            fallback_reason=writer_report["fallback_reason"],
        )

    def _render_scenes(self, ctx: JobContext, plan: Any, versions: Versions) -> list[Path]:
        d = self._d
        videos: dict[int, Path] = {}
        hashes: dict[int, str] = {}
        for scene in plan.scenes:
            key = f"rendering/scene-{scene.index:02d}"
            hashes[scene.index] = stable_hash(
                {
                    "scene": scene.model_dump(),
                    "width": plan.width,
                    "height": plan.height,
                    "fps": plan.fps,
                    "theme": plan.theme,
                    "scenes": len(plan.scenes),
                }
            )
            valid = d.checkpoints.load_valid(
                ctx.job.id, stage_key=key, input_hash=hashes[scene.index], versions=versions
            )
            if valid is not None:
                videos[scene.index] = valid[0] / "scene.mp4"
        missing = [s.index for s in plan.scenes if s.index not in videos]
        if len(missing) < len(plan.scenes):
            ctx.emit(
                "stage_skipped_checkpoint",
                str(Stage.RENDERING),
                {"stage_key": "rendering/scene-*", "reused": sorted(videos)},
            )
        if missing:
            out = d.checkpoints.scratch_dir(ctx.lease, "rendering/out")
            result = d.renderer.render(plan, out, scenes=missing, cancel=ctx.lease_lost)
            ctx.check()
            d.ledger.record_production_estimate(
                operation="render_compute",
                usd=render_compute_estimate(result.cpu_seconds),
                units={"cpu_seconds": result.cpu_seconds, "scenes": len(missing)},
                job_id=ctx.job.id,
                attempt=ctx.lease.attempt,
                stage=str(Stage.RENDERING),
            )
            for scene in result.scenes:
                if not scene.ok or scene.path is None:
                    continue
                key = f"rendering/scene-{scene.index:02d}"
                scratch = d.checkpoints.scratch_dir(ctx.lease, key)
                shutil.copyfile(scene.path, scratch / "scene.mp4")
                d.checkpoints.promote(
                    ctx.lease,
                    stage_key=key,
                    input_hash=hashes[scene.index],
                    versions=versions,
                    files=["scene.mp4"],
                    payload={"frames": scene.frames, "message": scene.message},
                )
                videos[scene.index] = scratch / "scene.mp4"
            failed = result.failed()
            if failed:
                raise render_error_to_app_error(failed)
        return [videos[s.index] for s in plan.scenes]

    @staticmethod
    def _quality_report(
        writer: dict[str, Any], voicing: dict[str, Any], media_qa: dict[str, Any]
    ) -> dict[str, Any]:
        clips = [{k: v for k, v in c.items() if k != "wav_path"} for c in voicing["clips"]]
        return {
            "summary": {
                "generation_source": writer["generation_source"],
                "fallback_reason": writer["fallback_reason"],
                "writer_attempts": len(writer["attempts"]),
                "critic_scores": writer["critic_scores"],
                "critic_skipped": writer["critic_skipped"],
                "asr_skipped": voicing["asr_skipped"],
                "voice_provider": voicing["provider"],
                "media_qa_passed": media_qa["passed"],
            },
            "writer": writer,
            "voicing": {**{k: v for k, v in voicing.items() if k != "clips"}, "clips": clips},
            "media_qa": media_qa,
        }
