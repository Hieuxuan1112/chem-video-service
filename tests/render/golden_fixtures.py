"""Golden snapshot fixtures: one real scene per template, rendered small and hashed perceptually.

The fixture scenes are the first scene of each visual type in the curated storyboards, with fixed
clip durations, so beats and every drawn value are deterministic.
"""

from pathlib import Path

from PIL import Image

from app.config import Settings
from app.render.ffmpeg import run_ffmpeg
from app.render.manim.scene_factory import FADE_OUT_S
from app.render.manim_renderer import ManimRenderer
from app.render.plan import PlannedScene, RenderPlan, build_render_plan
from app.topics.registry import TOPICS_ROOT, TopicRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
TEMPLATES = (
    "title_card", "key_points", "comparison_table", "ph_scale", "log_ratio",
    "atom_shells", "electron_sharing", "electron_transfer",
)  # fmt: skip
TOPICS = ("ph_scale", "covalent_bonds", "ionic_vs_covalent")
CLIP_S = 5.0
MAX_HAMMING = 6
SETTINGS = Settings(_env_file=None, width=640, height=360, fps=15, render_timeout_s=1800.0)  # type: ignore[call-arg]


def golden_plan() -> RenderPlan:
    registry = TopicRegistry.load(TOPICS_ROOT)
    first: dict[str, PlannedScene] = {}
    for topic in TOPICS:
        storyboard = registry.curated_storyboard(topic)
        plan = build_render_plan(
            job_id="golden",
            question=registry.get(topic).title,
            storyboard=storyboard,
            clip_durations_s=[CLIP_S] * len(storyboard.scenes),
            settings=SETTINGS,
        )
        for scene in plan.scenes:
            first.setdefault(scene.visual.type, scene)
    scenes: list[PlannedScene] = []
    start = 0
    for i, template in enumerate(TEMPLATES):
        scene = first[template]
        end = start + scene.frames
        scenes.append(scene.model_copy(update={"index": i, "start_frame": start, "end_frame": end}))
        start = end
    return RenderPlan(
        job_id="golden",
        width=SETTINGS.width,
        height=SETTINGS.height,
        fps=SETTINGS.fps,
        question="How does the pH scale work?",
        total_frames=start,
        scenes=scenes,
    )


def still_frame_index(plan: RenderPlan, index: int) -> int:
    """The last frame before the fade-out (the very last frame is almost empty)."""
    return plan.scenes[index].frames - max(1, round(FADE_OUT_S * plan.fps)) - 1


def render_goldens(out_dir: Path, work_dir: Path) -> dict[str, Path]:
    """Renders every fixture scene and writes <template>.png stills into out_dir."""
    plan = golden_plan()
    result = ManimRenderer(SETTINGS).render(plan, work_dir)
    failed = [f"{s.index}: {s.error_type} {s.message}" for s in result.failed()]
    if failed:
        raise RuntimeError(f"golden render failed: {failed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stills = {}
    for scene, template in zip(result.scenes, TEMPLATES, strict=True):
        assert scene.path is not None
        png = out_dir / f"{template}.png"
        frame = still_frame_index(plan, scene.index)
        run_ffmpeg(
            ["-i", str(scene.path), "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", str(png)]
        )
        stills[template] = png
    return stills


def average_hash(png: Path) -> int:
    """64-bit average hash: 8x8 greyscale, one bit per pixel brighter than the mean."""
    with Image.open(png) as image:
        pixels = list(image.convert("L").resize((8, 8), Image.Resampling.BOX).getdata())
    mean = sum(pixels) / len(pixels)
    return sum(1 << i for i, p in enumerate(pixels) if p > mean)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()
