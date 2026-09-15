"""Manim runner CLI: one process renders the requested scenes and writes a result per scene.

python -m app.render.manim.runner --plan plan.json --out DIR [--scenes 0,3] [--fonts assets/fonts]
"""

import argparse
import json
import shutil
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from manim import tempconfig

from app.render.manim import theme
from app.render.manim.scene_factory import (
    LayoutAssertionError,
    frames_written,
    make_scene_class,
    set_fonts,
)
from app.render.plan import RenderPlan

_TRACEBACK_TAIL_CHARS = 2000


def _write_result(out: Path, index: int, **fields: Any) -> None:
    result = {"ok": False, "frames": None, "error_type": None, "message": None}
    result.update(fields)
    result.setdefault("traceback_tail", None)
    (out / f"scene-{index:02d}.result.json").write_text(json.dumps(result), encoding="utf-8")


def render_scene(plan: RenderPlan, index: int, out: Path) -> None:
    planned = plan.scenes[index]
    media = out / "_media" / f"scene-{index:02d}"
    try:
        scene_class = make_scene_class(planned, plan)
        with tempconfig(
            {
                "pixel_width": plan.width,
                "pixel_height": plan.height,
                "frame_rate": plan.fps,
                "media_dir": str(media),
                "output_file": f"scene-{index:02d}",
                "disable_caching": True,
                "background_color": theme.BACKGROUND,
                "progress_bar": "none",
                "verbosity": "WARNING",
            }
        ):
            scene = scene_class()
            scene.render()
            frames = frames_written(scene)
            movie = Path(scene.renderer.file_writer.movie_file_path)
        shutil.move(str(movie), out / f"scene-{index:02d}.mp4")
        _write_result(out, index, ok=True, frames=frames)
    except LayoutAssertionError as exc:
        _write_result(
            out,
            index,
            error_type="layout_error",
            message=str(exc),
            traceback_tail=traceback.format_exc()[-_TRACEBACK_TAIL_CHARS:],
        )
    except Exception as exc:
        _write_result(
            out,
            index,
            error_type="template_error",
            message=f"{type(exc).__name__}: {exc}",
            traceback_tail=traceback.format_exc()[-_TRACEBACK_TAIL_CHARS:],
        )
    finally:
        shutil.rmtree(media, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.render.manim.runner")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", default=None, help="comma-separated scene indices")
    parser.add_argument("--fonts", type=Path, default=Path("assets/fonts"))
    args = parser.parse_args(argv)

    plan = RenderPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    set_fonts(theme.register_fonts(args.fonts))
    args.out.mkdir(parents=True, exist_ok=True)
    indices = [int(i) for i in args.scenes.split(",")] if args.scenes else range(len(plan.scenes))
    for index in indices:
        render_scene(plan, index, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
