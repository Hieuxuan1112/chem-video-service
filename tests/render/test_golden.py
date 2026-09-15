import itertools
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from tests.render.golden_fixtures import (
    GOLDEN_DIR,
    MAX_HAMMING,
    TEMPLATES,
    average_hash,
    golden_plan,
    hamming,
    render_goldens,
)

pytestmark = pytest.mark.render

MISSING = "golden missing; generate in Docker via scripts/update_goldens.py"


@pytest.fixture(scope="module")
def stills(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if not any((GOLDEN_DIR / f"{t}.png").exists() for t in TEMPLATES):
        pytest.skip(MISSING)
    work = tmp_path_factory.mktemp("golden")
    return render_goldens(work / "stills", work / "render")


@pytest.mark.parametrize("template", TEMPLATES)
def test_template_matches_golden(template: str, request: pytest.FixtureRequest) -> None:
    golden = GOLDEN_DIR / f"{template}.png"
    if not golden.exists():
        pytest.skip(MISSING)
    rendered: dict[str, Path] = request.getfixturevalue("stills")

    distance = hamming(average_hash(rendered[template]), average_hash(golden))
    assert distance <= MAX_HAMMING, f"{template}: hamming distance {distance}"


def test_golden_plan_covers_every_template_once() -> None:
    plan = golden_plan()
    assert [s.visual.type for s in plan.scenes] == list(TEMPLATES)
    assert all(a.end_frame == b.start_frame for a, b in itertools.pairwise(plan.scenes))
    assert plan.total_frames == plan.scenes[-1].end_frame


def _card(path: Path, boxes: list[tuple[int, int, int, int]]) -> Path:
    image = Image.new("RGB", (640, 360), "#0F172A")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, fill="#F8FAFC")
    image.save(path)
    return path


def test_average_hash_tolerates_small_changes_but_not_layout_changes(tmp_path: Path) -> None:
    base = _card(tmp_path / "base.png", [(40, 40, 600, 100), (80, 160, 320, 320)])
    nudged = _card(tmp_path / "nudged.png", [(42, 40, 602, 100), (80, 162, 320, 322)])
    moved = _card(tmp_path / "moved.png", [(40, 260, 600, 320), (360, 40, 600, 200)])

    assert hamming(average_hash(base), average_hash(base)) == 0
    assert hamming(average_hash(base), average_hash(nudged)) <= MAX_HAMMING
    assert hamming(average_hash(base), average_hash(moved)) > MAX_HAMMING
