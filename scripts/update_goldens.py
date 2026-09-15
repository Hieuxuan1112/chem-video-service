"""Regenerate the golden template stills in tests/golden/.

Run inside the Docker image (Task 11.3) so fonts and Cairo match CI:
    python scripts/update_goldens.py [--out tests/golden]
"""

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.render.golden_fixtures import GOLDEN_DIR, render_goldens  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=GOLDEN_DIR)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as work:
        stills = render_goldens(args.out, Path(work))
    for template, path in stills.items():
        print(f"{template}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
