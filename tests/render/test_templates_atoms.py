from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from app.render.manim.templates._atoms import bohr_atom, ion_label, shell_radius_px
from manim import Dot

from app.pipeline.storyboard import ElectronSharing, ElectronTransfer
from app.render.manim import theme
from app.render.manim.scene_factory import assert_inside_safe_area
from app.render.plan import (
    ResolvedAtom,
    ResolvedAtomShells,
    ResolvedElectronSharing,
    ResolvedElectronTransfer,
    resolve_visual,
)
from app.topics.chemistry_data import ELEMENTS, IonicCompound, Molecule
from tests.render.harness import (
    Built,
    assert_no_text_overlap,
    build_offscreen,
    load_fonts,
    plan_for,
    tagged,
    texts,
)

pytestmark = pytest.mark.render

QUESTION = "What is a covalent bond?"


def _atom(symbol: str) -> ResolvedAtom:
    e = ELEMENTS[symbol]  # type: ignore[index]
    return ResolvedAtom(
        symbol=e.symbol,
        name=e.name,
        shells=list(e.shells),
        valence=e.valence,
        cpk=e.cpk,
        category=e.category,
    )


def _resolve(visual: Any) -> Any:
    return resolve_visual(visual, question=QUESTION, title="Bonds")


@pytest.fixture
def build(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], Built]:
    load_fonts()
    return lambda visual: build_offscreen(plan_for(visual, question=QUESTION), monkeypatch)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


@pytest.mark.parametrize("symbol", [e.value for e in ELEMENTS])
def test_electron_counts_equal_shells(symbol: str) -> None:
    load_fonts()
    atom = _atom(symbol)
    group = bohr_atom(atom, radius_px=shell_radius_px(len(atom.shells)))

    dots = [m for m in group.get_family() if isinstance(m, Dot)]
    per_shell = [sum(d.shell_index == k for d in dots) for k in range(len(atom.shells))]
    assert per_shell == atom.shells
    assert len(dots) == ELEMENTS[symbol].z  # type: ignore[index]
    center = group.nucleus.get_center()
    for dot in dots:
        radius = group.shell_circles[dot.shell_index].radius
        assert abs(_distance(dot.get_center(), center) - radius) < 1e-6


@pytest.mark.parametrize("symbol", [e.value for e in ELEMENTS])
@pytest.mark.parametrize("slots", [None, 8])
def test_valence_dots_count(symbol: str, slots: int | None) -> None:
    load_fonts()
    atom = _atom(symbol)
    if slots is not None and atom.valence > slots:
        pytest.skip("more electrons than slots")
    group = bohr_atom(atom, radius_px=shell_radius_px(len(atom.shells)), valence_slots=slots)

    assert len(group.valence_dots) == atom.valence
    assert all(d.get_color().to_hex().upper() == theme.ELECTRON for d in group.valence_dots)
    assert len(group.gap_offsets) == (0 if slots is None else slots - atom.valence)


@pytest.mark.parametrize("molecule", list(Molecule))
def test_sharing_moves_exactly_two_per_pair(
    build: Callable[[Any], Built], molecule: Molecule
) -> None:
    visual = _resolve(ElectronSharing(type="electron_sharing", molecule=molecule))
    assert isinstance(visual, ResolvedElectronSharing)
    scene = build(visual).scene

    atoms = sorted(tagged(scene, "valence_dots"), key=lambda a: a.nucleus.get_center()[0])
    assert [a.atom_symbol for a in atoms] == [a.symbol for a in visual.atoms]
    shared = tagged(scene, "shared")
    assert len(shared) == 2 * visual.shared_pairs
    for dot in shared:
        for atom in atoms:
            outer = atom.shell_circles[-1].radius
            assert _distance(dot.get_center(), atom.nucleus.get_center()) <= outer + 1e-6
    dots = [m for a in atoms for m in a.get_family() if isinstance(m, Dot)]
    assert len(dots) == sum(ELEMENTS[a.symbol].z for a in visual.atoms)  # type: ignore[index]


@pytest.mark.parametrize("compound", list(IonicCompound))
def test_transfer_moves_resolved_electron_count(
    build: Callable[[Any], Built], compound: IonicCompound
) -> None:
    visual = _resolve(ElectronTransfer(type="electron_transfer", compound=compound))
    assert isinstance(visual, ResolvedElectronTransfer)
    scene = build(visual).scene

    atoms = {a.atom_symbol: a for a in tagged(scene, "valence_dots")}
    donor, acceptor = atoms[visual.donor.symbol], atoms[visual.acceptor.symbol]
    moved = tagged(scene, "transferred")
    assert len(moved) == visual.electrons
    acceptor_radius = acceptor.shell_circles[-1].radius
    for dot in moved:
        assert (
            abs(_distance(dot.get_center(), acceptor.nucleus.get_center()) - acceptor_radius) < 1e-6
        )
    staying = [d for d in donor.valence_dots if not hasattr(d, "transferred")]
    assert len(staying) == visual.donor.valence - visual.electrons
    drawn = [t.original_text for t in texts(scene)]
    assert ion_label(visual.donor_ion) in drawn
    assert ion_label(visual.acceptor_ion) in drawn


def test_ion_label_uses_superscripts() -> None:
    assert ion_label("Na+") == "Na\u207a"
    assert ion_label("Mg2+") == "Mg\u00b2\u207a"
    assert ion_label("O2-") == "O\u00b2\u207b"


LARGEST = [
    ResolvedAtomShells(type="atom_shells", atoms=[_atom("Cl"), _atom("Mg")]),
    ResolvedAtomShells(type="atom_shells", atoms=[_atom("Cl")]),
    _resolve(ElectronSharing(type="electron_sharing", molecule=Molecule.Cl2)),
    _resolve(ElectronSharing(type="electron_sharing", molecule=Molecule.H2)),
    _resolve(ElectronSharing(type="electron_sharing", molecule=Molecule.HCl)),
    _resolve(ElectronTransfer(type="electron_transfer", compound=IonicCompound.NaCl)),
    _resolve(ElectronTransfer(type="electron_transfer", compound=IonicCompound.MgO)),
]


@pytest.mark.parametrize("visual", LARGEST, ids=lambda v: v.type)
def test_inside_safe_area_for_largest_atoms(build: Callable[[Any], Built], visual: Any) -> None:
    result = build(visual)

    assert_inside_safe_area(result.scene.mobjects)
    assert_no_text_overlap(result.scene)
    assert min(g.text_scale for g in result.groups) >= 0.9
