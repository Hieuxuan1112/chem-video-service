import pytest

from app.topics.chemistry_data import (
    ELEMENTS,
    IONIC_COMPOUNDS,
    MOLECULES,
    SUBSTANCE_PH,
    ElementSymbol,
    IonicCompound,
    Molecule,
    SubstanceName,
    ph_class,
)


def test_all_enum_members_have_data() -> None:
    assert set(ELEMENTS) == set(ElementSymbol)
    assert set(MOLECULES) == set(Molecule)
    assert set(IONIC_COMPOUNDS) == set(IonicCompound)
    assert set(SUBSTANCE_PH) == set(SubstanceName)
    assert {e.value for e in ElementSymbol} == {"H", "C", "N", "O", "F", "Na", "Mg", "Cl"}
    assert {m.value for m in Molecule} == {"H2", "Cl2", "F2", "O2", "N2", "HCl"}
    assert {c.value for c in IonicCompound} == {"NaCl", "NaF", "MgO"}


def test_shells_sum_to_z() -> None:
    for symbol, element in ELEMENTS.items():
        assert element.symbol == symbol.value
        assert sum(element.shells) == element.z


def test_valence_values() -> None:
    expected = {"H": 1, "C": 4, "N": 5, "O": 6, "F": 7, "Na": 1, "Mg": 2, "Cl": 7}
    assert {s.value: e.valence for s, e in ELEMENTS.items()} == expected


def test_categories() -> None:
    metals = {s.value for s, e in ELEMENTS.items() if e.category == "metal"}
    assert metals == {"Na", "Mg"}


def test_shared_pairs_complete_outer_shells() -> None:
    expected_pairs = {"H2": 1, "Cl2": 1, "F2": 1, "O2": 2, "N2": 3, "HCl": 1}
    for formula, data in MOLECULES.items():
        assert data.shared_pairs == expected_pairs[formula.value]
        for atom in data.atoms:
            full = 2 if atom is ElementSymbol.H else 8
            assert ELEMENTS[atom].valence + data.shared_pairs == full, (formula, atom)
            assert ELEMENTS[atom].category == "nonmetal"


def test_ionic_electrons_equal_donor_valence_and_acceptor_gap() -> None:
    expected = {"NaCl": (1, "Na+", "Cl-"), "NaF": (1, "Na+", "F-"), "MgO": (2, "Mg2+", "O2-")}
    for formula, data in IONIC_COMPOUNDS.items():
        donor, acceptor = ELEMENTS[data.donor], ELEMENTS[data.acceptor]
        assert (data.electrons, data.donor_ion, data.acceptor_ion) == expected[formula.value]
        assert donor.category == "metal" and acceptor.category == "nonmetal"
        assert data.electrons == donor.valence
        assert data.electrons == 8 - acceptor.valence


def test_ph_values_in_range_and_sourced() -> None:
    for name, sub in SUBSTANCE_PH.items():
        assert sub.name is name
        assert 0 <= sub.ph <= 14
        assert 0 <= sub.tolerance <= 1
        assert sub.source.startswith("https://")
        assert sub.aliases and all(a == a.lower() for a in sub.aliases)
    assert SUBSTANCE_PH[SubstanceName.PURE_WATER].ph == 7.0
    assert SUBSTANCE_PH[SubstanceName.LEMON_JUICE].ph < 7 < SUBSTANCE_PH[SubstanceName.BLEACH].ph


def test_aliases_unique_across_substances() -> None:
    seen: dict[str, SubstanceName] = {}
    for name, sub in SUBSTANCE_PH.items():
        for alias in sub.aliases:
            assert alias not in seen, (alias, seen.get(alias), name)
            seen[alias] = name


@pytest.mark.parametrize(
    ("ph", "cls"),
    [(0, "acidic"), (6.99, "acidic"), (7, "neutral"), (7.01, "basic"), (14, "basic")],
)
def test_ph_class_boundaries(ph: float, cls: str) -> None:
    assert ph_class(ph) == cls
