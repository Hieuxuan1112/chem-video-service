"""Code-owned chemistry reference data (spec §5.1).

The LLM may mention these facts but never supplies them: every number drawn on screen and every
numeric claim check comes from this module.

pH reference values
-------------------
Source: US EPA, "pH of Common Substances" chart in the Gold King Mine pH data summary
(document 1622624, data as of 2015-08-06),
https://www.epa.gov/sites/default/files/2015-10/documents/1622624.pdf, retrieved 2026-09-13.
The spec named the USGS "pH and Water" page, but it refused automated access (HTTP 403) on
2026-09-13; the plan allows another published pH chart.

- The chart places substances on whole pH units, so each value carries a tolerance of 0.5
  (its rounding resolution). Pure water is exact: the same document states "the pH of pure water
  is 7".
- Only substances printed directly under a whole-unit box are included. Labels printed between
  boxes (e.g. "Coke and Pepsi", "Soapy Water") and ranges (e.g. "Shampoos (7.0 to 10.0)") are
  omitted rather than guessed.
- Blood is omitted on purpose: the chart groups it with pure water at 7, but blood is slightly
  basic, and a neutral marker would teach the wrong classification.

CPK colours are the Jmol element palette; they are display choices, not chemistry facts.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

EPA_PH_SOURCE = "https://www.epa.gov/sites/default/files/2015-10/documents/1622624.pdf"


class ElementSymbol(StrEnum):
    H = "H"
    C = "C"
    N = "N"
    O = "O"  # noqa: E741
    F = "F"
    Na = "Na"
    Mg = "Mg"
    Cl = "Cl"


@dataclass(frozen=True)
class Element:
    symbol: str
    name: str
    z: int
    shells: tuple[int, ...]
    category: Literal["metal", "nonmetal"]
    cpk: str

    @property
    def valence(self) -> int:
        return self.shells[-1]


ELEMENTS: Mapping[ElementSymbol, Element] = {
    ElementSymbol.H: Element("H", "hydrogen", 1, (1,), "nonmetal", "#FFFFFF"),
    ElementSymbol.C: Element("C", "carbon", 6, (2, 4), "nonmetal", "#909090"),
    ElementSymbol.N: Element("N", "nitrogen", 7, (2, 5), "nonmetal", "#3050F8"),
    ElementSymbol.O: Element("O", "oxygen", 8, (2, 6), "nonmetal", "#FF0D0D"),
    ElementSymbol.F: Element("F", "fluorine", 9, (2, 7), "nonmetal", "#90E050"),
    ElementSymbol.Na: Element("Na", "sodium", 11, (2, 8, 1), "metal", "#AB5CF2"),
    ElementSymbol.Mg: Element("Mg", "magnesium", 12, (2, 8, 2), "metal", "#8AFF00"),
    ElementSymbol.Cl: Element("Cl", "chlorine", 17, (2, 8, 7), "nonmetal", "#1FF01F"),
}


class Molecule(StrEnum):
    H2 = "H2"
    Cl2 = "Cl2"
    F2 = "F2"
    O2 = "O2"
    N2 = "N2"
    HCl = "HCl"


@dataclass(frozen=True)
class MoleculeData:
    formula: Molecule
    atoms: tuple[ElementSymbol, ElementSymbol]
    shared_pairs: int
    display: str
    name: str


_E = ElementSymbol
MOLECULES: Mapping[Molecule, MoleculeData] = {
    Molecule.H2: MoleculeData(Molecule.H2, (_E.H, _E.H), 1, "H₂", "hydrogen"),
    Molecule.Cl2: MoleculeData(Molecule.Cl2, (_E.Cl, _E.Cl), 1, "Cl₂", "chlorine"),
    Molecule.F2: MoleculeData(Molecule.F2, (_E.F, _E.F), 1, "F₂", "fluorine"),
    Molecule.O2: MoleculeData(Molecule.O2, (_E.O, _E.O), 2, "O₂", "oxygen"),
    Molecule.N2: MoleculeData(Molecule.N2, (_E.N, _E.N), 3, "N₂", "nitrogen"),
    Molecule.HCl: MoleculeData(Molecule.HCl, (_E.H, _E.Cl), 1, "HCl", "hydrogen chloride"),
}


class IonicCompound(StrEnum):
    NaCl = "NaCl"
    NaF = "NaF"
    MgO = "MgO"


@dataclass(frozen=True)
class IonicData:
    formula: IonicCompound
    donor: ElementSymbol
    acceptor: ElementSymbol
    electrons: int
    donor_ion: str
    acceptor_ion: str
    display: str
    name: str


IONIC_COMPOUNDS: Mapping[IonicCompound, IonicData] = {
    IonicCompound.NaCl: IonicData(
        IonicCompound.NaCl, _E.Na, _E.Cl, 1, "Na+", "Cl-", "NaCl", "sodium chloride"
    ),
    IonicCompound.NaF: IonicData(
        IonicCompound.NaF, _E.Na, _E.F, 1, "Na+", "F-", "NaF", "sodium fluoride"
    ),
    IonicCompound.MgO: IonicData(
        IonicCompound.MgO, _E.Mg, _E.O, 2, "Mg2+", "O2-", "MgO", "magnesium oxide"
    ),
}


class SubstanceName(StrEnum):
    BATTERY_ACID = "battery_acid"
    STOMACH_ACID = "stomach_acid"
    LEMON_JUICE = "lemon_juice"
    VINEGAR = "vinegar"
    ORANGE_JUICE = "orange_juice"
    TOMATO_JUICE = "tomato_juice"
    BLACK_COFFEE = "black_coffee"
    MILK = "milk"
    PURE_WATER = "pure_water"
    SEAWATER = "seawater"
    BAKING_SODA = "baking_soda"
    HAND_SOAP = "hand_soap"
    MILK_OF_MAGNESIA = "milk_of_magnesia"
    HOUSEHOLD_AMMONIA = "household_ammonia"
    BLEACH = "bleach"
    DRAIN_CLEANER = "drain_cleaner"


@dataclass(frozen=True)
class SubstancePH:
    name: SubstanceName
    display: str
    ph: float
    tolerance: float
    aliases: tuple[str, ...]
    source: str


def _sub(
    name: SubstanceName, display: str, ph: float, aliases: tuple[str, ...], tolerance: float = 0.5
) -> SubstancePH:
    return SubstancePH(name, display, ph, tolerance, aliases, EPA_PH_SOURCE)


_S = SubstanceName
SUBSTANCE_PH: Mapping[SubstanceName, SubstancePH] = {
    s.name: s
    for s in (
        _sub(_S.BATTERY_ACID, "Battery acid", 0, ("battery acid",)),
        _sub(_S.STOMACH_ACID, "Stomach acid", 1, ("stomach acid",)),
        _sub(_S.LEMON_JUICE, "Lemon juice", 2, ("lemon juice",)),
        _sub(_S.VINEGAR, "Vinegar", 2, ("vinegar",)),
        _sub(_S.ORANGE_JUICE, "Orange juice", 3, ("orange juice",)),
        _sub(_S.TOMATO_JUICE, "Tomato juice", 4, ("tomato juice",)),
        _sub(_S.BLACK_COFFEE, "Black coffee", 5, ("black coffee", "coffee")),
        _sub(_S.MILK, "Milk", 6, ("milk",)),
        _sub(_S.PURE_WATER, "Pure water", 7, ("pure water",), tolerance=0.0),
        _sub(_S.SEAWATER, "Seawater", 8, ("seawater", "sea water")),
        _sub(_S.BAKING_SODA, "Baking soda", 8, ("baking soda",)),
        _sub(_S.HAND_SOAP, "Hand soap", 9, ("hand soap",)),
        _sub(_S.MILK_OF_MAGNESIA, "Milk of magnesia", 10, ("milk of magnesia",)),
        _sub(_S.HOUSEHOLD_AMMONIA, "Household ammonia", 11, ("household ammonia", "ammonia")),
        _sub(_S.BLEACH, "Bleach", 13, ("bleach",)),
        _sub(_S.DRAIN_CLEANER, "Drain cleaner", 14, ("liquid drain cleaner", "drain cleaner")),
    )
}


def ph_class(ph: float) -> Literal["acidic", "neutral", "basic"]:
    if ph < 7:
        return "acidic"
    if ph > 7:
        return "basic"
    return "neutral"
