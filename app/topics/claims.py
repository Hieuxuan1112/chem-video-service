"""Claim checkers for G2c (spec §5.5): numeric chemistry claims in text are verified against
`chemistry_data`. A clause that cannot be read unambiguously is never guessed: its numbers stay
unconsumed and the clause is reported as unverified (and forwarded to the critic)."""

import itertools
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from app.topics.chemistry_data import (
    ELEMENTS,
    IONIC_COMPOUNDS,
    MOLECULES,
    SUBSTANCE_PH,
    Molecule,
    SubstancePH,
    ph_class,
)

TIMES_SIGN = chr(0x00D7)  # multiplication sign, written after a multiplier ("10" + sign)
MINUS_SIGN = chr(0x2212)  # unicode minus sign, sometimes used in ion charges

Span = tuple[int, int]


@dataclass(frozen=True)
class ClaimFinding:
    checker: str
    clause: str
    ok: bool
    message: str
    spans: tuple[Span, ...] = field(default=(), compare=False)  # number tokens this claim used


@dataclass
class ClaimReport:
    findings: list[ClaimFinding]
    unverified: list[str]

    @property
    def violations(self) -> list[ClaimFinding]:
        return [f for f in self.findings if not f.ok]


NUMBER_WORDS: Mapping[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "hundred": 100, "thousand": 1000,
}  # fmt: skip

_CLAUSE_SPLIT = re.compile(r"\.(?!\d)|[;!?]|\b(?:while|whereas|but)\b", re.IGNORECASE)


def split_clauses(text: str) -> list[str]:
    return [c.strip(" ,") for c in _CLAUSE_SPLIT.split(text) if c.strip(" ,")]


# --- number tokens ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Token:
    start: int
    end: int
    value: float

    @property
    def span(self) -> Span:
        return (self.start, self.end)


_BIG_WORD = re.compile(r"\b(?:(?:one|a)\s+)?(hundred|thousand)\b", re.IGNORECASE)
_WORD = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)
_FOLD = re.compile(r"\b(ten|hundred|thousand)fold\b", re.IGNORECASE)
_DIGITS = re.compile(r"\d+(?:\.\d+)?")


def _tokens(clause: str) -> list[_Token]:
    """Numbers written as digits or words. Digits attached to letters (H2, Mg2+) are formulas."""
    found: list[_Token] = []
    taken: list[Span] = []

    def free(start: int, end: int) -> bool:
        return all(end <= s or start >= e for s, e in taken)

    def add(start: int, end: int, value: float) -> None:
        if free(start, end):
            found.append(_Token(start, end, value))
            taken.append((start, end))

    for m in _FOLD.finditer(clause):
        add(m.start(), m.end(), NUMBER_WORDS[m.group(1).lower()])
    for m in _BIG_WORD.finditer(clause):
        add(m.start(), m.end(), NUMBER_WORDS[m.group(1).lower()])
    for m in _WORD.finditer(clause):
        add(m.start(), m.end(), NUMBER_WORDS[m.group(1).lower()])
    for m in _DIGITS.finditer(clause):
        before = clause[m.start() - 1] if m.start() > 0 else " "
        after = clause[m.end()] if m.end() < len(clause) else " "
        if (
            before.isalnum()
            or before == "_"
            or (after.isalpha() and after not in ("x", TIMES_SIGN))
        ):
            continue
        add(m.start(), m.end(), float(m.group()))
    return sorted(found, key=lambda t: t.start)


_MULTIPLIER_AFTER = re.compile(r"\s*(?:times\b|x\b|" + TIMES_SIGN + ")", re.IGNORECASE)
_UNIT_AFTER = re.compile(r"\s+(?:ph\s+)?(?:units?|steps?|levels?|points?)\b", re.IGNORECASE)
_THRESHOLD_BEFORE = re.compile(
    r"\b(below|less than|under|lower than|above|more than|greater than|over|higher than)"
    r"\s+(?:a\s+)?(?:ph\s+)?(?:of\s+)?$",
    re.IGNORECASE,
)


def _is_multiplier(clause: str, t: _Token) -> bool:
    return clause[t.start : t.end].lower().endswith("fold") or bool(
        _MULTIPLIER_AFTER.match(clause, t.end)
    )


def _is_unit_count(clause: str, t: _Token) -> bool:
    return bool(_UNIT_AFTER.match(clause, t.end))


def _threshold(clause: str, t: _Token) -> str | None:
    m = _THRESHOLD_BEFORE.search(clause[: t.start])
    if m is None:
        return None
    return (
        "below" if m.group(1).lower() in ("below", "less than", "under", "lower than") else "above"
    )


def _fmt(value: float) -> str:
    return f"{value:g}"


# --- pH checkers --------------------------------------------------------------------------------

_CLASS_WORDS = {
    "acidic": "acidic", "acid": "acidic", "acids": "acidic", "neutral": "neutral",
    "basic": "basic", "alkaline": "basic", "base": "basic", "bases": "basic",
}  # fmt: skip
_CLASS_RE = re.compile(r"\b(" + "|".join(_CLASS_WORDS) + r")\b", re.IGNORECASE)


def ph_classification(clause: str) -> list[ClaimFinding]:
    labels = {_CLASS_WORDS[m.group(1).lower()] for m in _CLASS_RE.finditer(clause)}
    items = [
        (t, _threshold(clause, t))
        for t in _tokens(clause)
        if not _is_multiplier(clause, t) and not _is_unit_count(clause, t) and 0 <= t.value <= 14
    ]
    if not items or len(labels) != 1:
        return []
    (label,) = labels

    def consistent(t: _Token, threshold: str | None) -> bool:
        if threshold == "below":
            return label == "acidic" and t.value <= 7
        if threshold == "above":
            return label == "basic" and t.value >= 7
        return ph_class(t.value) == label

    spans = tuple(t.span for t, _ in items)
    if all(consistent(t, th) for t, th in items):
        return [ClaimFinding("ph_classification", clause, True, "classification matches", spans)]
    if len(items) == 1:
        t, th = items[0]
        where = f"{th} {_fmt(t.value)}" if th else f"pH {_fmt(t.value)}"
        actual = "acidic (below 7) or basic (above 7)" if th else ph_class(t.value)
        return [
            ClaimFinding(
                "ph_classification",
                clause,
                False,
                f"a solution with {where} is {actual}, not {label}.",
                spans,
            )
        ]
    return []


_QUANTIFIER_UNIT = re.compile(
    r"\b(?:each|every|per|a single)\s+(?:ph\s+)?(?:unit|step|level|point)\b", re.IGNORECASE
)
_DOUBLING = re.compile(r"\b(?:twice|double|doubles)\b", re.IGNORECASE)


def ph_ratio(clause: str) -> list[ClaimFinding]:
    tokens = _tokens(clause)
    multipliers = [t for t in tokens if _is_multiplier(clause, t)]
    doubling = bool(_DOUBLING.search(clause))
    if not multipliers and not doubling:
        return []
    unit_tokens = [t for t in tokens if _is_unit_count(clause, t)]
    unit_values = {t.value for t in unit_tokens}
    if _QUANTIFIER_UNIT.search(clause):
        unit_values.add(1)
    spans = tuple(t.span for t in multipliers + unit_tokens)
    if len(unit_values) > 1:
        return []
    if unit_values:
        (units,) = unit_values
    else:
        ph_values = [
            t
            for t in tokens
            if t not in multipliers and t not in unit_tokens and 0 <= t.value <= 14
        ]
        if len(ph_values) != 2 or ph_values[0].value == ph_values[1].value:
            return []
        units = abs(ph_values[0].value - ph_values[1].value)
        spans += tuple(t.span for t in ph_values)
    expected = 10**units
    stated = [t.value for t in multipliers] + ([2.0] if doubling else [])
    for value in stated:
        if abs(value - expected) > 0.01 * expected:
            return [
                ClaimFinding(
                    "ph_ratio",
                    clause,
                    False,
                    f"{_fmt(units)} pH unit(s) is a {_fmt(expected)}x change in hydrogen ions, "
                    f"not {_fmt(value)}x.",
                    spans,
                )
            ]
    return [ClaimFinding("ph_ratio", clause, True, "ratio matches", spans)]


_ALIASES = sorted(
    ((alias, sub) for sub in SUBSTANCE_PH.values() for alias in sub.aliases),
    key=lambda pair: -len(pair[0]),
)


def _alias_mentions(clause: str) -> list[tuple[int, int, SubstancePH]]:
    lowered = clause.lower()
    mentions: list[tuple[int, int, SubstancePH]] = []
    for alias, sub in _ALIASES:
        for m in re.finditer(rf"\b{re.escape(alias)}\b", lowered):
            if all(m.end() <= s or m.start() >= e for s, e, _ in mentions):
                mentions.append((m.start(), m.end(), sub))
    return sorted(mentions, key=lambda x: x[0])


def substance_ph(clause: str) -> list[ClaimFinding]:
    tokens = [
        t
        for t in _tokens(clause)
        if 0 <= t.value <= 14
        and not _is_multiplier(clause, t)
        and not _is_unit_count(clause, t)
        and _threshold(clause, t) is None
    ]
    mentions = _alias_mentions(clause)
    findings: list[ClaimFinding] = []
    for i, (_, end, sub) in enumerate(mentions):
        limit = mentions[i + 1][0] if i + 1 < len(mentions) else len(clause)
        token = next((t for t in tokens if end <= t.start < limit), None)
        if token is None:
            continue
        ok = abs(token.value - sub.ph) <= sub.tolerance
        message = (
            "substance pH matches"
            if ok
            else f"{sub.display} has a pH of about {_fmt(sub.ph)}, not {_fmt(token.value)}."
        )
        findings.append(ClaimFinding("substance_ph", clause, ok, message, (token.span,)))
    return findings


_SCALE_CONTEXT = re.compile(r"\b(?:scale|ph)\b", re.IGNORECASE)
_RANGE_BETWEEN = re.compile(r"\s+(?:to|and)\s+", re.IGNORECASE)


def ph_range(clause: str) -> list[ClaimFinding]:
    if not _SCALE_CONTEXT.search(clause):
        return []
    tokens = _tokens(clause)
    for a, b in itertools.pairwise(tokens):
        if _RANGE_BETWEEN.fullmatch(clause[a.end : b.start]):
            ok = (a.value, b.value) == (0, 14)
            message = (
                "range matches"
                if ok
                else f"the pH scale runs from 0 to 14, not {_fmt(a.value)} to {_fmt(b.value)}."
            )
            return [ClaimFinding("ph_range", clause, ok, message, (a.span, b.span))]
    return []


# --- bonding checkers ---------------------------------------------------------------------------

_ELEMENT_BY_NAME = {e.name: e for e in ELEMENTS.values()}
_ELEMENT_NAME_RE = re.compile(r"\b(" + "|".join(_ELEMENT_BY_NAME) + r")\b", re.IGNORECASE)
_VALENCE_AFTER = re.compile(
    r"\s+(?:(?:outer|valence|outermost)(?:\s+shell)?\s+electrons?\b"
    r"|electrons?\s+in\s+(?:its|the|their)\s+outer(?:most)?\s+shell\b)",
    re.IGNORECASE,
)


def valence_claims(clause: str) -> list[ClaimFinding]:
    findings: list[ClaimFinding] = []
    names = list(_ELEMENT_NAME_RE.finditer(clause))
    for t in _tokens(clause):
        if not _VALENCE_AFTER.match(clause, t.end):
            continue
        before = [m for m in names if m.end() <= t.start]
        if not before:
            continue
        element = _ELEMENT_BY_NAME[before[-1].group(1).lower()]
        ok = t.value == element.valence
        message = (
            "valence matches"
            if ok
            else f"{element.name.capitalize()} has {element.valence} outer electron(s), "
            f"not {_fmt(t.value)}."
        )
        findings.append(ClaimFinding("valence_claims", clause, ok, message, (t.span,)))
    return findings


_MOLECULE_NAMES: list[tuple[str, Molecule]] = sorted(
    [(m.value.lower(), m) for m in MOLECULES] + [(d.name, d.formula) for d in MOLECULES.values()],
    key=lambda pair: -len(pair[0]),
)
_PAIRS_AFTER = re.compile(r"\s+(?:shared\s+)?pairs?\b", re.IGNORECASE)


def shared_pairs_claims(clause: str) -> list[ClaimFinding]:
    text = unicodedata.normalize("NFKC", clause)
    lowered = text.lower()
    taken: list[Span] = []
    molecules: set[Molecule] = set()
    for name, molecule in _MOLECULE_NAMES:
        for m in re.finditer(rf"\b{re.escape(name)}\b", lowered):
            if all(m.end() <= s or m.start() >= e for s, e in taken):
                taken.append((m.start(), m.end()))
                molecules.add(molecule)
    pair_tokens = [t for t in _tokens(text) if _PAIRS_AFTER.match(text, t.end)]
    if len(molecules) != 1 or not pair_tokens:
        return []
    (molecule,) = molecules
    data = MOLECULES[molecule]
    findings = []
    for t in pair_tokens:
        ok = t.value == data.shared_pairs
        message = (
            "shared pairs match"
            if ok
            else f"{data.name.capitalize()} ({data.display}) shares {data.shared_pairs} pair(s) "
            f"of electrons, not {_fmt(t.value)}."
        )
        findings.append(ClaimFinding("shared_pairs_claims", clause, ok, message, (t.span,)))
    return findings


_EXPECTED_IONS = {"H": "H+"}
for _data in IONIC_COMPOUNDS.values():
    _EXPECTED_IONS[ELEMENTS[_data.donor].symbol] = _data.donor_ion
    _EXPECTED_IONS[ELEMENTS[_data.acceptor].symbol] = _data.acceptor_ion
_ION_RE = re.compile(r"(?<![A-Za-z])(Na|Mg|Cl|F|O|H)(\d*)([+\-])(?![\w+\-])")


def ion_charge_claims(clause: str) -> list[ClaimFinding]:
    text = unicodedata.normalize("NFKC", clause).replace(MINUS_SIGN, "-")
    findings = []
    for m in _ION_RE.finditer(text):
        symbol, written = m.group(1), m.group(0)
        expected = _EXPECTED_IONS[symbol]
        ok = written == expected
        message = "ion charge matches" if ok else f"{symbol} forms {expected} ions, not {written}."
        findings.append(ClaimFinding("ion_charge_claims", clause, ok, message))
    return findings


CHECKERS: Mapping[str, Callable[[str], list[ClaimFinding]]] = {
    "ph_classification": ph_classification,
    "ph_ratio": ph_ratio,
    "substance_ph": substance_ph,
    "ph_range": ph_range,
    "valence_claims": valence_claims,
    "shared_pairs_claims": shared_pairs_claims,
    "ion_charge_claims": ion_charge_claims,
}


def check_texts(texts: Sequence[str], checker_names: Sequence[str]) -> ClaimReport:
    findings: list[ClaimFinding] = []
    unverified: list[str] = []
    for text in texts:
        for clause in split_clauses(text):
            clause_findings = [f for name in checker_names for f in CHECKERS[name](clause)]
            findings.extend(clause_findings)
            used = {span for f in clause_findings for span in f.spans}
            numbers = _tokens(unicodedata.normalize("NFKC", clause))
            if any(t.span not in used for t in numbers) and clause not in unverified:
                unverified.append(clause)
    return ClaimReport(findings=findings, unverified=unverified)
