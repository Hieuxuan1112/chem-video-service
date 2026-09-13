import pytest

from app.pipeline.gates.g2_claims import g2_claims
from app.pipeline.storyboard import Storyboard
from app.topics.claims import CHECKERS, NUMBER_WORDS, check_texts, split_clauses
from app.topics.registry import TOPICS_ROOT, TopicRegistry

REGISTRY = TopicRegistry.load(TOPICS_ROOT)
PH = ["ph_classification", "ph_ratio", "substance_ph", "ph_range"]
BONDS = ["valence_claims", "shared_pairs_claims", "ion_charge_claims"]

TABLE = [
    (PH, "A pH of 7 is neutral.", "ok"),
    (PH, "pH 7 is acidic.", "violation"),
    (PH, "Anything below 7 is acidic.", "ok"),
    (PH, "A solution with pH 3 is basic.", "violation"),
    (PH, "Each step on the scale is ten times stronger.", "ok"),
    (PH, "Two pH units means 100 times more hydrogen ions.", "ok"),
    (PH, "Two pH units means 20 times more.", "violation"),
    (PH, "Each unit is twice as acidic.", "violation"),
    (PH, "Lemon juice has a pH of about 2.", "ok"),
    (PH, "Lemon juice has a pH of 9.", "violation"),
    (PH, "The pH scale runs from 0 to 14.", "ok"),
    (PH, "The scale runs from 1 to 10.", "violation"),
    (PH, "pH 2 is acidic while pH 12 is basic.", "ok"),
    (PH, "pH 2 and pH 12 are acidic and basic.", "unverified"),
    (BONDS, "Sodium has one outer electron.", "ok"),
    (BONDS, "Chlorine has two outer electrons.", "violation"),
    (BONDS, "Nitrogen molecules share three pairs of electrons.", "ok"),
    (BONDS, "Oxygen atoms share one pair.", "violation"),
    (BONDS, "Magnesium forms Mg2+ ions.", "ok"),
    (BONDS, "Sodium forms Na2+ ions.", "violation"),
    (PH, "Water covers 71 percent of Earth.", "unverified"),
]


@pytest.mark.parametrize(("checkers", "text", "expected"), TABLE, ids=[t[1] for t in TABLE])
def test_claim_table(checkers: list[str], text: str, expected: str) -> None:
    report = check_texts([text], checkers)
    if expected == "ok":
        assert report.violations == [], report
        assert report.findings and all(f.ok for f in report.findings)
        assert report.unverified == []
    elif expected == "violation":
        assert report.violations, report
        assert all(v.message for v in report.violations)
    else:
        assert report.violations == [], report
        assert report.unverified, report


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("Seawater is about 8, and household ammonia is about 11.", "ok"),
        ("Seawater is about 8, and household ammonia is about 3.", "violation"),
        ("Milk of magnesia is about 10.", "ok"),  # longest alias wins over "milk"
        ("Black coffee is about 5.", "ok"),
        ("A pH below 7 is acidic, like lemon juice.", "ok"),  # 7 is not lemon juice's value
        ("Lemon juice has 100 times more hydrogen ions than water.", "unverified"),
    ],
)
def test_substance_pairing(text: str, value: str) -> None:
    report = check_texts([text], PH)
    assert bool(report.violations) is (value == "violation"), report
    if value == "unverified":
        assert report.unverified


def test_ratio_from_ph_pair_and_threshold_rules() -> None:
    ok = "So a solution with pH 3 has ten times more hydrogen ions than a solution with pH 4."
    assert check_texts([ok], PH).violations == []
    assert check_texts([ok], PH).unverified == []
    bad = "A solution with pH 3 has ten times more hydrogen ions than a solution with pH 5."
    assert check_texts([bad], PH).violations
    assert check_texts(["Anything above 7 is acidic."], PH).violations
    assert (
        check_texts(["Below 7 is acidic; exactly 7 is neutral; above 7 is basic."], PH).violations
        == []
    )


def test_partially_consumed_clause_is_unverified_not_violation() -> None:
    report = check_texts(
        ["Chlorine has seven outer electrons, so it needs one more to reach eight."], BONDS
    )
    assert report.violations == []
    assert report.unverified == [
        "Chlorine has seven outer electrons, so it needs one more to reach eight"
    ]


def test_numbers_inside_formulas_are_not_claims() -> None:
    report = check_texts(["Hydrogen gas is H2 and oxygen gas is O₂."], BONDS)
    assert report.unverified == [] and report.findings == []


def test_split_clauses() -> None:
    assert split_clauses("pH 2 is acidic while pH 12 is basic. Blood is 7.4; soap is basic!") == [
        "pH 2 is acidic",
        "pH 12 is basic",
        "Blood is 7.4",
        "soap is basic",
    ]


def test_number_words() -> None:
    assert NUMBER_WORDS["zero"] == 0 and NUMBER_WORDS["twenty"] == 20
    assert NUMBER_WORDS["hundred"] == 100 and NUMBER_WORDS["thousand"] == 1000


def test_pack_claim_checks_exist() -> None:
    for pack in REGISTRY.all():
        assert set(pack.claim_checks) <= set(CHECKERS), pack.id


def test_g2_claims_checks_on_screen_text_and_locates_scene() -> None:
    raw = REGISTRY.curated_storyboard("ph_scale").model_dump(mode="json")
    raw["scenes"][5]["visual"]["points"][0] = "pH 7 is acidic"
    result, unverified = g2_claims(Storyboard.model_validate(raw), REGISTRY.get("ph_scale"))
    assert not result.passed
    (violation,) = result.violations
    assert (violation.gate, violation.code, violation.scene_index) == (
        "G2c",
        "claim_ph_classification",
        5,
    )
    assert "Scene 5" in violation.message and "pH 7 is acidic" in violation.message
    assert result.details["unverified_numeric_claims"] == unverified


@pytest.mark.parametrize("topic_id", ["ph_scale", "covalent_bonds", "ionic_vs_covalent"])
def test_curated_storyboards_have_no_claim_violations(topic_id: str) -> None:
    result, _ = g2_claims(REGISTRY.curated_storyboard(topic_id), REGISTRY.get(topic_id))
    assert result.passed, result.violations
