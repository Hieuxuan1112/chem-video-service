import pytest

from app.topics.resolver import normalize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("How does the pH-scale work?!", "how does the ph scale work"),
        ("  Ionic\tvs.\n covalent  ", "ionic vs covalent"),
        (chr(0xFF30) + chr(0xFF28) + " scale", "ph scale"),  # full-width "PH" folds under NFKC
        ("H₂O & NaCl_bonds", "h2o nacl bonds"),
        ("", ""),
        ("?!...", ""),
    ],
)
def test_normalize(text: str, expected: str) -> None:
    assert normalize(text) == expected
