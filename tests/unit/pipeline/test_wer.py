import pytest

from app.pipeline.wer import normalize_for_wer, word_error_rate


def test_identical_zero() -> None:
    text = "The pH scale runs from 0 to 14."
    assert word_error_rate(text, text) == 0.0
    assert word_error_rate(text, "the ph scale runs from 0 to 14") == 0.0  # case, punctuation


def test_spelled_letters_join() -> None:
    assert normalize_for_wer("P H") == ["ph"]
    assert word_error_rate("What does pH measure?", "what does p h measure") == 0.0
    # "a" and "I" are words, not spelled letters
    assert normalize_for_wer("a pH of 7") == ["a", "ph", "of", "7"]
    assert normalize_for_wer("I think H C l") == ["i", "think", "hcl"]


def test_number_words() -> None:
    assert normalize_for_wer("zero to fourteen") == ["0", "to", "14"]
    assert normalize_for_wer("Twenty-one and ninety nine, then ten times") == [
        "21", "and", "99", "then", "10", "times",
    ]  # fmt: skip
    assert word_error_rate("from zero to fourteen", "From 0 to 14.") == 0.0


def test_normalization_details() -> None:
    assert normalize_for_wer("Let's  see:\tH+ ions!") == ["lets", "see", "h", "ions"]
    assert normalize_for_wer("\uff30\uff28") == ["ph"]  # NFKC folds full-width letters


def test_substitution_deletion_insertion_counts() -> None:
    ref = "acids have a low ph"  # 5 words
    assert word_error_rate(ref, "acids have a high ph") == pytest.approx(1 / 5)  # substitution
    assert word_error_rate(ref, "acids have low ph") == pytest.approx(1 / 5)  # deletion
    assert word_error_rate(ref, "acids really have a low ph") == pytest.approx(1 / 5)  # insertion
    assert word_error_rate(ref, "bases have high values ph now") == pytest.approx(4 / 5)
    assert word_error_rate("one two", "red green blue cyan pink gold") == pytest.approx(6 / 2)


def test_empty_cases() -> None:
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("...", "") == 0.0
    assert word_error_rate("", "hello") == 1.0
    assert word_error_rate("hello world", "") == 1.0
