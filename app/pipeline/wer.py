"""Word error rate for A5 reverse-ASR checks (spec §7.3), with chemistry-aware normalization."""

import re
import unicodedata

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}  # fmt: skip
_SINGLE_LETTER_WORDS = {"a", "i"}


def _numbers_to_digits(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _TENS:
            value = _TENS[token]
            if i + 1 < len(tokens) and 1 <= _UNITS.get(tokens[i + 1], 0) <= 9:
                value += _UNITS[tokens[i + 1]]
                i += 1
            out.append(str(value))
        elif token in _UNITS:
            out.append(str(_UNITS[token]))
        else:
            out.append(token)
        i += 1
    return out


def _is_spelled_letter(token: str) -> bool:
    return len(token) == 1 and token.isalpha() and token not in _SINGLE_LETTER_WORDS


def _join_spelled_letters(tokens: list[str]) -> list[str]:
    """Runs of two or more single letters are one spelled word: "p h" -> "ph"."""
    out: list[str] = []
    run: list[str] = []
    for token in [*tokens, ""]:
        if _is_spelled_letter(token):
            run.append(token)
            continue
        if len(run) >= 2:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []
        if token:
            out.append(token)
    return out


def normalize_for_wer(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower().replace("'", "").replace("\u2019", "")
    tokens = re.sub(r"[^\w]+|_", " ", text).split()
    return _join_spelled_letters(_numbers_to_digits(tokens))


def _levenshtein(ref: list[str], hyp: list[str]) -> int:
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)
