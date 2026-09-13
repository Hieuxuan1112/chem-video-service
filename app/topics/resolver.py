"""Deterministic topic resolution (spec §5.2)."""

import re
import unicodedata
from typing import Protocol

_NON_ALNUM = re.compile(r"[\W_]+")


class TopicResolver(Protocol):
    def resolve(self, query: str) -> str | None: ...

    def topics(self) -> list[tuple[str, str, str]]:
        """(topic_id, title, canonical_query) for every supported topic."""
        ...


def normalize(text: str) -> str:
    """NFKC, lowercase, non-alphanumeric runs to one space, trimmed."""
    folded = unicodedata.normalize("NFKC", text).lower()
    return _NON_ALNUM.sub(" ", folded).strip()
