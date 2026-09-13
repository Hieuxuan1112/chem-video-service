"""Deterministic topic resolution (spec §5.2)."""

import re
import unicodedata
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.topics.pack import TopicPack
    from app.topics.registry import TopicRegistry

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


def contains_phrase(normalized_text: str, phrase: str) -> bool:
    """Whole-word phrase match; both sides are normalized."""
    return f" {normalize(phrase)} " in f" {normalized_text} "


class KeywordResolver:
    """Implements TopicResolver with each pack's match rules (spec §5.2)."""

    def __init__(self, registry: "TopicRegistry") -> None:
        self._registry = registry

    @staticmethod
    def score(pack: "TopicPack", normalized_query: str) -> int | None:
        """len(all_of) if every group matches and no none_of phrase is present, else None."""
        if any(contains_phrase(normalized_query, p) for p in pack.match.none_of):
            return None
        for group in pack.match.all_of:
            if not any(contains_phrase(normalized_query, p) for p in group):
                return None
        return len(pack.match.all_of)

    def resolve(self, query: str) -> str | None:
        normalized = normalize(query)
        scored = [
            (score, pack.id)
            for pack in self._registry.all()
            if (score := self.score(pack, normalized)) is not None
        ]
        if not scored:
            return None
        best = max(score for score, _ in scored)
        return min(topic_id for score, topic_id in scored if score == best)

    def topics(self) -> list[tuple[str, str, str]]:
        return [(p.id, p.title, p.canonical_query) for p in self._registry.all()]
