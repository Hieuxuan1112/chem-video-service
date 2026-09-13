import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.pipeline.storyboard import VISUAL_TYPES
from app.topics.pack import TopicPack
from app.topics.registry import TOPICS_ROOT, TopicRegistry
from app.topics.resolver import KeywordResolver, normalize

CANONICAL = {
    "ph_scale": "How does the pH scale work?",
    "covalent_bonds": "Why do atoms form covalent bonds?",
    "ionic_vs_covalent": "What is the difference between ionic and covalent bonding?",
}


@pytest.fixture(scope="module")
def registry() -> TopicRegistry:
    return TopicRegistry.load(TOPICS_ROOT)


@pytest.fixture(scope="module")
def resolver(registry: TopicRegistry) -> KeywordResolver:
    return KeywordResolver(registry)


def test_three_packs_load(registry: TopicRegistry) -> None:
    packs = registry.all()
    assert [p.id for p in packs] == sorted(CANONICAL)
    for pack in packs:
        assert pack.canonical_query == CANONICAL[pack.id]
        assert pack.subject == "chemistry"
        assert pack.required_facts and pack.misconceptions and pack.claim_checks
        assert registry.get(pack.id) is pack
        assert registry.pack_dir(pack.id) == TOPICS_ROOT / pack.id
    with pytest.raises(KeyError):
        registry.get("photosynthesis")


def test_canonical_queries_resolve_to_own_pack(resolver: KeywordResolver) -> None:
    for topic_id, query in CANONICAL.items():
        assert resolver.resolve(query) == topic_id


def test_no_score_tie_on_canonical_queries(registry: TopicRegistry) -> None:
    for query in CANONICAL.values():
        scores = sorted(
            (
                s
                for p in registry.all()
                if (s := KeywordResolver.score(p, normalize(query))) is not None
            ),
            reverse=True,
        )
        assert len(scores) == 1 or scores[0] > scores[1], (query, scores)


@pytest.mark.parametrize(
    ("query", "topic_id"),
    [
        ("explain pH", "ph_scale"),
        ("what makes something acidic or alkaline on the ph scale", "ph_scale"),
        ("Why is lemon juice acidic?", "ph_scale"),
        ("why do atoms share electrons in covalent bonds", "covalent_bonds"),
        ("ionic vs covalent bonds", "ionic_vs_covalent"),
        ("compare covalent and ionic bonding", "ionic_vs_covalent"),
    ],
)
def test_paraphrases(resolver: KeywordResolver, query: str, topic_id: str) -> None:
    assert resolver.resolve(query) == topic_id


@pytest.mark.parametrize(
    "query", ["what is photosynthesis", "tell me about gravity", "how do ionic bonds form"]
)
def test_unsupported_queries_return_none(resolver: KeywordResolver, query: str) -> None:
    assert resolver.resolve(query) is None


def test_covalent_with_ionic_goes_to_comparison(resolver: KeywordResolver) -> None:
    assert resolver.resolve("Are covalent bonds stronger than ionic bonds?") == "ionic_vs_covalent"


def test_topics_listing(resolver: KeywordResolver) -> None:
    assert resolver.topics() == [
        ("covalent_bonds", "Covalent bonds", CANONICAL["covalent_bonds"]),
        ("ionic_vs_covalent", "Ionic vs covalent bonding", CANONICAL["ionic_vs_covalent"]),
        ("ph_scale", "The pH scale", CANONICAL["ph_scale"]),
    ]


def test_pack_visuals_subset_of_catalogue(registry: TopicRegistry) -> None:
    for pack in registry.all():
        assert set(pack.allowed_visuals) <= set(VISUAL_TYPES)
        assert set(pack.required_visuals) <= set(pack.allowed_visuals)
        assert {"title_card", "key_points"} <= set(pack.allowed_visuals)


def test_misconception_regexes_compile(registry: TopicRegistry) -> None:
    for pack in registry.all():
        for m in pack.misconceptions:
            re.compile(m.pattern, re.IGNORECASE)


@pytest.mark.parametrize(
    ("topic_id", "text", "misconception"),
    [
        ("ph_scale", "A pH of 7 is acidic.", True),
        ("ph_scale", "A higher pH means more acidic.", True),
        ("ph_scale", "Each unit is twice as strong.", True),
        ("ph_scale", "The pH scale is linear.", True),
        ("ph_scale", "The pH scale is not linear; each unit is ten times.", False),
        ("ph_scale", "A pH of 7 is neutral and a lower pH means more acidic.", False),
        ("covalent_bonds", "Covalent bonds involve the transfer of electrons.", True),
        ("covalent_bonds", "Atoms lose electrons to form covalent bonds.", True),
        ("covalent_bonds", "Covalent bonds involve sharing pairs of electrons.", False),
        ("ionic_vs_covalent", "Ionic bonds involve sharing electrons.", True),
        ("ionic_vs_covalent", "Ionic bonds form between two non-metals.", True),
        ("ionic_vs_covalent", "Ionic bonds form between a metal and a non-metal.", False),
    ],
)
def test_misconception_patterns_match_normalized_text(
    registry: TopicRegistry, topic_id: str, text: str, misconception: bool
) -> None:
    pack = registry.get(topic_id)
    hit = any(re.search(m.pattern, normalize(text), re.IGNORECASE) for m in pack.misconceptions)
    assert hit is misconception


def test_pack_rejects_unknown_visual(registry: TopicRegistry) -> None:
    raw = registry.get("ph_scale").model_dump()
    raw["allowed_visuals"] = [*raw["allowed_visuals"], "hologram"]
    with pytest.raises(ValidationError):
        TopicPack.model_validate(raw)
    raw = registry.get("ph_scale").model_dump()
    raw["required_visuals"] = ["electron_transfer"]
    with pytest.raises(ValidationError):
        TopicPack.model_validate(raw)


def test_pack_id_must_match_directory(tmp_path: Path, registry: TopicRegistry) -> None:
    (tmp_path / "wrong_dir").mkdir()
    source = (TOPICS_ROOT / "ph_scale" / "pack.yaml").read_text(encoding="utf-8")
    (tmp_path / "wrong_dir" / "pack.yaml").write_text(source, encoding="utf-8")
    with pytest.raises(ValueError):
        TopicRegistry.load(tmp_path)
