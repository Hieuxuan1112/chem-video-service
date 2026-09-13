"""Topic pack models (spec §5.1). A pack is data; adding a STEM topic means adding a pack."""

import re

from pydantic import BaseModel, ConfigDict, model_validator

from app.pipeline.storyboard import VISUAL_TYPES
from app.topics.chemistry_data import ElementSymbol, IonicCompound, Molecule


class _PackModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MatchRules(_PackModel):
    all_of: list[list[str]]  # every group needs one whole-word phrase in the normalized query
    none_of: list[str] = []


class RequiredFact(_PackModel):
    id: str
    statement: str
    detect: list[list[str]]  # every group needs one phrase in the normalized full narration


class Misconception(_PackModel):
    id: str
    pattern: str  # regex, case-insensitive, over normalized text
    explanation: str


class TopicPack(_PackModel):
    id: str
    subject: str
    title: str
    version: str
    canonical_query: str
    match: MatchRules
    required_facts: list[RequiredFact]
    misconceptions: list[Misconception]
    claim_checks: list[str]
    allowed_visuals: list[str]
    required_visuals: list[str]
    allowed_elements: list[ElementSymbol] = []
    allowed_molecules: list[Molecule] = []
    allowed_compounds: list[IonicCompound] = []
    pronunciations: dict[str, str] = {}

    @model_validator(mode="after")
    def _check_consistency(self) -> "TopicPack":
        unknown = set(self.allowed_visuals) - set(VISUAL_TYPES)
        if unknown:
            raise ValueError(f"unknown visuals in allowed_visuals: {sorted(unknown)}")
        missing = set(self.required_visuals) - set(self.allowed_visuals)
        if missing:
            raise ValueError(f"required_visuals not in allowed_visuals: {sorted(missing)}")
        for m in self.misconceptions:
            re.compile(m.pattern, re.IGNORECASE)
        return self
