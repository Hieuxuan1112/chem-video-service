"""Loads and validates every topic pack under a root directory."""

from pathlib import Path

import yaml

from app.pipeline.storyboard import Storyboard
from app.topics.pack import TopicPack

TOPICS_ROOT = Path(__file__).parent


class TopicRegistry:
    def __init__(self, packs: dict[str, TopicPack], dirs: dict[str, Path]) -> None:
        self._packs = packs
        self._dirs = dirs

    @classmethod
    def load(cls, root: Path) -> "TopicRegistry":
        packs: dict[str, TopicPack] = {}
        dirs: dict[str, Path] = {}
        for pack_file in sorted(root.glob("*/pack.yaml")):
            pack = TopicPack.model_validate(yaml.safe_load(pack_file.read_text(encoding="utf-8")))
            if pack.id != pack_file.parent.name:
                raise ValueError(f"pack id {pack.id!r} does not match directory {pack_file.parent}")
            packs[pack.id] = pack
            dirs[pack.id] = pack_file.parent
        return cls(packs, dirs)

    def get(self, topic_id: str) -> TopicPack:
        return self._packs[topic_id]

    def all(self) -> list[TopicPack]:
        return [self._packs[k] for k in sorted(self._packs)]

    def pack_dir(self, topic_id: str) -> Path:
        return self._dirs[topic_id]

    def curated_storyboard(self, topic_id: str) -> Storyboard:
        path = self.pack_dir(topic_id) / "curated_storyboard.json"
        return Storyboard.model_validate_json(path.read_text(encoding="utf-8"))
