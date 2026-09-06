from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from dataset.base import EpisodeDataset


class LongFormDataset(EpisodeDataset):
    """MedMosaic Long-Form dataset: 106 medical-conversation WAV files.

    Each item in Long_Form_metadata.json maps 1-to-1 with one audio file.
    Episode ID is the item's ``id`` field; audio resolved as:
        data_dir / Path(audio_path).name
    """

    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        ids: Optional[List[str]] = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._episodes = self._load(Path(metadata_path), ids)

    def _load(
        self, meta_path: Path, ids: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = json.loads(
            meta_path.read_text(encoding="utf-8")
        )
        episodes = []
        for item in items:
            if ids is not None and item["id"] not in ids:
                continue
            audio_path = self._data_dir / Path(item["audio_path"]).name
            episodes.append({
                "episode_id": Path(item["audio_path"]).stem,
                "audio_path": str(audio_path),
                "gt_segments": [],
                "question": item.get("question"),
                "options": item.get("options", []),
                "ground_truth": item.get("ground_truth"),
                "difficulty_level": item.get("difficulty_level"),
            })
        return episodes

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._episodes)

    def get(self, episode_id: str) -> Dict[str, Any]:
        for ep in self._episodes:
            if ep["episode_id"] == episode_id:
                return ep
        raise KeyError(episode_id)
