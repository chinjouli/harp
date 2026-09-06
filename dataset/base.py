from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator


class EpisodeDataset(ABC):
    """Yields EpisodeRecord dicts for extraction and evaluation.

    EpisodeRecord schema:
        episode_id:     str
        audio_path:     str  (absolute path to .flac/.wav)
        gt_segments:    List[Dict]  ([] if unavailable)
        ct_annotations: Dict        (omitted if not a Conversation episode)
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]: ...

    @abstractmethod
    def get(self, episode_id: str) -> Dict[str, Any]: ...
