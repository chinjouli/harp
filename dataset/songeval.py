from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

from dataset.base import EpisodeDataset


class SongEvalDataset(EpisodeDataset):
    """Long-music tracks from the SongEval dataset.

    Each track (track_00.mp3, …) is one episode. GT segments come from
    tracks_metadata.jsonl, grouped by track_name.

    Args:
        data_dir:       Directory containing track_XX.mp3 files.
        metadata_path:  Path to tracks_metadata.jsonl.
        tracks:         Optional list of track names to include
                        (e.g. ["track_00", "track_01"]); None = all.
    """

    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        tracks: List[str] | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._episodes = self._load(Path(metadata_path), tracks)

    def _load(
        self, meta_path: Path, tracks: List[str] | None
    ) -> List[Dict[str, Any]]:
        by_track: Dict[str, List[Dict]] = {}
        with meta_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                tid = entry["track_name"]
                by_track.setdefault(tid, []).append(entry)

        selected = sorted(by_track.keys()) if tracks is None else tracks
        episodes = []
        for tid in selected:
            segs = sorted(by_track.get(tid, []), key=lambda x: x["start_time"])
            audio_path = self._data_dir / f"{tid}.mp3"
            episodes.append({
                "episode_id": tid,
                "audio_path": str(audio_path),
                "gt_segments": segs,
            })
        return episodes

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._episodes)

    def get(self, episode_id: str) -> Dict[str, Any]:
        for ep in self._episodes:
            if ep["episode_id"] == episode_id:
                return ep
        raise KeyError(episode_id)
