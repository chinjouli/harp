from __future__ import annotations

from typing import Any, Dict, List

from dataset.process_gt_music import _aggregate  # noqa: F401  re-exported


def build_file_index(gt_segs: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map mp3 file_name → SONG_XX speaker tag from pre-processed GT segments."""
    return {
        s["file_name"]: s["speaker"]
        for s in gt_segs
        if s.get("file_name")
    }
