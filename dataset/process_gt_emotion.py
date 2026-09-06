from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dataset.base import EpisodeDataset


def process_gt_episode(
    episode: Dict[str, Any],
    out_dir: Path,
    extracted_dir: Optional[Path] = None,
) -> None:
    """Write GT segments to gt/segments/{episode_id}.jsonl.

    Produces the same segment format as extract/pipeline.py so
    the retriever works on both paths without modification.

    If extracted_dir is provided, borrows speaker/domain FAISS indices
    from extracted segments via time-overlap matching.
    """
    episode_id: str = episode["episode_id"]
    gt_dir = out_dir / "gt" / "segments"
    gt_dir.mkdir(parents=True, exist_ok=True)
    out_path = gt_dir / f"{episode_id}.jsonl"
    if out_path.exists():
        return

    gt_segments: List[Dict[str, Any]] = episode.get("gt_segments", [])
    if not gt_segments:
        out_path.touch()
        return

    extracted: List[Dict[str, Any]] = []
    if extracted_dir is not None:
        ext_path = extracted_dir / "segments" / f"{episode_id}.jsonl"
        if ext_path.exists():
            extracted = _load_jsonl(ext_path)

    with out_path.open("w", encoding="utf-8") as f:
        for seg in gt_segments:
            if extracted:
                spk_idx, dom_idxs = borrow_faiss_idx(seg, extracted)
                seg = {
                    **seg,
                    "audio_faiss_idx": spk_idx,
                    "domain_faiss_idxs": dom_idxs,
                }
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")


def borrow_faiss_idx(
    gt_seg: Dict[str, Any],
    extracted_segments: List[Dict[str, Any]],
) -> Tuple[Optional[int], List[int]]:
    """Find extracted segment with max time overlap → return its indices."""
    g_start, g_end = float(gt_seg["start"]), float(gt_seg["end"])
    best_overlap = 0.0
    best: Optional[Dict[str, Any]] = None

    for ext in extracted_segments:
        overlap = max(
            0.0,
            min(g_end, float(ext["end"])) - max(g_start, float(ext["start"])),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best = ext

    if best is None:
        return None, []
    return best.get("audio_faiss_idx"), best.get("domain_faiss_idxs", [])


def process_gt_dataset(
    dataset: EpisodeDataset,
    out_dir: Path,
    extracted_dir: Optional[Path] = None,
) -> None:
    """Process all episodes in dataset into GT JSONL files."""
    for episode in dataset:
        process_gt_episode(episode, out_dir, extracted_dir)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
