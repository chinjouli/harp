#!/usr/bin/env python3
"""Convert tracks_metadata.jsonl into per-track GT segment JSONL files
and enrich a music queries.jsonl with song timing in one pass.

Usage:
    python dataset/process_gt_music.py \\
        ../songeval/tracks_metadata.jsonl \\
        data/songeval/gt \\
        data/songeval/queries.jsonl
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_ANN_DIMS = ("Coherence", "Musicality", "Memorability", "Clarity", "Naturalness")


def _aggregate(annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores: Dict[str, float] = {}
    for dim in _ANN_DIMS:
        vals = [a[dim] for a in annotations if dim in a]
        scores[dim] = sum(vals) / len(vals) if vals else 0.0
    mean = sum(scores.values()) / len(scores) if scores else 0.0
    return {"scores": scores, "mean": round(mean, 3)}


def process(
    metadata_path: Path,
    out_dir: Path,
    queries_path: Optional[Path] = None,
) -> None:
    """Write GT segments and optionally enrich queries, reading metadata once."""
    tracks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    file_timing: Dict[str, Dict[str, float]] = {}

    with Path(metadata_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tracks[rec["track_name"]].append(rec)
            file_timing[rec["file_name"]] = {
                "start": float(rec["start_time"]),
                "end": float(rec["end_time"]),
            }

    # write GT segments
    seg_dir = Path(out_dir) / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    for track_name, records in tracks.items():
        records.sort(key=lambda r: r["start_time"])
        with (seg_dir / f"{track_name}.jsonl").open("w", encoding="utf-8") as f:
            for idx, rec in enumerate(records):
                seg = {
                    "seg_id": f"{track_name}_{idx:04d}",
                    "episode_id": track_name,
                    "start": float(rec["start_time"]),
                    "end": float(rec["end_time"]),
                    "speaker": f"SONG_{idx+1:02d}",
                    "file_name": rec["file_name"],
                    "text": "",
                    "domain_labels": _aggregate(rec.get("annotation", [])),
                    "speaker_faiss_idx": None,
                    "domain_faiss_idxs": [],
                }
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")
    print(f"Wrote GT segments for {len(tracks)} tracks to {seg_dir}")

    if queries_path is None:
        return

    # enrich queries
    queries: List[Dict[str, Any]] = []
    with Path(queries_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    for q in queries:
        # resolve near-ties to A/B; keep "tie" only for delta == 0
        gt = q.get("ground_truth", {})
        if gt.get("winner") == "tie":
            delta = gt.get("delta", 0)
            if delta > 0:
                gt["winner"] = "A"
            elif delta < 0:
                gt["winner"] = "B"

        ji = q.setdefault("judge_info", {})
        for key, label in (("song_a", "a"), ("song_b", "b")):
            blk = q.get(key)
            if not isinstance(blk, dict):
                continue
            timing = file_timing.get(blk.get("file_name", ""))
            if timing is None:
                continue
            blk["time_ref"] = (timing["start"] + timing["end"]) / 2
            ji[f"start_{label}"] = timing["start"]
            ji[f"end_{label}"] = timing["end"]

    with Path(queries_path).open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Enriched {len(queries)} queries in {queries_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", help="Path to tracks_metadata.jsonl")
    parser.add_argument("out_dir", help="Output GT directory")
    parser.add_argument("queries", nargs="?", help="Path to queries.jsonl to enrich")
    args = parser.parse_args()
    process(
        Path(args.metadata),
        Path(args.out_dir),
        Path(args.queries) if args.queries else None,
    )
