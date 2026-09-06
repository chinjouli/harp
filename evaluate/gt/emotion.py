from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import evaluate.gt_retriever as gtr


_TEXT_LIMIT = 200


def format_gt_segments(segs: List[Dict[str, Any]], max_segs: int = 3) -> str:
    """Format GT segments with emotion-specific fields for judge prompts."""
    if not segs:
        return "(no ground truth segments)"
    from evaluate.gt_retriever import _r
    blocks: List[str] = []
    for s in segs[:max_segs]:
        start, end = float(s["start"]), float(s["end"])
        spk = s.get("speaker", "?")
        lines = [f"{spk} [{start:.1f}s–{end:.1f}s]"]
        txt = s.get("text", "")
        if txt:
            if len(txt) > _TEXT_LIMIT:
                txt = txt[:_TEXT_LIMIT] + "…"
            lines.append(f"- text: \"{txt}\"")
        dl = s.get("domain_labels") or {}
        primary = dl.get("primary")
        scores = dl.get("scores") or {}
        if primary or scores:
            sc_str = ", ".join(f"{k}={_r(v)}" for k, v in scores.items()
                               if v is not None)
            emo = f"primary={primary}" if primary else ""
            lines.append(
                f"- emotion: {emo} [{sc_str}]" if sc_str else
                f"- emotion: {emo}"
            )
        avd = [(d, dl[d]) for d in ("valence", "arousal", "dominance")
               if dl.get(d) is not None]
        if avd:
            avd_str = ", ".join(f"{d}={_r(v)}" for d, v in avd)
            lines.append(f"- AVD [0–1]: {avd_str}")
        votes = dl.get("votes")
        if votes:
            lines.append(f"- votes: {votes}")
        blocks.append("\n".join(lines))
    if len(segs) > max_segs:
        blocks.append(f"… ({len(segs) - max_segs} more segments)")
    return "\n\n".join(blocks)


def get_gt_for_window(
    episode_id: str,
    start: float,
    end: float,
    gt_dir: Path,
) -> Dict[str, Any]:
    """GT segments overlapping [start, end] plus CT availability flag."""
    segs = gtr.load_gt_segments(episode_id, gt_dir)
    overlapping = gtr.filter_segments_window(segs, start, end)
    conv_dir = gt_dir.parent.parent / "msp_conversation_v2.0"
    ct_available = (
        (conv_dir / "Annotations").exists()
        and _has_conv_entry(episode_id, conv_dir)
    )
    return {"segments": overlapping, "ct_available": ct_available}


def read_ct_csv(
    episode_id: str,
    conv_dir: Path,
    dim: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Continuous-time annotations for one episode and dimension.

    Returns {start, end, mean, annotator_id} dicts from all annotator CSVs.
    """
    ann_dir = conv_dir / "Annotations" / dim
    if not ann_dir.exists():
        return []
    conv_id = episode_id.replace("MSP-PODCAST_", "MSP-Conversation_")
    pattern = re.compile(rf"^{re.escape(conv_id)}_P_\w+\.csv$")
    rows: List[Dict[str, Any]] = []
    for csv_path in sorted(ann_dir.iterdir()):
        if not pattern.match(csv_path.name):
            continue
        annotator = csv_path.stem.split("_P_")[-1]
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                r_start = float(row.get("Start_Time", 0.0))
                r_end = float(row.get("End_Time", 0.0))
                if start is not None and r_end <= start:
                    continue
                if end is not None and r_start >= end:
                    continue
                rows.append({
                    "start": r_start,
                    "end": r_end,
                    "mean": float(row.get("Mean", 0.0)),
                    "annotator_id": annotator,
                })
    return rows


def read_ct_all_dims(
    episode_id: str,
    conv_dir: Path,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """CT annotations for Arousal, Valence, and Dominance."""
    return {
        dim: read_ct_csv(episode_id, conv_dir, dim, start, end)
        for dim in ("Arousal", "Valence", "Dominance")
    }


def _has_conv_entry(episode_id: str, conv_dir: Path) -> bool:
    conv_id = episode_id.replace("MSP-PODCAST_", "MSP-Conversation_")
    time_file = conv_dir / "Time_Labels" / "conversations.txt"
    if not time_file.exists():
        return False
    with time_file.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith(conv_id + ";"):
                return True
    return False
