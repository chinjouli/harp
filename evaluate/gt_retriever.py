from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


_TEXT_LIMIT = 200  # max chars of transcript text per segment


def _r(v: Any) -> Any:
    """Round floats to 2 dp; pass everything else through."""
    return round(v, 2) if isinstance(v, float) else v


def format_gt_segments(segs: List[Dict[str, Any]], max_segs: int = 3) -> str:
    """Format GT segments as headed bullet lists for judge prompts."""
    if not segs:
        return "(no ground truth segments)"
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
        mean = dl.get("mean")
        if primary or scores:
            sc_str = ", ".join(f"{k}={_r(v)}" for k, v in scores.items()
                               if v is not None)
            parts = []
            if primary:
                parts.append(f"primary={primary}")
            if mean is not None:
                parts.append(f"mean={_r(mean)}")
            if sc_str:
                parts.append(f"[{sc_str}]")
            lines.append(f"- domain: {', '.join(parts)}")
        blocks.append("\n".join(lines))
    if len(segs) > max_segs:
        blocks.append(f"… ({len(segs) - max_segs} more segments)")
    return "\n\n".join(blocks)


def filter_segments_window(
    segs: List[Dict[str, Any]], start: float, end: float
) -> List[Dict[str, Any]]:
    return [s for s in segs if float(s["end"]) > start and float(s["start"]) < end]


def search_quote(
    segs: List[Dict[str, Any]], text: str
) -> List[Dict[str, Any]]:
    needle = text.lower().strip()
    return [s for s in segs if needle in (s.get("text") or "").lower()]


def get_segments_speaker(
    segs: List[Dict[str, Any]], speaker_id: str
) -> List[Dict[str, Any]]:
    return [s for s in segs if s.get("speaker") == speaker_id]


def load_gt_segments(
    episode_id: str, gt_dir: Path
) -> List[Dict[str, Any]]:
    """Load GT segments for one episode from gt/segments/."""
    path = gt_dir / "segments" / f"{episode_id}.jsonl"
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
