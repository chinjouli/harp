from __future__ import annotations

from pathlib import Path
from typing import Any

from extract.utils import load_audio


def preprocess_clips(
    query_item: dict[str, Any],
    task: str,
    clip_audio_dir: Path,
    asr: Any | None = None,
    domain_labeler: Any | None = None,
    label_cache: dict | None = None,
) -> str | None:
    """Hardcoded clip preprocessing for baseline (no-embedding) modes.

    Conditions:
      task==emotion + query_type==locate  →  run domain_labeler on audio_emo
      task==music   + query_type==audio   →  run asr on song_a/song_b clips

    Returns a clip_info string to inject into the plan prompt, or None.
    """
    qt = query_item.get("query_type")
    clip_audio_dir = Path(clip_audio_dir)

    if task == "emotion" and qt in ("locate", "locate_hard"):
        return _label_emotion_clip(
            query_item, clip_audio_dir, domain_labeler, label_cache
        )

    if task == "music" and qt == "audio":
        return _transcribe_music_clips(
            query_item, clip_audio_dir, asr, label_cache
        )

    return None


def _label_emotion_clip(
    query_item: dict[str, Any],
    clip_audio_dir: Path,
    domain_labeler: Any | None,
    label_cache: dict | None = None,
) -> str | None:
    fname = query_item.get("audio_emo")
    if not fname:
        return None
    path = str(clip_audio_dir / fname)
    if label_cache and path in label_cache:
        labels = label_cache[path]
    elif domain_labeler is not None:
        import os
        if not os.path.exists(path):
            return None
        wave, sr = load_audio(path)
        labels = domain_labeler.label(wave, sr)
    else:
        return None
    primary = labels.get("primary", "unknown")
    scores = labels.get("scores", {})
    scores_str = ", ".join(f"{k}={v:.2f}" for k, v in scores.items())
    return f"Emotion reference clip ({fname}): primary={primary}, scores={{{scores_str}}}"


def _transcribe_music_clips(
    query_item: dict[str, Any],
    clip_audio_dir: Path,
    asr: Any | None,
    label_cache: dict | None = None,
) -> str | None:
    lines = []
    for key, label in (("song_a", "Song A"), ("song_b", "Song B")):
        info = query_item.get(key)
        if not info:
            continue
        fname = info.get("audio_clip")
        if not fname:
            continue
        p = clip_audio_dir / fname
        if not p.exists():
            p2 = clip_audio_dir / "audio" / Path(fname).name
            if p2.exists():
                p = p2
        path = str(p)
        if label_cache and path in label_cache:
            text = label_cache[path]
        elif asr is not None:
            import os
            if not os.path.exists(path):
                continue
            wave, sr = load_audio(path)
            text = asr.transcribe(wave, sr)
        else:
            continue
        lines.append(f"{label} clip ({fname}): lyrics=\"{text.strip()}\"")
    return "\n".join(lines) if lines else None
