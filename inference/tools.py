from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from extract.utils import load_audio

_HALF = 30  # seconds on each side for point-mode localize

_MMSS_RE = re.compile(r"(\d+):(\d{2})")  # MM:SS tokens
_SONG_ORD_RE = re.compile(r"(\d+)(?:st|nd|rd|th)?\s*song", re.I)
_FULL_TRACK_RE = re.compile(r"full\s*(track|recording)|whole\s*track|entire\s*track|all\s+sections?", re.I)
_FULL_TRACK_WINDOW = (0, 10 ** 4)
# "X min Y sec" compound unit → total seconds
_COMPOUND_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*min(?:utes?)?\s+(\d+(?:\.\d+)?)\s*sec(?:onds?)?",
    re.I,
)
# simple "X to Y <unit>" — only valid when both values share one unit token
_RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(min(?:utes?)?|sec(?:onds?)?)?\s+to\s+"
    r"(\d+(?:\.\d+)?)\s*(min(?:utes?)?|sec(?:onds?)?)",
    re.I,
)
_POINT_RE = re.compile(
    r"(?:around|at|~)?\s*(\d+(?:\.\d+)?)\s*(min(?:utes?)?|sec(?:onds?)?)",
    re.I,
)


def _compound_to_sec(desc: str) -> list[int]:
    """Extract all 'X min Y sec' compound times from desc, in order."""
    return [
        int(float(m.group(1)) * 60 + float(m.group(2)))
        for m in _COMPOUND_RE.finditer(desc)
    ]


@dataclass
class FaissStore:
    """A FAISS index paired with its metadata, loaded once at agent init."""
    index: Any                        # faiss.Index
    meta: list[dict[str, Any]] = field(default_factory=list)


def load_faiss_store(index_path: Path, meta_path: Path) -> FaissStore | None:
    """Load index + meta from disk; returns None if either file is missing."""
    import faiss
    index_path, meta_path = Path(index_path), Path(meta_path)
    if not index_path.exists() or not meta_path.exists():
        return None
    index = faiss.read_index(str(index_path))
    meta: list[dict] = []
    with meta_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    return FaissStore(index=index, meta=meta)


# -- time localization --

def _mmss_to_sec(mm: str, ss: str) -> int:
    return int(mm) * 60 + int(ss)


def localize(
    desc: str, segments: list[dict] | None = None
) -> tuple[int, int]:
    """Parse a time description into (start, end) seconds.

    Compound range: "52 min 30 sec to 53 min 30 sec" → (3150, 3210)
    Compound point: "52 min 30 sec"                  → (3120, 3180)
    MM:SS range:    "78:30 to 79:30"                 → (4710, 4770)
    MM:SS point:    "78:30"                          → (4680, 4740)
    Simple range:   "2 to 5 minutes"                 → (120, 300)
    Simple point:   "around 5 minutes"               → (270, 330)
    Song ordinal:   "16th song" (needs segments)     → that song's window
    """
    if _FULL_TRACK_RE.search(desc):
        return _FULL_TRACK_WINDOW
    if segments is not None:
        m = _SONG_ORD_RE.search(desc)
        if m:
            n = int(m.group(1))
            tag = f"SONG_{n:02d}"
            for seg in segments:
                if seg.get("speaker") == tag:
                    return int(float(seg["start"])), int(float(seg["end"]))
            raise ValueError(f"Song position {n} (tag {tag!r}) not in segments")
    # Compound "X min Y sec" — handles multi-unit specs before simpler regexes
    compound = _compound_to_sec(desc)
    if len(compound) >= 2:
        return min(compound[0], compound[1]), max(compound[0], compound[1])
    if len(compound) == 1:
        t = compound[0]
        return max(0, t - _HALF), t + _HALF

    # MM:SS tokens (e.g. "78:30 to 79:30")
    mmss_hits = _MMSS_RE.findall(desc)
    if len(mmss_hits) >= 2:
        a = _mmss_to_sec(*mmss_hits[0])
        b = _mmss_to_sec(*mmss_hits[1])
        return min(a, b), max(a, b)
    if len(mmss_hits) == 1:
        t = _mmss_to_sec(*mmss_hits[0])
        return max(0, t - _HALF), t + _HALF

    # Simple "X to Y <unit>" (both on the same unit)
    m = _RANGE_RE.search(desc)
    if m:
        a, ua, b, ub = m.group(1), m.group(2), m.group(3), m.group(4)
        a, b = float(a), float(b)
        ua = ua or ub  # "2 to 5 minutes" → both minutes
        if ua and ua[0].lower() == "m":
            a *= 60
        if ub and ub[0].lower() == "m":
            b *= 60
        return int(a), int(b)

    m = _POINT_RE.search(desc)
    if m:
        t, u = float(m.group(1)), m.group(2)
        if u and u[0].lower() == "m":
            t *= 60
        return max(0, int(t) - _HALF), int(t) + _HALF
    raise ValueError(f"Cannot parse time description: {desc!r}")


# -- segment loading --

def load_segments(episode_id: str, segments_dir: Path) -> list[dict[str, Any]]:
    path = Path(segments_dir) / f"{episode_id}.jsonl"
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# -- retrieval tools --

def search_by_time(
    segments: list[dict[str, Any]],
    window: tuple[int, int],
) -> list[dict[str, Any]]:
    """Return all segments overlapping the window, scored by overlap ratio."""
    lo, hi = window
    w = hi - lo or 1
    hits = []
    for s in segments:
        ov = min(float(s["end"]), hi) - max(float(s["start"]), lo)
        if ov > 0:
            hits.append({**s, "_sources": {"time": round(ov / w, 3)}})
    hits.sort(key=lambda x: x["_sources"]["time"], reverse=True)
    return hits


def search_by_label(
    segments: list[dict[str, Any]],
    label: str,
    window: tuple[int, int],
) -> list[dict[str, Any]]:
    """Return segments whose domain_labels.primary matches label (case-insensitive)."""
    lo, hi = window
    label_l = label.lower()
    hits = []
    for s in segments:
        if float(s["end"]) <= lo or float(s["start"]) >= hi:
            continue
        dl = s.get("domain_labels") or {}
        primary = dl.get("primary", "")
        if str(primary).lower() == label_l:
            score = (dl.get("scores") or {}).get(primary, 1.0)
            hits.append({**s, "_sources": {"label": round(score, 3)}})
    return hits


def search_keyword(
    segments: list[dict[str, Any]],
    keyword: str,
    window: tuple[int, int],
    limit: int = 10,
) -> list[dict[str, Any]]:
    lo, hi = window
    kw = keyword.lower()
    hits = []
    for s in segments:
        if float(s["end"]) <= lo or float(s["start"]) >= hi:
            continue
        if kw in str(s.get("text", "")).lower():
            hits.append({**s, "_sources": {"keyword": 1.0}})
    return hits[:limit]


def search_audio(
    episode_id: str,
    window: tuple[int, int],
    audio_dir: Path,
    _cache: dict | None = None,
) -> tuple[np.ndarray, int] | None:
    """Return (audio_array, sample_rate) for the window, or None.

    Pass a dict as _cache to avoid reloading the same track file per query.
    """
    for ext in ("flac", "wav", "mp3"):
        p = Path(audio_dir) / f"{episode_id}.{ext}"
        if p.exists():
            key = str(p)
            if _cache is not None and key in _cache:
                wave, sr = _cache[key]
            else:
                wave, sr = load_audio(str(p))
                if _cache is not None:
                    _cache[key] = (wave, sr)
            lo, hi = window
            return wave[int(lo * sr): int(hi * sr)], sr
    return None


def search_text_emb(
    text: str,
    emb_model: Any,
    store: FaissStore | None,
    seg_map: dict[str, dict[str, Any]],
    episode_id: str,
    window: tuple[int, int],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Semantic search using a text embedding of the given phrase."""
    if store is None or emb_model is None:
        return []
    qvec = emb_model.embed(text)
    meta_hits = _faiss_search_meta(qvec, store, episode_id, window, top_k)
    results = [
        {**seg, "_sources": {"text_emb": round(entry["score"], 3)}}
        for entry in meta_hits
        if (seg := seg_map.get(entry.get("seg_id", ""))) is not None
    ]
    results.sort(key=lambda x: x["_sources"]["text_emb"], reverse=True)
    return results


def search_audio_emb(
    wav_path: str,
    emb_model: Any,
    store: FaissStore | None,
    seg_map: dict[str, dict[str, Any]],
    episode_id: str,
    window: tuple[int, int],
    top_k: int = 5,
    use_vec_cache: bool = False,
    vec_cache: dict | None = None,
) -> list[dict[str, Any]]:
    """Find segments by the same speaker via audio embedding."""
    return _emb_search(wav_path, emb_model, store, seg_map, episode_id, window, top_k,
                       "audio_emb", use_vec_cache, vec_cache)


def search_domain_emb(
    wav_path: str,
    emb_model: Any,
    store: FaissStore | None,
    seg_map: dict[str, dict[str, Any]],
    episode_id: str,
    window: tuple[int, int],
    top_k: int = 5,
    use_vec_cache: bool = False,
    vec_cache: dict | None = None,
) -> list[dict[str, Any]]:
    """Find segments with similar acoustic/domain content via embedding."""
    return _emb_search(wav_path, emb_model, store, seg_map, episode_id, window, top_k,
                       "domain_emb", use_vec_cache, vec_cache)


def gather_evidence(
    hits_list: list[list[dict[str, Any]]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Merge hits from multiple tools: dedup by seg_id, keep top_k by score."""
    merged: dict[str, dict] = {}
    for hits in hits_list:
        for h in hits:
            sid = h.get("seg_id", "")
            if sid not in merged:
                merged[sid] = {**h, "_sources": {}}
            for src, score in (h.get("_sources") or {}).items():
                if score > merged[sid]["_sources"].get(src, 0):
                    merged[sid]["_sources"][src] = round(score, 3)
    ranked = sorted(
        merged.values(),
        key=lambda s: sum(s.get("_sources", {}).values()),
        reverse=True,
    )
    return [_format_seg(s) for s in sorted(ranked[:top_k], key=lambda s: float(s["start"]))]


# -- internal helpers --

def _emb_search(
    wav_path: str,
    emb_model: Any,
    store: FaissStore | None,
    seg_map: dict[str, dict[str, Any]],
    episode_id: str,
    window: tuple[int, int],
    top_k: int,
    source: str,
    use_vec_cache: bool = False,
    vec_cache: dict | None = None,
) -> list[dict[str, Any]]:
    import warnings
    if store is None:
        warnings.warn(f"_emb_search({source}): FAISS store is None, skipping.")
        return []
    if use_vec_cache:
        if wav_path not in vec_cache:
            import warnings
            warnings.warn(
                f"_emb_search({source}): {wav_path} not in cache, skipping."
            )
            return []
        qvec = vec_cache[wav_path]
    elif emb_model is None:
        warnings.warn(f"_emb_search({source}): emb_model is None, skipping.")
        return []
    else:
        wave, sr = load_audio(wav_path)
        qvec = emb_model.embed(wave, sr)
    meta_hits = _faiss_search_meta(qvec, store, episode_id, window, top_k)
    results = [
        {**seg, "_sources": {source: round(entry["score"], 3)}}
        for entry in meta_hits
        if (seg := seg_map.get(entry.get("seg_id", ""))) is not None
    ]
    results.sort(key=lambda x: x["_sources"][source], reverse=True)
    return results


def _faiss_search_meta(
    qvec: np.ndarray,
    store: FaissStore,
    episode_id: str,
    window: tuple[int, int],
    top_k: int,
) -> list[dict[str, Any]]:
    """Search store.index restricted to the episode+window subset."""
    import faiss

    lo, hi = window

    # Filter meta to the episode+window subset
    subset = [
        e for e in store.meta
        if e.get("episode_id") == episode_id
        and float(e.get("end", 0)) > lo
        and float(e.get("start", 0)) < hi
        and int(e.get("faiss_idx", -1)) >= 0
    ]
    if not subset:
        return []

    valid_ids = np.array([int(e["faiss_idx"]) for e in subset], dtype=np.int64)
    sel = faiss.IDSelectorArray(len(valid_ids), faiss.swig_ptr(valid_ids))

    qv = np.asarray(qvec, dtype=np.float32)
    if qv.ndim == 1:
        qv = qv[None, :]
    faiss.normalize_L2(qv)

    k = min(len(valid_ids), top_k)
    dists, idxs = store.index.search(qv, k, params=faiss.SearchParameters(sel=sel))

    score_map = {int(idxs[0][r]): float(dists[0][r]) for r in range(k) if int(idxs[0][r]) >= 0}
    fi_to_entry = {int(e["faiss_idx"]): e for e in subset}
    results = [
        {**fi_to_entry[fi], "score": score}
        for fi, score in score_map.items()
        if fi in fi_to_entry
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _format_seg(s: dict[str, Any]) -> dict[str, Any]:
    labels = s.get("domain_labels") or {}
    return {
        "time": [float(s["start"]), float(s["end"])],
        "speaker": s.get("speaker"),
        "text": s.get("text", ""),
        "domain_labels": {k: v for k, v in labels.items() if v is not None},
        "_sources": s.get("_sources", {}),
    }
