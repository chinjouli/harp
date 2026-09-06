from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from dataset.base import EpisodeDataset


class _NaNFixStream:
    """Wraps a binary file, replacing bare NaN/Infinity with null."""

    _pat = re.compile(rb"\bNaN\b|\bInfinity\b|-Infinity\b")

    def __init__(self, fobj: Any) -> None:
        self._f = fobj

    def read(self, n: int = -1) -> bytes:
        return self._pat.sub(b"null", self._f.read(n))

# MSP emotion class labels (from labels_consensus.csv EmoClass field)
# N=Neutral, A=Angry, S=Sad, H=Happy, U=Surprised, F=Fear,
# D=Disgust, C=Contempt, O=Other, X=No agreement
_SAM_MIN, _SAM_MAX = 1.0, 7.0  # SAM scale range in MSP-Podcast v2.0


def _norm_sam(v: float) -> float:
    return (float(v) - _SAM_MIN) / (_SAM_MAX - _SAM_MIN)


class LongEmoDataset(EpisodeDataset):
    """MSP-Podcast + MSP-Conversation dataset loader.

    Yields EpisodeRecord dicts. GT emotion labels come from
    msp_podcast_v2.0 labels; transcription text from all_data.json.
    CT traces are NOT pre-loaded — use evaluate.gt_retriever.read_ct_csv.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "all",
        include_ct: bool = True,
        max_episodes: int = -1,
        streaming_json: bool = True,
        episode_list: Optional[str] = None,
    ) -> None:
        self._root = Path(data_dir)
        self._split = split
        self._include_ct = include_ct
        self._max_episodes = max_episodes
        self._streaming_json = streaming_json

        # paths
        self._flac_dir = self._root / "podcasts_flac"
        self._v2_dir = self._root / "msp_podcast_v2.0"
        self._conv_dir = self._root / "msp_conversation_v2.0"
        self._all_data = self._root / "all_data.json"

        # load label tables once (small compared to audio)
        self._consensus = self._load_consensus()
        self._detailed = self._load_detailed()
        self._conv_episodes: Set[str] = (
            self._load_conv_episodes() if include_ct else set()
        )
        # episode_list overrides split filtering
        if episode_list is not None:
            self._split_episodes: Optional[Set[str]] = (
                self._load_episode_list(episode_list)
            )
        else:
            self._split_episodes = (
                self._load_split_episodes() if split != "all" else None
            )

    # EpisodeDataset interface

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        count = 0
        if self._streaming_json:
            yield from self._iter_streaming_json(count)
        else:
            yield from self._iter_full(count)

    def get(self, episode_id: str) -> Dict[str, Any]:
        # single-episode lookup: load only that episode from all_data.json
        with open(self._all_data, "r", encoding="utf-8") as f:
            data = json.load(f)
        ep_data = data.get(episode_id, {})
        return self._build_record(episode_id, ep_data)

    # Internal helpers

    def _iter_streaming_json(self, count: int) -> Iterator[Dict[str, Any]]:
        import ijson
        with open(self._all_data, "rb") as f:
            for episode_id, ep_data in ijson.kvitems(
                _NaNFixStream(f), ""
            ):
                if not self._episode_in_scope(episode_id):
                    continue
                yield self._build_record(episode_id, ep_data)
                count += 1
                if self._max_episodes > 0 and count >= self._max_episodes:
                    return

    def _iter_full(self, count: int) -> Iterator[Dict[str, Any]]:
        with open(self._all_data, "r", encoding="utf-8") as f:
            data = json.load(f)
        for episode_id, ep_data in data.items():
            if not self._episode_in_scope(episode_id):
                continue
            yield self._build_record(episode_id, ep_data)
            count += 1
            if self._max_episodes > 0 and count >= self._max_episodes:
                return

    def _episode_in_scope(self, episode_id: str) -> bool:
        if self._split_episodes is not None:
            return episode_id in self._split_episodes
        return True

    def _build_record(
        self, episode_id: str, ep_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        audio_path = str(
            self._flac_dir / f"{episode_id}.flac"
        )
        gt_segments = self._build_gt_segments(episode_id, ep_data)
        return {
            "episode_id": episode_id,
            "audio_path": audio_path,
            "gt_segments": gt_segments,
            # ct_annotations intentionally empty here:
            # use evaluate.gt_retriever.read_ct_csv at eval time
            "ct_annotations": (
                {"available": True}
                if episode_id in self._conv_episodes
                else {}
            ),
        }

    def _build_gt_segments(
        self, episode_id: str, ep_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        for seg_num, seg in ep_data.items():
            # seg_num is "1", "2", ... → pad to 4 digits for seg_id
            seg_id = f"{episode_id}_{int(seg_num):04d}"
            wav_key = f"{seg_id}.wav"

            con = self._consensus.get(wav_key, {})
            det = self._detailed.get(wav_key, [])
            domain_labels = _build_emotion_labels(con, det)

            # speaker: first global speaker ID, fall back to ""
            spk_info = seg.get("global_speaker2", [])
            speaker = str(spk_info[0]) if spk_info else ""

            segments.append({
                "seg_id": seg_id,
                "episode_id": episode_id,
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "speaker": speaker,
                "text": str(seg.get("text", "")),
                "domain_labels": domain_labels,
                "audio_faiss_idx": None,
                "domain_faiss_idxs": [],
            })
        return segments

    def _load_consensus(self) -> Dict[str, Dict[str, Any]]:
        path = self._v2_dir / "Labels" / "labels_consensus.csv"
        out: Dict[str, Dict[str, Any]] = {}
        if not path.exists():
            return out
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[row["FileName"]] = row
        return out

    def _load_detailed(self) -> Dict[str, List[Dict[str, Any]]]:
        path = self._v2_dir / "Labels" / "labels_detailed.csv"
        out: Dict[str, List[Dict[str, Any]]] = {}
        if not path.exists():
            return out
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out.setdefault(row["FileName"], []).append(row)
        return out

    def _load_conv_episodes(self) -> Set[str]:
        path = (
            self._conv_dir / "Time_Labels" / "conversations.txt"
        )
        ids: Set[str] = set()
        if not path.exists():
            return ids
        with path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(";")
                if parts:
                    # MSP-Conversation_DDDD → MSP-PODCAST_DDDD
                    conv_id = parts[0].strip()
                    ep_id = conv_id.replace(
                        "MSP-Conversation_", "MSP-PODCAST_"
                    )
                    ids.add(ep_id)
        return ids

    def _load_episode_list(self, path: str) -> Set[str]:
        """Load episode IDs from a JSONL with 'conversation' field.

        Maps MSP-Conversation_DDDD → MSP-PODCAST_DDDD.
        """
        ids: Set[str] = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                obj = json.loads(line)
                conv = obj.get("conversation", "")
                # MSP-Conversation_DDDD → MSP-PODCAST_DDDD
                suffix = conv.split("_")[-1]
                ids.add(f"MSP-PODCAST_{suffix}")
        return ids

    def _load_split_episodes(self) -> Set[str]:
        path = self._v2_dir / "Partitions.txt"
        ids: Set[str] = set()
        if not path.exists():
            return ids
        with path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("; ", 1)
                if len(parts) == 2 and parts[0] == self._split:
                    # "MSP-PODCAST_0001_0001.wav" → "MSP-PODCAST_0001"
                    wav = parts[1]
                    ep_id = "_".join(wav.split("_")[:2])
                    ids.add(ep_id)
        return ids


def _build_emotion_labels(
    con: Dict[str, Any],
    det: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build domain_labels dict from consensus + detailed rows."""
    if not con:
        return {
            "primary": None, "scores": {}, "secondary": None,
            "votes": None, "valence": None, "arousal": None,
            "dominance": None,
        }

    primary = con.get("EmoClass", "").strip() or None

    # vote counts and scores from per-annotator rows
    vote_counts: Dict[str, int] = {}
    for row in det:
        cls = row.get("EmoClass_Major", "").strip()
        if cls:
            vote_counts[cls] = vote_counts.get(cls, 0) + 1

    total = sum(vote_counts.values()) or 1
    scores = {k: v / total for k, v in vote_counts.items()}

    sorted_votes = sorted(
        vote_counts.items(), key=lambda x: x[1], reverse=True
    )
    secondary = (
        sorted_votes[1][0]
        if len(sorted_votes) > 1 else None
    )

    def _safe_sam(key: str) -> Optional[float]:
        val = con.get(key, "").strip()
        try:
            return _norm_sam(float(val))
        except (ValueError, TypeError):
            return None

    return {
        "primary": primary,
        "scores": scores,
        "secondary": secondary,
        "votes": vote_counts if vote_counts else None,
        "valence": _safe_sam("EmoVal"),
        "arousal": _safe_sam("EmoAct"),
        "dominance": _safe_sam("EmoDom"),
    }
