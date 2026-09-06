from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import evaluate.gt_retriever as gtr
from evaluate import metrics
from evaluate.gt.emotion import read_ct_all_dims
from evaluate.gt.music import build_file_index

if TYPE_CHECKING:
    from evaluate.judge.base import JudgeBase


class _TracingJudge:
    """Wraps a JudgeBase and records every (prompt, verdict) pair."""

    def __init__(self, judge: JudgeBase) -> None:
        self._judge = judge
        self.trace: List[Dict[str, str]] = []

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        verdict = self._judge(prompt, **kwargs)
        self.trace.append({"prompt": prompt, "verdict": verdict})
        return verdict

    def reset(self) -> None:
        self.trace = []


class EvalAgent:
    """Evaluation agent: loads GT, retrieves evidence, scores a prediction.

    Args:
        judge:    JudgeBase instance for scoring.
        gt_dir:   Directory containing gt/segments/{episode_id}.jsonl files.
        prompts:  Task-specific prompts module exposing SCHEMA_DESC.
        conv_dir: Optional path to msp_conversation_v2.0/ for CT traces.
    """

    def __init__(
        self,
        judge: JudgeBase,
        gt_dir: Optional[Path] = None,
        prompts: Any = None,
        conv_dir: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        skip_faithful: bool = False,
        rationale_only: bool = False,
    ) -> None:
        self._tracing = _TracingJudge(judge)
        self._gt_dir = Path(gt_dir) if gt_dir else None
        self._prompts = prompts
        self._conv_dir = Path(conv_dir) if conv_dir else None
        self._metadata_path = Path(metadata_path) if metadata_path else None
        self._file_index_cache: Dict[str, Dict[str, str]] = {}
        self._skip_faithful = skip_faithful
        self._rationale_only = rationale_only

    def __call__(
        self,
        prediction: Dict[str, Any],
        query_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        episode_id = str(query_item.get("episode_id", ""))

        gt_segs = gtr.load_gt_segments(episode_id, self._gt_dir) if self._gt_dir else []
        ct_info = (
            read_ct_all_dims(episode_id, self._conv_dir)
            if self._conv_dir and self._conv_dir.exists()
            else {}
        )
        query_item = self._enrich_query(query_item, episode_id, gt_segs)

        schema_desc = getattr(self._prompts, "SCHEMA_DESC", "") if self._prompts else ""
        fmt_segs = getattr(self._prompts, "format_gt_segments", None) if self._prompts else None
        self._tracing.reset()
        pred_str = str(prediction.get("prediction", ""))
        rationale = str(prediction.get("rationale", ""))
        evidence_sets = prediction.get("evidence_sets") or []
        plan = prediction.get("plan") or {}
        scores = metrics.score_all(
            pred_str, rationale, query_item, gt_segs, ct_info,
            evidence_sets, plan, self._tracing, schema_desc, fmt_segs,
            self._skip_faithful, self._rationale_only,
        )
        gt_ev = metrics.get_gt_evidence_segs(query_item, gt_segs)
        gt_side = metrics._retrieve_gt_side(
            query_item, gt_segs, ct_info, fmt_segs or gtr.format_gt_segments
        )
        return {
            "query_id": query_item.get("query_id"),
            "episode_id": episode_id,
            **scores,
            "judge_trace": self._tracing.trace,
            "_gt_ev": gt_ev,
            "_gt_side": gt_side,
        }

    def _enrich_query(
        self,
        query_item: Dict[str, Any],
        episode_id: str,
        gt_segs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Inject gt_seg_ids for audio queries using file_name → SONG_XX index."""
        if episode_id not in self._file_index_cache:
            self._file_index_cache[episode_id] = build_file_index(gt_segs)
        file_index = self._file_index_cache[episode_id]
        speaker_to_seg = {s["speaker"]: s for s in gt_segs}
        seg_ids = []
        for key in ("song_a", "song_b", "speaker_a", "speaker_b"):
            block = query_item.get(key)
            if not isinstance(block, dict) or "file_name" not in block:
                continue
            speaker = file_index.get(block["file_name"])
            seg = speaker and speaker_to_seg.get(speaker)
            if seg:
                seg_ids.append(seg["seg_id"])
        if not seg_ids or query_item.get("gt_seg_ids"):
            return query_item
        return {**query_item, "gt_seg_ids": seg_ids}
