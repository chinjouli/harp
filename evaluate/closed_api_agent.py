"""EvalAgent variant that reads pre-extracted GT evidence from inf_evidence/
instead of loading GT files, and scores only answer + rationale.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

from evaluate import metrics
from evaluate.metrics import _parse_yn
from evaluate.agent import _TracingJudge
from inference.format import format_evidence_sets as _fmt_ev

if TYPE_CHECKING:
    from evaluate.judge.base import JudgeBase


class _CollectingJudge:
    """Records prompts without making API calls (returns dummy YES)."""
    def __init__(self) -> None:
        self.prompts: List[str] = []

    def __call__(self, prompt: str, **_: Any) -> str:
        self.prompts.append(prompt)
        return "YES"


class _InjectingJudge:
    """Returns pre-computed results in order."""
    def __init__(self, results: List[str]) -> None:
        self._it = iter(results)

    def __call__(self, prompt: str, **_: Any) -> str:
        return next(self._it)


class ClosedAPIEvalAgent:
    """Score predictions using pre-extracted GT evidence (answer + rationale only).

    Args:
        judge:             JudgeBase instance.
        inf_evidence_path: JSONL with {query_id, episode_id, gt_ev, gt_side}.
        prompts:           Optional task prompts module exposing SCHEMA_DESC.
    """

    def __init__(
        self,
        judge: "JudgeBase",
        inf_evidence_path: Path,
        prompts: Any = None,
        skip_faithful: bool = False,
        rationale_only: bool = False,
    ) -> None:
        self._judge = _TracingJudge(judge)
        self._prompts = prompts
        self._skip_faithful = skip_faithful
        self._rationale_only = rationale_only
        self._evidence: Dict[Any, Dict[str, Any]] = {}
        with open(inf_evidence_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    self._evidence[r["query_id"]] = r

    def __call__(
        self,
        prediction: Dict[str, Any],
        query_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        qid = query_item.get("query_id") or prediction.get("query_id")
        pred_str = str(prediction.get("prediction", ""))
        rationale = str(prediction.get("rationale", ""))
        evidence_sets: List[List[Dict[str, Any]]] = prediction.get("evidence_sets") or []

        schema_desc = getattr(self._prompts, "SCHEMA_DESC", "") if self._prompts else ""
        strip_spk_types = getattr(self._prompts, "STRIP_SPEAKER_TYPES", set()) if self._prompts else set()
        strip_spk = query_item.get("query_type", "") in strip_spk_types

        self._judge.reset()
        rat = self._score_rationale(
            pred_str, rationale, query_item,
            evidence_sets, self._judge, schema_desc, strip_spk,
            self._skip_faithful,
        )
        result: Dict[str, Any] = {
            "query_id": qid,
            "episode_id": (
                prediction.get("episode_id") or query_item.get("episode_id", "")
            ),
            **rat,
            "judge_trace": self._judge.trace,
        }
        if not self._rationale_only:
            result["answer"] = metrics.score_answer(pred_str, query_item, self._judge)
        return result

    def batch_score(
        self,
        predictions: List[Dict[str, Any]],
        queries: Dict[Any, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Score all predictions in one batch API call.

        Requires the wrapped judge to expose a batch(prompts) -> list[str] method.
        """
        real_judge = self._judge._judge

        def _run_pass(inner: Any) -> List[Dict[str, Any]]:
            self._judge._judge = inner
            out = []
            for pred in predictions:
                qid = pred.get("query_id")
                query_item = queries.get(qid, pred)
                pred_str = str(pred.get("prediction", ""))
                rationale = str(pred.get("rationale", ""))
                evidence_sets = pred.get("evidence_sets") or []
                schema_desc = (
                    getattr(self._prompts, "SCHEMA_DESC", "")
                    if self._prompts else ""
                )
                strip_spk_types = (
                    getattr(self._prompts, "STRIP_SPEAKER_TYPES", set())
                    if self._prompts else set()
                )
                strip_spk = query_item.get("query_type", "") in strip_spk_types
                self._judge.reset()
                rat = self._score_rationale(
                    pred_str, rationale, query_item,
                    evidence_sets, self._judge, schema_desc, strip_spk,
                    self._skip_faithful,
                )
                result: Dict[str, Any] = {
                    "query_id": qid,
                    "episode_id": (
                        pred.get("episode_id") or query_item.get("episode_id", "")
                    ),
                    **rat,
                    "judge_trace": self._judge.trace,
                }
                if not self._rationale_only:
                    result["answer"] = metrics.score_answer(
                        pred_str, query_item, self._judge
                    )
                out.append(result)
            return out

        # Pass 1: collect all prompts without API calls
        collector = _CollectingJudge()
        _run_pass(collector)

        # Batch API call
        all_results = real_judge.batch(collector.prompts)

        # Pass 2: inject results to build scores
        scores = _run_pass(_InjectingJudge(all_results))
        self._judge._judge = real_judge
        return scores

    def _snippet_prefix(self) -> str:
        """Prepend SNIPPET_NOTE from task prompts if available."""
        note = getattr(self._prompts, "SNIPPET_NOTE", "") if self._prompts else ""
        return f"{note}\n\n" if note else ""

    def _score_rationale(
        self,
        prediction: str,
        rationale: str,
        query_item: Dict[str, Any],
        evidence_sets: List[List[Dict[str, Any]]],
        judge: _TracingJudge,
        schema_desc: str,
        strip_speaker: bool,
        skip_faithful: bool = False,
    ) -> Dict[str, int]:
        factual = self._score_factual(
            prediction, rationale, query_item, judge, schema_desc,
        )
        faithful = (
            1 if skip_faithful
            else self._score_faithful(
                prediction, rationale, query_item, evidence_sets, judge,
                strip_speaker, schema_desc,
            )
        )
        return {
            "factual": factual,
            "faithful": faithful,
            "rationale": factual & faithful,
        }

    def _score_factual(
        self,
        prediction: str,
        rationale: str,
        query_item: Dict[str, Any],
        judge: _TracingJudge,
        schema_desc: str,
    ) -> int:
        """Checks answer/rationale against the structured GT label (judge_info)."""
        prefix = self._snippet_prefix()
        header = f"{prefix}{schema_desc}\n\n" if schema_desc else prefix
        query_text = query_item.get("query", "")
        query_type = query_item.get("query_type", "")
        gt_summary = metrics._format_judge_info(query_item)
        prompt = (
            f"{header}"
            f"Query ({query_type}): {query_text}\n\n"
            f"## GT label\n{gt_summary}\n\n"
            f"Model answer: {prediction}\n"
            f"Model rationale: {rationale}\n\n"
            "Do the factual claims in the rationale agree with the GT label above? "
            "Numeric values may be on a different scale — accept if the direction "
            "or relative ordering is correct. "
            "Time-range phrases in the query ('from X to Y', 'between X and Y') "
            "define a search window — any event timestamp strictly within that "
            "interval is valid, not just times at the stated boundaries. "
            + (
                "If the GT label shows a timestamp, the rationale must cite a time "
                "within that window (±3 s); wrong timestamp is a factual error — mark NO. "
                if query_type in ("locate", "locate_hard") else ""
            )
            + "Where a reference text is shown, the spoken content cited in the "
            "rationale should be from the same segment (similar words or topic). "
            "Reply YES or NO. If NO, add one short sentence explaining why."
        )
        verdict = judge(prompt).strip()
        return _parse_yn(verdict)

    def _score_faithful(
        self,
        prediction: str,
        rationale: str,
        query_item: Dict[str, Any],
        evidence_sets: List[List[Dict[str, Any]]],
        judge: _TracingJudge,
        strip_speaker: bool = False,
        schema_desc: str = "",
    ) -> int:
        """Checks rationale is grounded in the retrieved evidence."""
        prefix = self._snippet_prefix()
        header = f"{prefix}{schema_desc}\n\n" if schema_desc else prefix
        query_text = query_item.get("query", "")
        query_type = query_item.get("query_type", "")
        retrieved = _fmt_ev(evidence_sets, strip_speaker=strip_speaker)
        prompt = (
            f"{header}"
            f"Query ({query_type}): {query_text}\n\n"
            f"## Retrieved evidence\n{retrieved}\n\n"
            f"Model answer: {prediction}\n"
            f"Model rationale: {rationale}\n\n"
            "Is the rationale faithful to the retrieved evidence above? "
            "Values in evidence are raw model outputs; the rationale may "
            "cite them with different rounding — accept if traceable. "
            "Claims about audio input clips ([Snippet E], [A's audio "
            "example], [B's audio example], [audio example], etc.) come "
            "from the model's direct audio perception, NOT the retrieved "
            "evidence text — do NOT mark faithful=NO for such claims. "
            "Reply YES if claims are grounded in evidence (or audio input), "
            "NO only if the rationale introduces scores or facts that "
            "cannot be found in the retrieved evidence and were not from "
            "an audio input clip. If NO, add one short sentence why."
        )
        verdict = judge(prompt).strip()
        return _parse_yn(verdict)
