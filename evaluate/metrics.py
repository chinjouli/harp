from __future__ import annotations

import json
import re
from typing import Any, Dict, List, TYPE_CHECKING

import evaluate.gt_retriever as gtr
from inference.format import format_evidence_sets as _format_evidence_sets
from inference.tools import localize as parse_localize

if TYPE_CHECKING:
    from evaluate.judge.base import JudgeBase

_GT_WINDOW_SEC = 30  # fallback window around time_ref when gt_seg_ids absent

# Required tool groups per query type.
# Each group is a set of alternative names; one active tool per group counts.
# Types absent from this map use only time coverage (no tool requirement).
_REQUIRED_TOOLS: Dict[str, List[set]] = {
    # --- emotion (MSP) ---
    "state": [
        {"audio_emb"},            # speaker identification
        {"text_emb", "keyword"},  # content / topic
    ],
    "comparison": [
        {"audio_emb"},            # speaker identification
        {"text_emb", "keyword"},  # content / topic
    ],
    "change": [
        {"audio_emb"},            # speaker identification
        {"text_emb", "keyword"},  # content / topic
    ],
    "locate": [
        {"audio_emb"},            # speaker identification
        {"domain_emb"},           # emotion reference matching
        {"label"},                # emotion label search
    ],
    "locate_hard": [
        {"domain_emb"},           # emotion reference matching
        {"label"},                # emotion label search
    ],
    # --- music (SongEval) ---
    "audio": [
        {"audio_emb", "domain_emb"}, # music reference matching
    ],
    # "time" and "position" types use only localize; no tool requirement
}


# Public scoring functions (each returns int 0/1 or float)

_KNOWN_JUDGE_TYPES = {"time_and_quote", "speaker_id", "trend", "categorical", None}


def _parse_yn(text: str) -> int:
    """Parse a YES/NO verdict from judge output.

    Priority:
    1. Explicit "Answer: YES/NO" or "Final answer: YES/NO" label anywhere.
    2. A line whose entire content is YES or NO (standalone decision), searching
       from the end of the text (handles "YES\n\n...reason...\nNO").
    3. First standalone YES/NO token (original behaviour; avoids picking up
       YES/NO mentions embedded in reasoning prose near the end).
    """
    explicit = re.search(
        r'\b(?:Final\s+)?Answer\s*:\s*\**\s*(YES|NO)\b', text, re.IGNORECASE
    )
    if explicit:
        return 1 if explicit.group(1).upper() == "YES" else 0
    for line in reversed(text.strip().splitlines()):
        stripped = re.sub(r'[\s`*_]', '', line)
        if re.fullmatch(r'(YES|NO)', stripped, re.IGNORECASE):
            return 1 if stripped.upper() == "YES" else 0
    tokens = re.findall(r'\b(YES|NO)\b', text.upper())
    return (1 if tokens[0] == "YES" else 0) if tokens else 0


def score_answer(
    prediction: str, query_item: Dict[str, Any], judge: "JudgeBase"
) -> int:
    expected = _get_expected(query_item)
    if expected is None:
        return 0
    judge_info = query_item.get("judge_info", {}) or {}
    ji_type = judge_info.get("type")
    if ji_type not in _KNOWN_JUDGE_TYPES:
        import warnings
        warnings.warn(
            f"Unknown judge_info type {ji_type!r} for query "
            f"{query_item.get('query_id')} ({query_item.get('query_type')}); "
            "falling back to generic answer prompt.",
            stacklevel=2,
        )
    if ji_type == "time_and_quote":
        prompt = _locate_answer_prompt(prediction, query_item, judge_info)
    elif ji_type == "categorical":
        prompt = _categorical_answer_prompt(prediction, query_item, judge_info)
    elif ji_type == "trend":
        prompt = _trend_answer_prompt(prediction, query_item, judge_info)
    else:
        prompt = _generic_answer_prompt(prediction, query_item, judge_info)
    verdict = judge(prompt).strip()
    return _parse_yn(verdict)


_EMO_NAMES: Dict[str, str] = {
    "H": "Happy", "S": "Sad", "A": "Angry", "N": "Neutral",
    "D": "Disgust", "F": "Fear", "C": "Contempt", "U": "Surprise",
}


def _locate_answer_prompt(
    prediction: str,
    query_item: Dict[str, Any],
    judge_info: Dict[str, Any],
) -> str:
    query_text = query_item.get("query", "")
    query_type = query_item.get("query_type", "")
    emo = judge_info.get("emo", "")
    emo_name = _EMO_NAMES.get(str(emo).upper(), emo)
    t0 = judge_info.get("t0")
    t1 = judge_info.get("t1")
    tol = 3  # fixed ±3 s on both bounds
    seg_text = judge_info.get("seg_text", "")
    lo = round(t0 - tol, 1) if t0 is not None else "?"
    hi = round(t1 + tol, 1) if t1 is not None else "?"
    return (
        f"Query ({query_type}): {query_text}\n"
        f"Model answer: {prediction}\n\n"
        f"Ground truth:\n"
        f"  Emotion: {emo_name} (code {emo})\n"
        f"  GT time window: {t0}s–{t1}s  |  accept window: {lo}s–{hi}s\n"
        f"  Reference text: \"{seg_text}\"\n\n"
        "Note: 'between X and Y seconds' in the query is the search window; "
        "the model should name any specific timestamp strictly inside [X, Y] "
        "— any such time is valid, not just times at the boundary values.\n\n"
        "Accept (YES) if the model answer satisfies BOTH:\n"
        "1. Identifies the correct emotion — accept the exact label, full "
        "name, code, common synonym, or any expression that clearly refers "
        "to the same emotional state (e.g. 'joyful' for Happy, 'upset' for "
        "Angry/Sad). Reject only if the stated emotion is clearly different.\n"
        "2. Reports a time that falls within the accept window above.\n"
        "The exact quote does not need to match.\n"
        "Reply with only YES or NO."
    )


def _generic_answer_prompt(
    prediction: str,
    query_item: Dict[str, Any],
    judge_info: Dict[str, Any],
) -> str:
    query_text = query_item.get("query", "")
    query_type = query_item.get("query_type", "")
    expected_str = _expand_answer(_get_expected(query_item))
    ab_line = _ab_label_line(query_item)
    # Explicitly map Set N → A/B so the judge handles "Set 2 has better" etc.
    set_note = (
        "Note: Set 1 in the retrieved evidence = A (first-mentioned item),"
        " Set 2 = B (second-mentioned), etc.\n"
        if ab_line else ""
    )
    return (
        f"Query ({query_type}): {query_text}\n"
        f"{ab_line}"
        f"{set_note}"
        f"Ground truth answer: {expected_str}\n"
        f"Ground truth metadata: {judge_info}\n"
        f"Model answer: {prediction}\n\n"
        "Does the model answer match the ground truth? "
        "The model may answer with A/B, Set N, or name the item directly. "
        "If the ground truth is 'tie', accept the model answer if it "
        "says tie, or only claims one is marginally better (not "
        "clearly or significantly better). "
        "Reply with only YES or NO."
    )


def _categorical_answer_prompt(
    prediction: str,
    query_item: Dict[str, Any],
    judge_info: Dict[str, Any],
) -> str:
    query_text = query_item.get("query", "")
    query_type = query_item.get("query_type", "")
    primary = judge_info.get("primary", "")
    primary_name = _EMO_NAMES.get(str(primary).upper(), primary)
    secondary = judge_info.get("secondary") or {}
    secondary_names = {_EMO_NAMES.get(str(k).upper(), k): v for k, v in secondary.items()}
    cf = judge_info.get("cf", "")
    votes = judge_info.get("votes") or {}
    votes_named = {_EMO_NAMES.get(str(k).upper(), k): v for k, v in votes.items()}
    return (
        f"Query ({query_type}): {query_text}\n"
        f"Model answer: {prediction}\n\n"
        f"Ground truth:\n"
        f"  Primary emotion: {primary_name} (code {primary})"
        f", confidence={cf}, votes={votes_named}\n"
        f"  Secondary emotions (minority annotators): {secondary_names}\n\n"
        "Accept (YES) if the model's answer includes the primary emotion "
        "(by full name or code), even if it also lists secondary or other "
        "emotions alongside it. Reject (NO) only if the primary emotion is "
        "absent or clearly contradicted.\n"
        "Reply with only YES or NO."
    )


def _trend_answer_prompt(
    prediction: str,
    query_item: Dict[str, Any],
    judge_info: Dict[str, Any],
) -> str:
    query_text = query_item.get("query", "")
    query_type = query_item.get("query_type", "")
    direction = judge_info.get("direction", "")
    half1 = judge_info.get("half1")
    half2 = judge_info.get("half2")
    _DIR = {"up": "increasing / rising", "down": "decreasing / falling",
            "none": "stable / no significant change"}
    direction_desc = _DIR.get(str(direction).lower(), direction)
    return (
        f"Query ({query_type}): {query_text}\n"
        f"Model answer: {prediction}\n\n"
        f"Ground truth:\n"
        f"  Trend direction: {direction_desc} (code: {direction!r})\n"
        f"  First-half mean: {half1}, second-half mean: {half2}\n\n"
        "Does the net movement the model reports agree with the GT direction? "
        "Compare the actual values in the model's answer — not just the "
        "direction label it states. Values may be on different scales. "
        "Reply with only YES or NO."
    )


def score_rationale(
    prediction: str,
    rationale: str,
    query_item: Dict[str, Any],
    gt_segs: List[Dict[str, Any]],
    ct_info: Dict[str, Any],
    evidence_sets: List[List[Dict[str, Any]]],
    judge: "JudgeBase",
    schema_desc: str = "",
    fmt_segs=None,
    skip_faithful: bool = False,
) -> Dict[str, int]:
    """Returns factual/faithful (each 0/1) and rationale = AND."""
    factual = _score_factual(
        prediction, rationale, query_item, judge, schema_desc,
    )
    faithful = (
        1 if skip_faithful
        else _score_faithful(prediction, rationale, query_item, evidence_sets, judge)
    )
    return {
        "factual": factual,
        "faithful": faithful,
        "rationale": factual & faithful,
    }


def score_retrieval(
    evidence_sets: List[List[Dict[str, Any]]],
    query_item: Dict[str, Any],
    gt_segs: List[Dict[str, Any]],
) -> float:
    """Fraction of GT evidence segments covered by any retrieved evidence.

    Evidence segments use time=[start,end]; match by time overlap.
    """
    gt_ev = get_gt_evidence_segs(query_item, gt_segs)
    if not gt_ev:
        return 0.0
    # flatten all retrieved time windows
    ret_windows = [
        (float(s["time"][0]), float(s["time"][1]))
        for ev in evidence_sets
        for s in ev
        if isinstance(s.get("time"), (list, tuple)) and len(s["time"]) == 2
    ]
    if not ret_windows:
        return 0.0
    hits = sum(
        1 for seg in gt_ev
        if any(
            float(seg["start"]) < hi and float(seg["end"]) > lo
            for lo, hi in ret_windows
        )
    )
    return hits / len(gt_ev)


def score_plan(
    plan: Dict[str, Any],
    query_item: Dict[str, Any],
    gt_segs: List[Dict[str, Any]],
) -> float:
    """Plan hit rate: (time_coverage + Σ tool_group_covered) / (1 + n_groups).

    time_coverage: fraction of GT evidence segs covered by localize windows.
    Each required tool group contributes 1 if any tool in the group is active.
    For types with no required tools, returns time_coverage alone.
    """
    if plan.get("_oracle"):
        return 1.0

    gt_ev = get_gt_evidence_segs(query_item, gt_segs)

    # time coverage (fraction of GT evidence segs inside any localize window)
    windows: List[tuple] = []
    for s in plan.get("sets", []):
        try:
            lo, hi = parse_localize(s["localize"], segments=gt_segs)
            windows.append((float(lo), float(hi)))
        except (KeyError, ValueError, TypeError):
            continue
    if gt_ev and windows:
        hits = sum(
            1 for seg in gt_ev
            if any(
                float(seg["start"]) < hi and float(seg["end"]) > lo
                for lo, hi in windows
            )
        )
        time_cov = hits / len(gt_ev)
    else:
        time_cov = 0.0

    # tool group coverage
    required = _REQUIRED_TOOLS.get(query_item.get("query_type", ""))
    if not required:
        return time_cov
    active: set = set()
    for s in plan.get("sets", []):
        for k, v in s.items():
            if k != "localize" and v:
                active.add(k)
    tool_hits = sum(1 for grp in required if grp & active)

    return (time_cov + tool_hits) / (1 + len(required))


def score_all(
    prediction: str,
    rationale: str,
    query_item: Dict[str, Any],
    gt_segs: List[Dict[str, Any]],
    ct_info: Dict[str, Any],
    evidence_sets: List[List[Dict[str, Any]]],
    plan: Dict[str, Any],
    judge: "JudgeBase",
    schema_desc: str = "",
    fmt_segs=None,
    skip_faithful: bool = False,
    rationale_only: bool = False,
) -> Dict[str, Any]:
    """Run all scores and return them as a flat dict."""
    rat = score_rationale(
        prediction, rationale, query_item, gt_segs, ct_info,
        evidence_sets, judge, schema_desc, fmt_segs, skip_faithful,
    )
    if rationale_only:
        return rat
    return {
        "answer": score_answer(prediction, query_item, judge),
        **rat,
        "retrieval_hit_rate": score_retrieval(evidence_sets, query_item, gt_segs),
        "plan_hit_rate": score_plan(plan, query_item, gt_segs),
    }


def get_gt_evidence_segs(
    query_item: Dict[str, Any],
    gt_segs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """GT segments relevant to this query (explicit list or type-specific lookup)."""
    gt_ids = query_item.get("gt_seg_ids")
    if gt_ids:
        id_set = set(gt_ids)
        return [s for s in gt_segs if s.get("seg_id") in id_set]

    ji = query_item.get("judge_info") or {}
    ji_type = ji.get("type")
    segs_sorted = sorted(gt_segs, key=lambda s: float(s["start"]))

    if ji_type in ("categorical", "time_and_quote"):
        # state / locate / locate_hard: single GT seg at the labeled event.
        # Use closest start to t0 (not first by start) to avoid picking the
        # seg that ends AT t0 over the one that starts AT t0.
        t = float(ji.get("t0") or query_item.get("time_ref") or 0)
        candidates = [
            s for s in segs_sorted
            if float(s["start"]) <= t + 1 and float(s["end"]) >= t - 1
        ]
        target = (
            min(candidates, key=lambda s: abs(float(s["start"]) - t))
            if candidates
            else (min(segs_sorted, key=lambda s: abs(float(s["start"]) - t))
                  if segs_sorted else None)
        )
        return [target] if target else []

    if ji_type == "trend":
        # change: [time_ref, time_ref+30] is the real window;
        # query text widens it with ~240 s of padding as distractor context
        t = float(query_item.get("time_ref") or 0)
        t0, t1 = t, t + 30
        segs = [s for s in gt_segs
                if float(s["start"]) < t1 + 1
                and float(s["end"]) > t0 - 1]
        if segs:
            return segs

    if ji_type == "speaker_id":
        # comparison: actual candidate window is [time_ref, time_ref+60]
        t = float(query_item.get("time_ref") or 0)
        segs = gtr.filter_segments_window(gt_segs, t, t + 60)
        if segs:
            return segs

    # music/comparison/unknown: per-block time_ref window or position lookup
    time_refs: List[float] = []
    time_ref = query_item.get("time_ref")
    if time_ref is not None:
        time_refs.append(float(time_ref))
    for key in ("song_a", "song_b", "speaker_a", "speaker_b"):
        block = query_item.get(key)
        if isinstance(block, dict) and "time_ref" in block:
            time_refs.append(float(block["time_ref"]))

    if time_refs:
        seen: set = set()
        result: List[Dict[str, Any]] = []
        for t in time_refs:
            for seg in gtr.filter_segments_window(
                gt_segs, max(0.0, t - _GT_WINDOW_SEC), t + _GT_WINDOW_SEC
            ):
                if seg["seg_id"] not in seen:
                    seen.add(seg["seg_id"])
                    result.append(seg)
        return result

    pos_segs: List[Dict[str, Any]] = []
    for key in ("song_a", "song_b", "speaker_a", "speaker_b"):
        block = query_item.get(key)
        if not isinstance(block, dict) or "position" not in block:
            continue
        tag = f"SONG_{int(block['position']) - 1:02d}"
        pos_segs.extend(s for s in gt_segs if s.get("speaker") == tag)
    if pos_segs:
        return pos_segs

    return gt_segs


# Internal helpers

def _retrieve_gt_from_evidence(
    evidence_sets: List[List[Dict[str, Any]]],
    gt_segs: List[Dict[str, Any]],
    ct_info: Dict[str, Any],
    fmt_segs,
) -> str:
    """GT segments and CT traces matching every time window in evidence_sets."""
    seen: set = set()
    matched: List[Dict[str, Any]] = []
    ct_parts: List[str] = []
    for ev in evidence_sets:
        for s in ev:
            t = s.get("time")
            if not (isinstance(t, (list, tuple)) and len(t) == 2):
                continue
            lo, hi = float(t[0]), float(t[1])
            for seg in gtr.filter_segments_window(gt_segs, lo, hi):
                if seg["seg_id"] not in seen:
                    seen.add(seg["seg_id"])
                    matched.append(seg)
            for dim, traces in ct_info.items():
                summary = _summarize_ct(traces, lo, hi)
                if summary:
                    ct_parts.append(f"CT {dim} [{lo:.1f}s–{hi:.1f}s]: {summary}")
    parts: List[str] = []
    if matched:
        parts.append(fmt_segs(matched))
    parts.extend(ct_parts)
    return "\n".join(parts) if parts else "(no matching GT segments found)"


def _score_factual(
    prediction: str,
    rationale: str,
    query_item: Dict[str, Any],
    judge: "JudgeBase",
    schema_desc: str = "",
) -> int:
    """Checks answer/rationale against the structured GT label (judge_info)."""
    header = f"{schema_desc}\n\n" if schema_desc else ""
    query_text = query_item.get("query", "")
    query_type = query_item.get("query_type", "")
    gt_summary = _format_judge_info(query_item)
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
    prediction: str,
    rationale: str,
    query_item: Dict[str, Any],
    evidence_sets: List[List[Dict[str, Any]]],
    judge: "JudgeBase",
) -> int:
    """Checks rationale is grounded in the retrieved evidence."""
    query_text = query_item.get("query", "")
    query_type = query_item.get("query_type", "")
    retrieved = _format_evidence_sets(evidence_sets)
    prompt = (
        f"Query ({query_type}): {query_text}\n\n"
        f"## Retrieved evidence\n{retrieved}\n\n"
        f"Model answer: {prediction}\n"
        f"Model rationale: {rationale}\n\n"
        "Is the rationale faithful to the retrieved evidence above? "
        "Reply YES if all claims in the rationale are grounded in the evidence, "
        "NO if the rationale introduces facts not found there. "
        "If NO, add one short sentence explaining why."
    )
    verdict = judge(prompt).strip()
    return _parse_yn(verdict)


def _retrieve_gt_side(
    query_item: Dict[str, Any],
    gt_segs: List[Dict[str, Any]],
    ct_info: Dict[str, Any],
    fmt_segs,
) -> str:
    parts: List[str] = []
    gt_block = query_item.get("ground_truth") or {}

    # quote field (emotion locate queries use "quote"; others may skip)
    quote = gt_block.get("quote") or gt_block.get("actual_quote") or ""
    if quote:
        hits = gtr.search_quote(gt_segs, quote)
        if hits:
            parts.append(f"### quote: '{quote}'\n" + fmt_segs(hits))

    # collect explicit time references from multiple sources:
    # 1. t0/t1 in ground_truth block
    # 2. top-level time_ref
    # 3. multi-item references: song_a/song_b, speaker_a/speaker_b (music/comparison)
    time_points: List[tuple] = []  # (label, center_or_None, lo_or_None, hi_or_None)

    t0 = gt_block.get("t0")
    t1 = gt_block.get("t1")
    if t0 is not None and t1 is not None:
        time_points.append(("window", None, float(t0), float(t1)))
    else:
        time_ref = query_item.get("time_ref")
        if time_ref is not None:
            time_points.append(("center", float(time_ref), None, None))

    # multi-song / multi-speaker fields (music comparison queries)
    for key in ("song_a", "song_b", "speaker_a", "speaker_b"):
        block = query_item.get(key) or {}
        if isinstance(block, dict) and "time_ref" in block:
            time_points.append((key, float(block["time_ref"]), None, None))

    for label, center, lo, hi in time_points:
        if lo is not None and hi is not None:
            window = gtr.filter_segments_window(gt_segs, lo, hi)
            tag = f"{lo:.1f}s–{hi:.1f}s"
        else:
            lo2, hi2 = max(0.0, center - 30), center + 30
            window = gtr.filter_segments_window(gt_segs, lo2, hi2)
            tag = f"around {center:.1f}s"
        if window:
            parts.append(f"### [{label}] {tag}\n" + fmt_segs(window))
        for dim, traces in ct_info.items():
            lw = lo if lo is not None else max(0.0, center - 30)
            hw = hi if hi is not None else center + 30
            summary = _summarize_ct(traces, lw, hw)
            if summary:
                parts.append(f"CT {dim} [{label}] {tag}: {summary}")

    return "\n\n".join(parts) if parts else "(no GT-side evidence found)"


def _summarize_ct(
    traces: List[Dict[str, Any]], start: float, end: float
) -> str:
    wt = [t for t in traces if t["end"] > start and t["start"] < end]
    if not wt:
        return ""
    vals = [t["mean"] for t in wt]
    mean = sum(vals) / len(vals)
    lo, hi = min(vals), max(vals)
    mid = len(vals) // 2
    if mid > 0:
        first_half = sum(vals[:mid]) / mid
        second_half = sum(vals[mid:]) / (len(vals) - mid)
        diff = second_half - first_half
        trend = (
            "rising" if diff > 0.05
            else "falling" if diff < -0.05
            else "stable"
        )
    else:
        trend = "stable"
    return f"mean={mean:.2f}, range=[{lo:.2f},{hi:.2f}], trend={trend}"


def _format_judge_info(query_item: Dict[str, Any]) -> str:
    """Render judge_info as a structured GT summary for the factual prompt."""
    ji = query_item.get("judge_info") or {}
    gt = query_item.get("ground_truth") or {}
    ji_type = ji.get("type")

    if ji_type == "speaker_id":
        dim = ji.get("dim", "dim")
        spk_val = ji.get("spk_val") or {}
        winner = ji.get("winner", "?")
        vals = "  ".join(f"Speaker {k} {dim}={v:.3f}" for k, v in spk_val.items())
        return f"GT: {vals}  →  winner = Speaker {winner}"

    if ji_type == "categorical":
        primary = ji.get("primary", "?")
        primary_name = _EMO_NAMES.get(str(primary).upper(), primary)
        cf = ji.get("cf", "")
        votes = ji.get("votes") or {}
        votes_named = {_EMO_NAMES.get(str(k).upper(), k): v for k, v in votes.items()}
        secondary = ji.get("secondary") or {}
        secondary_named = {_EMO_NAMES.get(str(k).upper(), k): v
                           for k, v in secondary.items()}
        lines = [f"GT primary emotion: {primary_name} ({primary}), confidence={cf}"]
        if votes_named:
            lines.append(f"  annotator votes: {votes_named}")
        if secondary_named:
            lines.append(f"  secondary (minority): {secondary_named}")
        return "\n".join(lines)

    if ji_type == "trend":
        direction = ji.get("direction", "?")
        _DIR = {"up": "rising", "down": "falling", "none": "stable"}
        half1 = ji.get("half1")
        half2 = ji.get("half2")
        desc = _DIR.get(str(direction).lower(), direction)
        line = f"GT trend: {desc} ({direction!r})"
        if half1 is not None and half2 is not None:
            line += f"  |  first-half mean={half1:.3f}, second-half mean={half2:.3f}"
        return line

    if ji_type == "time_and_quote":
        emo = ji.get("emo", "?")
        emo_name = _EMO_NAMES.get(str(emo).upper(), emo)
        t0 = ji.get("t0")
        t1 = ji.get("t1")
        seg_text = ji.get("seg_text", "")
        line = f"GT emotion: {emo_name} ({emo})  at [{t0}s – {t1}s]"
        if seg_text:
            line += f'\n  reference text: "{seg_text}"'
        return line

    # music comparison: gt has mean_a/mean_b/delta/winner; ji has per-dim scores
    if gt.get("mean_a") is not None or gt.get("mean_b") is not None:
        def _qlabel(v):
            if v is None: return "?"
            return "low" if v < 2 else "below-avg" if v < 3 else "mid" if v < 4 else "high"
        mean_a = gt.get("mean_a")
        mean_b = gt.get("mean_b")
        winner = gt.get("winner", "?")
        delta  = gt.get("delta")
        la, lb = _qlabel(mean_a), _qlabel(mean_b)
        lines = [(f"GT: Song A mean={mean_a} ({la})  Song B mean={mean_b} ({lb})"
                  f"  →  winner = Song {winner}"
                  + (f"  (delta={delta:.3f})" if delta is not None else ""))]
        for label, key in (("Song A", "scores_a"), ("Song B", "scores_b")):
            scores = ji.get(key) or {}
            if scores:
                dim_str = "  ".join(f"{k}={v}" for k, v in scores.items())
                lines.append(f"  {label} dims: {dim_str}")
        return "\n".join(lines)

    # generic fallback
    parts = [f"{k}={v}" for k, v in ji.items() if v is not None and k != "type"]
    prefix = f"GT ({ji_type}): " if ji_type else "GT: "
    return prefix + ", ".join(parts) if parts else str(ji)


def _expand_answer(expected: Any) -> str:
    _MAP = {"a": "A", "b": "B"}
    return _MAP.get(str(expected).strip().lower(), str(expected))


def _ab_label_line(query_item: Dict[str, Any]) -> str:
    """Return 'A = <desc>, B = <desc>\n' if the query has A/B items."""
    labels = []
    for letter, key in (("A", "song_a"), ("B", "song_b"),
                        ("A", "speaker_a"), ("B", "speaker_b")):
        block = query_item.get(key)
        if not isinstance(block, dict):
            continue
        if "position" in block:
            labels.append(f"{letter} = song #{block['position']}")
        elif "time_ref" in block:
            t = block["time_ref"]
            labels.append(f"{letter} = song/speaker around {t}s")
        elif "name" in block:
            labels.append(f"{letter} = {block['name']}")
    return (", ".join(labels) + "\n") if labels else ""


def _get_expected(gt: Dict[str, Any]) -> Any:
    # explicit answer field wins
    if "answer" in gt:
        return gt["answer"]
    ji = gt.get("judge_info") or {}
    if isinstance(ji, dict):
        if "answer" in ji:
            return ji["answer"]
        # change queries store GT as direction in judge_info
        if "direction" in ji:
            return ji["direction"]
        # state queries store GT as primary emotion code in judge_info
        if "primary" in ji:
            return ji["primary"]
    gtr_block = gt.get("ground_truth") or {}
    if isinstance(gtr_block, dict):
        # change queries may also store direction in ground_truth
        for key in ("direction", "answer", "winner", "emo"):
            if key in gtr_block:
                return gtr_block[key]
    dl = gt.get("domain_labels") or {}
    return dl.get("primary")
