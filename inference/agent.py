from __future__ import annotations

import base64
import io
import json
import math
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

import soundfile as sf

from inference import tools
from inference.format import format_evidence_sets
from inference.preprocess import preprocess_clips

if TYPE_CHECKING:
    from extract.audio_expert.base import EmbBase
    from extract.audio_expert.asr.whisper import WhisperASR
    from extract.domain_expert.base import DomainLabelerBase
    from inference.llm.base import LLMBase

# All modalities available; config may restrict to a subset.
ALL_MODALITIES: frozenset[str] = frozenset(
    {"time", "keyword", "label", "text_emb", "audio_emb", "domain_emb", "audio"}
)


class HARPAgent:
    """Plan → dispatch tools → answer.

    Args:
        llm:          LLMBase for planning and answering.
        segments_dir: Directory with {episode_id}.jsonl files.
        faiss_dir:    Directory with FAISS index + meta files.
        audio_dir:    Directory with episode audio files.
        prompts:      Module with TASK_DESC, SCHEMA_DESC, TOOL_DESCS,
                      PLAN_PROMPT, ANSWER_PROMPT.
        modalities:   Enabled tool names; defaults to all.
        audio_emb:  EmbBase for audio_emb search.
        domain_emb:   EmbBase for domain_emb search.
        top_k:        Max evidence segments per set.
    """

    def __init__(
        self,
        llm: LLMBase,
        segments_dir: Path,
        faiss_dir: Path,
        audio_dir: Path,
        prompts: Any,
        modalities: list[str] | None = None,
        text_emb: EmbBase | None = None,
        audio_emb: EmbBase | None = None,
        domain_emb: EmbBase | None = None,
        top_k: int = 10,
        max_audio_clip_sec: int = 120,
        strip_domain: bool = False,
        use_vec_cache: bool = False,
        audio_vec_cache: dict | None = None,
        domain_vec_cache: dict | None = None,
        # baseline preprocessing
        task: str = "",
        asr: WhisperASR | None = None,
        domain_labeler: DomainLabelerBase | None = None,
        label_cache: dict | None = None,
        clip_audio_dir: Path | None = None,
    ) -> None:
        self._llm = llm
        self._seg_dir = Path(segments_dir)
        self._audio_dir = Path(audio_dir)
        self._max_audio_clip_sec = max_audio_clip_sec
        self._strip_domain = strip_domain
        faiss_dir = Path(faiss_dir)
        self._prompts = prompts
        self._modalities: frozenset[str] = (
            frozenset(modalities) if modalities is not None else ALL_MODALITIES
        )
        self._text_emb = text_emb
        self._audio_emb = audio_emb
        self._domain_emb = domain_emb
        self._top_k = top_k
        self._use_vec_cache = use_vec_cache
        self._audio_vec_cache = audio_vec_cache
        self._domain_vec_cache = domain_vec_cache
        self._task = task
        self._asr = asr
        self._domain_labeler = domain_labeler
        self._label_cache = label_cache
        self._clip_audio_dir = Path(clip_audio_dir) if clip_audio_dir else None
        # load FAISS indices once; None if index files not yet created
        self._stores = {
            "text":    tools.load_faiss_store(faiss_dir / "text.index",    faiss_dir / "text_meta.jsonl"),
            "audio":   tools.load_faiss_store(faiss_dir / "audio.index",   faiss_dir / "audio_meta.jsonl"),
            "domain":  tools.load_faiss_store(faiss_dir / "domain_0.index",  faiss_dir / "domain_0_meta.jsonl"),
        }

    def __call__(self, query_item: dict[str, Any]) -> dict[str, Any]:
        query = str(query_item["query"])
        episode_id = str(query_item["episode_id"])

        segments = tools.load_segments(episode_id, self._seg_dir)
        seg_map = {s["seg_id"]: s for s in segments}

        plan = self._plan(query_item, episode_id)
        evidence_sets, audio_clips = self._dispatch(
            plan, segments, seg_map, episode_id, query_item
        )
        ref_clips = _load_ref_clips(query_item)
        prediction, rationale = self._answer(
            query, evidence_sets, ref_clips + audio_clips, plan=plan,
        )

        return {
            "query_id": query_item.get("query_id"),
            "episode_id": episode_id,
            "prediction": prediction,
            "rationale": rationale,
            "plan": plan,
            "evidence_sets": evidence_sets,
        }

    def _plan(self, query_item: dict[str, Any], episode_id: str = "") -> dict[str, Any]:
        wav_map = query_item.get("speaker_wav_map") or {}
        speaker_examples = ", ".join(sorted(wav_map)) if wav_map else "none"
        tool_descs = self._prompts.TOOL_DESCS
        _BOOL_FIELDS = {"time"}
        schema_fields: dict[str, Any] = {"localize": "..."}
        tools_lines: list[str] = []
        # "audio" is handled post-retrieval, not by the planner
        for name in ("time", "keyword", "label", "text_emb", "audio_emb", "domain_emb"):
            if name not in self._modalities:
                continue
            if name in tool_descs:
                tools_lines.append(
                    "  " + tool_descs[name].format(
                        audio_examples=speaker_examples,
                        speaker_examples=speaker_examples,
                    )
                )
            schema_fields[name] = False if name in _BOOL_FIELDS else None
        tools_desc = "\n".join(tools_lines)
        schema_example = json.dumps({"sets": [schema_fields]})
        clip_info = ""
        if self._clip_audio_dir and (
            self._asr or self._domain_labeler or self._label_cache
        ):
            result = preprocess_clips(
                query_item, self._task, self._clip_audio_dir,
                asr=self._asr, domain_labeler=self._domain_labeler,
                label_cache=self._label_cache,
            )
            if result:
                clip_info = f"\nReference clips (preprocessed):\n{result}\n"
        track_info = _get_track_info(episode_id, self._audio_dir)
        prompt = self._prompts.PLAN_PROMPT.format(
            task_desc=self._prompts.TASK_DESC,
            schema_desc=self._prompts.SCHEMA_DESC,
            tools_desc=tools_desc,
            clip_info=clip_info,
            track_info=track_info,
            schema_example=schema_example,
            query=query_item["query"],
        )
        raw = self._llm(prompt, max_tokens=512)
        return _parse_json(raw) or {"sets": []}

    def _dispatch(
        self,
        plan: dict[str, Any],
        segments: list[dict[str, Any]],
        seg_map: dict[str, dict[str, Any]],
        episode_id: str,
        query_item: dict[str, Any],
    ) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
        evidence_sets: list[list[dict]] = []
        audio_clips: list[dict[str, Any]] = []
        _wave_cache: dict = {}  # avoids reloading the same track per query

        for s in plan.get("sets", []):
            try:
                window = tools.localize(s["localize"], segments=segments)
            except (KeyError, ValueError):
                continue

            hits: list[list[dict]] = []

            if "time" in self._modalities:
                hits.append(tools.search_by_time(segments, window))

            if "label" in self._modalities and (lbl := s.get("label")):
                hits.append(tools.search_by_label(segments, lbl, window))

            if "text_emb" in self._modalities and (te := s.get("text_emb")):
                hits.append(tools.search_text_emb(
                    te, self._text_emb, self._stores["text"],
                    seg_map, episode_id, window,
                ))

            if "keyword" in self._modalities and (kw := s.get("keyword")):
                hits.append(tools.search_keyword(segments, kw, window))

            if "audio_emb" in self._modalities and (spk_ref := s.get("audio_emb")):
                wav_path = _resolve_wav(spk_ref, query_item)
                if wav_path:
                    hits.append(tools.search_audio_emb(
                        wav_path, self._audio_emb, self._stores["audio"],
                        seg_map, episode_id, window,
                        use_vec_cache=self._use_vec_cache,
                        vec_cache=self._audio_vec_cache,
                    ))

            if "domain_emb" in self._modalities and (dom_ref := s.get("domain_emb")):
                wav_path = _resolve_wav(dom_ref, query_item)
                if wav_path:
                    hits.append(tools.search_domain_emb(
                        wav_path, self._domain_emb, self._stores["domain"],
                        seg_map, episode_id, window,
                        use_vec_cache=self._use_vec_cache,
                        vec_cache=self._domain_vec_cache,
                    ))

            ev = tools.gather_evidence(hits, top_k=self._top_k)
            evidence_sets.append(ev)

            # load audio for this evidence set's span (post-retrieval)
            if "audio" in self._modalities:
                if ev:
                    lo = int(min(float(seg["time"][0]) for seg in ev))
                    hi = min(
                        int(max(float(seg["time"][1]) for seg in ev)),
                        lo + self._max_audio_clip_sec,
                    )
                else:
                    lo = int(window[0])
                    hi = min(int(window[1]), lo + self._max_audio_clip_sec)
                result = tools.search_audio(
                    episode_id, (lo, hi), self._audio_dir, _cache=_wave_cache
                )
                if result is not None:
                    wave, sr = result
                    set_n = len(evidence_sets)  # 1-based after append
                    audio_clips.append({
                        "label": f"[Set {set_n}: {lo}-{hi}s]",
                        "window": (lo, hi), "audio": wave, "sr": sr,
                    })

        return evidence_sets, audio_clips

    def _answer(
        self,
        query: str,
        evidence_sets: list[list[dict[str, Any]]],
        audio_clips: list[dict[str, Any]],
        plan: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        context = format_evidence_sets(evidence_sets, strip_domain=self._strip_domain)
        if plan is not None:
            guide = _set_guide(plan)
            if guide:
                context = guide + "\n\n" + context
        llm_clips = [_encode_clip(c) for c in audio_clips] or None
        if llm_clips:
            labels = ", ".join(c["label"] for c in llm_clips)
            context += (
                f"\n\nAudio clips are provided as input above ({labels})."
                " Clips labeled [Set N: ...] correspond to the evidence set"
                " with the same number above."
                " Use what you hear to support or refine your answer."
            )
        prompt = self._prompts.ANSWER_PROMPT.format(
            task_desc=self._prompts.TASK_DESC,
            query=query, context=context,
        )
        raw = self._llm(prompt, audio_clips=llm_clips)
        return _parse_answer(raw)


class OracleAgent(HARPAgent):
    """Skips LLM planning; builds the plan directly from GT segment windows.

    _dispatch runs unchanged: search_by_time retrieves metadata and
    search_audio loads clips, both gated by the configured modalities.
    Use strip_domain: true in config to pass only audio to the answer step.

    Args:
        gt_dir: base GT directory; segments are read from gt_dir/segments/.
    """

    def __init__(
        self, *args: Any,
        gt_dir: Path | None = None,
        spk_wav_to_id: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._gt_seg_dir = Path(gt_dir) / "segments" if gt_dir else None
        self._spk_wav_to_id: dict = spk_wav_to_id or {}

    def __call__(self, query_item: dict[str, Any]) -> dict[str, Any]:
        query = str(query_item["query"])
        episode_id = str(query_item["episode_id"])

        segments = tools.load_segments(episode_id, self._seg_dir)
        seg_map = {s["seg_id"]: s for s in segments}

        gt_segs = (
            tools.load_segments(episode_id, self._gt_seg_dir)
            if self._gt_seg_dir else []
        )
        plan = self._gt_plan(query_item, gt_segs, segments)
        evidence_sets, audio_clips = self._dispatch(
            plan, segments, seg_map, episode_id, query_item
        )
        ref_clips = _load_ref_clips(query_item)
        prediction, rationale = self._answer(
            query, evidence_sets, ref_clips + audio_clips, plan=plan,
        )

        return {
            "query_id": query_item.get("query_id"),
            "episode_id": episode_id,
            "prediction": prediction,
            "rationale": rationale,
            "plan": plan,
            "evidence_sets": evidence_sets,
        }

    def _gt_plan(
        self, query_item: dict[str, Any], gt_segs: list[dict[str, Any]],
        segments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self._task == "health":
            return {
                "_oracle": True,
                "sets": [{"localize": f"0 sec to {self._max_audio_clip_sec} sec",
                           "time": True}]
            }

        ji = (query_item.get("judge_info") or {})
        ji_type = ji.get("type")
        segs_sorted = sorted(gt_segs, key=lambda s: float(s["start"]))

        if ji_type in ("categorical", "time_and_quote"):
            # state / locate / locate_hard: find the extracted segment at time_ref
            # so search_by_time returns exactly that segment
            t = float(ji.get("t0") or query_item.get("time_ref") or 0)
            ext_sorted = sorted(segments or [], key=lambda s: float(s["start"]))
            seg = _seg_at(ext_sorted, t) if ext_sorted else _seg_at(segs_sorted, t)
            sets = (
                [{"localize": f"{math.floor(float(seg['start']))} sec"
                  f" to {math.ceil(float(seg['end']))} sec", "time": True}]
                if seg else []
            )

        elif ji_type == "trend":
            # change: [time_ref, time_ref+30] is the real window;
            # query text widens it with ~240 s of padding as distractor context
            t = float(query_item.get("time_ref") or 0)
            t0, t1 = int(t), int(t + 30)
            sets = [{"localize": f"{t0} sec to {t1} sec", "time": True}]

        elif ji_type == "speaker_id":
            # comparison: actual candidate window is [time_ref, time_ref+60]
            # (query text adds ~240 s of padding as distractor context)
            t = float(query_item.get("time_ref") or 0)
            t0, t1 = int(t), int(t + 60)
            sets = []
            for spk_key in ("audio_spk", "audio_spk2"):
                wav = query_item.get(spk_key, "")
                spk_id = self._spk_wav_to_id.get(Path(wav).name)
                if spk_id is not None:
                    spk_segs = [
                        s for s in segs_sorted
                        if (s.get("speaker") == spk_id
                            or s.get("speaker") == str(spk_id))
                        and float(s["start"]) < t1 and float(s["end"]) > t0
                    ]
                else:
                    spk_segs = []
                if spk_segs:
                    best = next(
                        (s for s in spk_segs
                         if (s.get("domain_labels") or {}).get("primary")),
                        spk_segs[0],
                    )
                    sets.append({
                        "localize": f"{math.floor(float(best['start']))} sec"
                                    f" to {math.ceil(float(best['end']))} sec",
                        "time": True,
                    })
            if not sets:
                # fallback when spk_wav_to_id unavailable
                mid = (t0 + t1) // 2
                sets = [
                    {"localize": f"{t0} sec to {mid} sec", "time": True},
                    {"localize": f"{mid} sec to {t1} sec", "time": True},
                ]

        else:
            # music (song_a/b blocks) and unknown types
            relevant = self._oracle_relevant(query_item, gt_segs)
            sets = [
                {"localize": f"{math.floor(float(seg['start']))} sec"
                 f" to {math.ceil(float(seg['end']))} sec", "time": True}
                for seg in relevant
            ]

        return {"_oracle": True, "sets": sets}

    def _oracle_relevant(
        self, query_item: dict[str, Any], gt_segs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Music / unknown types: find segs via song_a/b blocks or position."""
        gt_ids = set(query_item.get("gt_seg_ids") or [])
        if gt_ids:
            return [s for s in gt_segs if s.get("seg_id") in gt_ids]

        segs_sorted = sorted(gt_segs, key=lambda s: float(s["start"]))
        seen: set[str] = set()
        result: list[dict[str, Any]] = []

        for key in ("song_a", "song_b", "speaker_a", "speaker_b"):
            blk = query_item.get(key)
            if isinstance(blk, dict) and "time_ref" in blk:
                seg = _seg_at(segs_sorted, float(blk["time_ref"]))
                if seg and seg["seg_id"] not in seen:
                    seen.add(seg["seg_id"])
                    result.append(seg)
        if result:
            return result

        for key in ("song_a", "song_b", "speaker_a", "speaker_b"):
            blk = query_item.get(key)
            if isinstance(blk, dict) and "position" in blk:
                tag = f"SONG_{int(blk['position']):02d}"
                return [s for s in gt_segs if s.get("speaker") == tag]

        return gt_segs


# -- helpers --

def _seg_at(
    segs_sorted: list[dict[str, Any]], t: float
) -> dict[str, Any] | None:
    """GT seg containing t, or the closest one by start time."""
    seg = next(
        (s for s in segs_sorted
         if float(s["start"]) <= t <= float(s["end"])),
        None,
    )
    if seg is None and segs_sorted:
        seg = min(segs_sorted, key=lambda s: abs(float(s["start"]) - t))
    return seg


def _set_guide(plan: dict[str, Any]) -> str:
    """One-line-per-set anchor showing which time window each Set covers.

    Only generated when ≥2 sets have a localize field, to help the model
    match Set N labels to the correct time/song reference.
    """
    sets = plan.get("sets", [])
    if len(sets) < 2:
        return ""
    _ORDER = {1: "first mentioned", 2: "second mentioned", 3: "third mentioned"}
    lines = []
    for i, s in enumerate(sets, 1):
        loc = s.get("localize", "")
        if not loc:
            continue
        order = _ORDER.get(i, f"{i}th")
        lines.append(f"Set {i}: {loc} ({order})")
    if not lines:
        return ""
    return "Evidence set time anchors:\n" + "\n".join(lines)


def _resolve_wav(ref: str, query_item: dict[str, Any]) -> str | None:
    wav_map = query_item.get("speaker_wav_map") or {}
    return wav_map.get(ref) or wav_map.get(ref.upper())


def _encode_clip(clip: dict[str, Any]) -> dict[str, Any]:
    label = clip.get("label") or "[{}-{}s]".format(*clip["window"])
    wave, sr = clip["audio"], clip["sr"]
    # vLLM Qwen Omni requires 16kHz PCM_16 WAV
    if sr != 16000:
        import scipy.signal
        wave = scipy.signal.resample_poly(wave, 16000, sr).astype("float32")
    pcm = (wave * 32767).clip(-32768, 32767).astype("int16")
    buf = io.BytesIO()
    sf.write(buf, pcm, samplerate=16000, format="WAV", subtype="PCM_16")
    return {"label": label, "audio_b64": base64.b64encode(buf.getvalue()).decode()}


def _load_ref_clips(query_item: dict[str, Any]) -> list[dict[str, Any]]:
    """Load speaker_wav_map audio files as reference clips for the answer step."""
    from extract.utils import load_audio
    wav_map = query_item.get("speaker_wav_map") or {}
    clips = []
    for key in sorted(wav_map):
        try:
            wave, sr = load_audio(wav_map[key])
            clips.append({"label": f"[Snippet {key}]", "audio": wave, "sr": sr})
        except Exception:
            pass
    return clips


def _get_track_info(episode_id: str, audio_dir: Path) -> str:
    """Return a one-line duration string for the episode, or empty string."""
    for ext in ("flac", "wav", "mp3"):
        p = Path(audio_dir) / f"{episode_id}.{ext}"
        if p.exists():
            try:
                info = sf.info(str(p))
                m, s = divmod(int(info.duration), 60)
                return f"Track duration: {m} min {s} sec ({int(info.duration)} sec total)"
            except Exception:
                pass
    return ""


def _parse_json(raw: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"```\w*\n?", "", raw).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None


def _parse_answer(raw: str) -> tuple[str, str]:
    if "rationale:" in raw.lower():
        parts = re.split(r"rationale:", raw, maxsplit=1, flags=re.I)
        pred, rat = parts[0].strip(), parts[1].strip()
        # LLM went straight to rationale without writing an answer first
        if not pred:
            return rat, ""
        return pred, rat
    return raw.strip(), ""
