from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import soundfile as sf

from extract.domain_expert.base import DomainEmbBase, DomainLabelerBase
from extract.audio_expert.base import ASRBase, DiarBase, EmbBase
from extract.utils import append_faiss, build_from_cfg, make_seg_id


def _merge_short_turns(
    turns: List[Dict[str, Any]], min_sec: float
) -> List[Dict[str, Any]]:
    """Merge turns shorter than min_sec into their temporal neighbor.

    Iterates until stable; turns whose total audio is still < min_sec
    after all merges (isolated fragments) are dropped."""
    if not turns:
        return turns
    changed = True
    while changed:
        changed = False
        out: List[Dict[str, Any]] = []
        for t in turns:
            if t["end"] - t["start"] < min_sec and out:
                out[-1] = {**out[-1], "end": t["end"]}
                changed = True
            else:
                out.append(t)
        if len(out) >= 2 and out[0]["end"] - out[0]["start"] < min_sec:
            out[1] = {**out[1], "start": out[0]["start"]}
            out = out[1:]
            changed = True
        turns = out
    return [t for t in turns if t["end"] - t["start"] >= min_sec]


def _split_turns(
    turns: List[Dict[str, Any]], max_sec: float
) -> List[Dict[str, Any]]:
    """Split any turn longer than max_sec into equal-length chunks."""
    out: List[Dict[str, Any]] = []
    for t in turns:
        dur = t["end"] - t["start"]
        if dur <= max_sec:
            out.append(t)
            continue
        n = int(np.ceil(dur / max_sec))
        chunk = dur / n
        for i in range(n):
            out.append({
                **t,
                "start": t["start"] + i * chunk,
                "end": t["start"] + (i + 1) * chunk,
            })
    return out


# Stage 1: VAD + ASR + speaker embedding → cache/

def extract_episode_stage1(
    episode: Dict[str, Any],
    *,
    asr: ASRBase,
    diar: DiarBase,
    audio_emb: EmbBase,
    out_dir: Path,
    batch_size: int = 8,
    text_emb=None,
    max_seg_sec: float | None = None,
) -> None:
    """VAD + ASR + speaker (+text) embedding → cache/.

    Writes {episode_id}_turns.jsonl, {episode_id}_spk.npy,
    and (optionally) {episode_id}_text.npy. Skips if all exist.
    If turns/spk exist but _text.npy is missing, only runs text emb."""
    episode_id: str = episode["episode_id"]
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    turns_path = cache_dir / f"{episode_id}_turns.jsonl"
    spk_path = cache_dir / f"{episode_id}_spk.npy"
    txt_path = cache_dir / f"{episode_id}_text.npy"
    need_text = text_emb is not None

    if turns_path.exists() and spk_path.exists():
        if not need_text or txt_path.exists():
            return
        # turns + spk already done; only text embedding needed
        _run_text_emb(turns_path, txt_path, text_emb, batch_size)
        return

    wave, sr = sf.read(
        episode["audio_path"], always_2d=False, dtype="float32"
    )
    if wave.ndim == 2:
        wave = wave.mean(axis=1)

    turns = diar.segment(wave, sr)
    if max_seg_sec is not None:
        turns = _split_turns(turns, max_seg_sec)
    turns = _merge_short_turns(turns, min_sec=1.0)
    if not turns:
        turns_path.touch()
        np.save(str(spk_path), np.empty((0, 0), dtype=np.float32))
        return

    turn_records: List[Dict[str, Any]] = []
    spk_vecs: List[np.ndarray] = []
    txt_vecs: List[np.ndarray] = []

    for batch_start in range(0, len(turns), batch_size):
        batch = turns[batch_start: batch_start + batch_size]
        waves = [
            wave[int(t["start"] * sr): int(t["end"] * sr)]
            for t in batch
        ]
        srs = [sr] * len(batch)

        texts = asr.transcribe_batch(waves, srs)
        s_vecs = audio_emb.embed_batch(waves, srs)
        if need_text:
            t_vecs = text_emb.embed_batch(texts)

        for i, (turn, text) in enumerate(zip(batch, texts)):
            turn_records.append({
                "turn_idx": batch_start + i,
                "start": float(turn["start"]),
                "end": float(turn["end"]),
                "speaker": turn.get("speaker", "SPEAKER_00"),
                "text": text,
            })
            spk_vecs.append(s_vecs[i])
            if need_text:
                txt_vecs.append(t_vecs[i])

    np.save(str(spk_path), np.stack(spk_vecs).astype(np.float32))
    with turns_path.open("w", encoding="utf-8") as f:
        for rec in turn_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if need_text and txt_vecs:
        np.save(str(txt_path), np.stack(txt_vecs).astype(np.float32))


def _run_text_emb(
    turns_path: Path,
    txt_path: Path,
    text_emb,
    batch_size: int,
) -> None:
    """Embed texts from an existing turns cache; save to txt_path."""
    turn_records: List[Dict[str, Any]] = []
    with turns_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                turn_records.append(json.loads(line))
    if not turn_records:
        return
    txt_vecs: List[np.ndarray] = []
    for batch_start in range(0, len(turn_records), batch_size):
        batch = turn_records[batch_start: batch_start + batch_size]
        vecs = text_emb.embed_batch([r["text"] for r in batch])
        for i in range(len(batch)):
            txt_vecs.append(vecs[i])
    np.save(str(txt_path), np.stack(txt_vecs).astype(np.float32))


# Stage 2: domain label + emb + cluster → segments/ + FAISS

def extract_episode_stage2(
    episode: Dict[str, Any],
    *,
    domain_labelers: List[DomainLabelerBase],
    domain_embs: List[DomainEmbBase],
    out_dir: Path,
    batch_size: int = 8,
) -> None:
    """Reads cache/, runs domain experts, writes final
    segments/{episode_id}.jsonl and appends to FAISS. Skips if segment
    file already exists.

    domain_labelers: list of labelers; outputs merged left-to-right
      (non-None values from later labelers override earlier ones).
    domain_embs: list of embedders; each gets its own FAISS index
      domain_{i}.index / domain_{i}_meta.jsonl.
    """
    episode_id: str = episode["episode_id"]
    seg_dir = out_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_path = seg_dir / f"{episode_id}.jsonl"
    if seg_path.exists():
        return

    cache_dir = out_dir / "cache"
    turns_path = cache_dir / f"{episode_id}_turns.jsonl"
    spk_path = cache_dir / f"{episode_id}_spk.npy"
    if not turns_path.exists() or not spk_path.exists():
        raise FileNotFoundError(
            f"Stage 1 cache missing for {episode_id}; run stage 1 first."
        )

    turn_records: List[Dict[str, Any]] = []
    with turns_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                turn_records.append(json.loads(line))

    if not turn_records:
        seg_path.touch()
        return

    spk_vecs = np.load(str(spk_path))
    txt_path = cache_dir / f"{episode_id}_text.npy"
    txt_vecs = np.load(str(txt_path)) if txt_path.exists() else None

    # load audio for domain processing
    wave, sr = sf.read(
        episode["audio_path"], always_2d=False, dtype="float32"
    )
    if wave.ndim == 2:
        wave = wave.mean(axis=1)

    # dom_vecs_per_emb[i] = list of vecs for embedder i
    dom_vecs_per_emb: List[List[np.ndarray]] = [
        [] for _ in domain_embs
    ]
    domain_labels: List[Dict[str, Any]] = []

    for batch_start in range(0, len(turn_records), batch_size):
        batch = turn_records[batch_start: batch_start + batch_size]
        waves = [
            wave[int(r["start"] * sr): int(r["end"] * sr)]
            for r in batch
        ]
        srs = [sr] * len(batch)

        # merge labelers left-to-right; non-None values override
        merged: List[Dict[str, Any]] = [
            domain_labelers[0].label(w, s)
            for w, s in zip(waves, srs)
        ] if domain_labelers else [{} for _ in batch]
        for labeler in domain_labelers[1:]:
            extra = [labeler.label(w, s) for w, s in zip(waves, srs)]
            for m, e in zip(merged, extra):
                for k, v in e.items():
                    if v is not None:
                        m[k] = v
        domain_labels.extend(merged)

        for ei, emb in enumerate(domain_embs):
            vecs = emb.embed_batch(waves, srs)
            for i in range(len(batch)):
                dom_vecs_per_emb[ei].append(vecs[i])

    spk_ids = [r.get("speaker", "SPEAKER_00") for r in turn_records]

    faiss_dir = out_dir / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    segments: List[Dict[str, Any]] = []
    for i, rec in enumerate(turn_records):
        seg_id = make_seg_id(episode_id, rec["turn_idx"])
        segments.append({
            "seg_id": seg_id,
            "episode_id": episode_id,
            "start": rec["start"],
            "end": rec["end"],
            "speaker": spk_ids[i] if i < len(spk_ids) else "SPEAKER_00",
            "text": rec["text"],
            "domain_labels": domain_labels[i],
            "audio_faiss_idx": None,
            "domain_faiss_idxs": [],
            "text_faiss_idx": None,
        })

    base_meta = [
        {
            "seg_id": s["seg_id"],
            "episode_id": episode_id,
            "start": s["start"],
            "end": s["end"],
        }
        for s in segments
    ]
    spk_meta = [dict(m) for m in base_meta]
    txt_meta = [dict(m) for m in base_meta] if txt_vecs is not None else None

    append_faiss(
        np.stack(spk_vecs),
        spk_meta,
        faiss_dir / "audio.index",
        faiss_dir / "audio_meta.jsonl",
    )
    for ei, vecs_list in enumerate(dom_vecs_per_emb):
        dom_meta = [dict(m) for m in base_meta]
        append_faiss(
            np.stack(vecs_list),
            dom_meta,
            faiss_dir / f"domain_{ei}.index",
            faiss_dir / f"domain_{ei}_meta.jsonl",
        )
        for seg, dm in zip(segments, dom_meta):
            seg["domain_faiss_idxs"].append(dm["faiss_idx"])
    if txt_vecs is not None and txt_meta is not None:
        append_faiss(
            txt_vecs,
            txt_meta,
            faiss_dir / "text.index",
            faiss_dir / "text_meta.jsonl",
        )

    for seg, sm in zip(segments, spk_meta):
        seg["audio_faiss_idx"] = sm["faiss_idx"]
    if txt_vecs is not None and txt_meta is not None:
        for seg, tm in zip(segments, txt_meta):
            seg["text_faiss_idx"] = tm["faiss_idx"]

    with seg_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")


# Stage 2 labeler-only patch (for updating a labeler after stage 2 ran)

def extract_episode_stage2_label(
    episode: Dict[str, Any],
    *,
    labeler: DomainLabelerBase,
    out_dir: Path,
    batch_size: int = 8,
) -> None:
    """Re-run one labeler and merge its non-None outputs into existing
    segments/{episode_id}.jsonl.  Skips if already patched (checks that
    any AVD dim has 'start' key, i.e. new CSER format)."""
    episode_id: str = episode["episode_id"]
    seg_path = out_dir / "segments" / f"{episode_id}.jsonl"
    if not seg_path.exists():
        raise FileNotFoundError(
            f"Segments missing for {episode_id}; run stage 2 first."
        )

    segments: List[Dict[str, Any]] = []
    with seg_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                segments.append(json.loads(line))
    if not segments:
        return

    # skip if first segment already has the new fields
    dl0 = segments[0].get("domain_labels", {})
    if isinstance(dl0.get("arousal"), dict) and "start" in dl0["arousal"]:
        return

    wave, sr = sf.read(
        episode["audio_path"], always_2d=False, dtype="float32"
    )
    if wave.ndim == 2:
        wave = wave.mean(axis=1)

    if hasattr(labeler, "label_episode"):
        labeler.label_episode(wave, sr, segments)
    else:
        for batch_start in range(0, len(segments), batch_size):
            batch = segments[batch_start: batch_start + batch_size]
            for seg in batch:
                w = wave[int(seg["start"] * sr): int(seg["end"] * sr)]
                result = labeler.label(w, sr)
                dl = seg.setdefault("domain_labels", {})
                for k, v in result.items():
                    if v is not None:
                        dl[k] = v

    with seg_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")


# Stage 2 embedder-only patch (for adding an embedder after stage 2 ran)

def extract_episode_stage2_emb(
    episode: Dict[str, Any],
    *,
    text_emb,
    out_dir: Path,
    batch_size: int = 8,
) -> None:
    """Patch text FAISS into existing stage-2 segments.

    Reads or computes _text.npy, appends to text.index, updates
    text_faiss_idx in segments/{episode_id}.jsonl. Skips if
    text_faiss_idx is already set on the first segment."""
    episode_id: str = episode["episode_id"]
    seg_path = out_dir / "segments" / f"{episode_id}.jsonl"
    if not seg_path.exists():
        raise FileNotFoundError(
            f"Segments missing for {episode_id}; run stage 2 first."
        )

    segments: List[Dict[str, Any]] = []
    with seg_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                segments.append(json.loads(line))
    if not segments:
        return
    if segments[0].get("text_faiss_idx") is not None:
        return  # already done

    cache_dir = out_dir / "cache"
    turns_path = cache_dir / f"{episode_id}_turns.jsonl"
    txt_path = cache_dir / f"{episode_id}_text.npy"

    if not txt_path.exists():
        if not turns_path.exists():
            raise FileNotFoundError(
                f"No turns cache for {episode_id}; cannot embed text."
            )
        _run_text_emb(turns_path, txt_path, text_emb, batch_size)

    txt_vecs = np.load(str(txt_path))
    faiss_dir = out_dir / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    meta = [
        {
            "seg_id": s["seg_id"],
            "episode_id": episode_id,
            "start": s["start"],
            "end": s["end"],
        }
        for s in segments
    ]
    append_faiss(
        txt_vecs,
        meta,
        faiss_dir / "text.index",
        faiss_dir / "text_meta.jsonl",
    )
    for seg, m in zip(segments, meta):
        seg["text_faiss_idx"] = m["faiss_idx"]

    with seg_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")


# Dataset-level drivers

def extract_dataset_stage1(
    episodes: Iterable[Dict[str, Any]],
    cfg: Dict[str, Any],
    out_dir: Path,
) -> None:
    import torch
    asr = build_from_cfg(cfg["asr"])
    diar_obj = build_from_cfg(cfg["diar"])
    spk_emb = build_from_cfg(cfg["audio_emb"])
    text_emb = (
        build_from_cfg(cfg["text_emb"]) if "text_emb" in cfg else None
    )
    batch_size = int(cfg.get("batch_size", 8))
    max_seg_sec = cfg.get("max_seg_sec", None)
    if max_seg_sec is not None:
        max_seg_sec = float(max_seg_sec)
    for ep in episodes:
        extract_episode_stage1(
            ep,
            asr=asr,
            diar=diar_obj,
            audio_emb=spk_emb,
            out_dir=out_dir,
            batch_size=batch_size,
            text_emb=text_emb,
            max_seg_sec=max_seg_sec,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_dataset_stage1_emb(
    episodes: Iterable[Dict[str, Any]],
    cfg: Dict[str, Any],
    out_dir: Path,
    embedder: str,
) -> None:
    """Run a single embedder over already-cached stage-1 turns.

    Currently supports embedder='text'. Skips episodes whose
    output already exists."""
    if embedder != "text":
        raise ValueError(f"Unknown embedder '{embedder}'; only 'text' supported")
    if "text_emb" not in cfg:
        raise KeyError("'text_emb' key missing from extract config")
    text_emb = build_from_cfg(cfg["text_emb"])
    batch_size = int(cfg.get("batch_size", 8))
    cache_dir = out_dir / "cache"
    for ep in episodes:
        episode_id = str(ep["episode_id"])
        turns_path = cache_dir / f"{episode_id}_turns.jsonl"
        txt_path = cache_dir / f"{episode_id}_text.npy"
        if txt_path.exists():
            continue
        if not turns_path.exists():
            print(f"  Skipping {episode_id}: no turns cache")
            continue
        _run_text_emb(turns_path, txt_path, text_emb, batch_size)


def extract_dataset_stage2(
    episodes: Iterable[Dict[str, Any]],
    cfg: Dict[str, Any],
    out_dir: Path,
) -> None:
    dom_labs = [build_from_cfg(c) for c in cfg["domain_labelers"]]
    dom_embs = [build_from_cfg(c) for c in cfg["domain_embedders"]]
    batch_size = int(cfg.get("batch_size", 8))
    for ep in episodes:
        extract_episode_stage2(
            ep,
            domain_labelers=dom_labs,
            domain_embs=dom_embs,
            out_dir=out_dir,
            batch_size=batch_size,
        )


def extract_dataset_stage2_emb(
    episodes: Iterable[Dict[str, Any]],
    cfg: Dict[str, Any],
    out_dir: Path,
    embedder: str,
) -> None:
    """Patch a single embedder into existing stage-2 segments."""
    if embedder != "text":
        raise ValueError(f"Unknown embedder '{embedder}'; only 'text' supported")
    if "text_emb" not in cfg:
        raise KeyError("'text_emb' key missing from extract config")
    text_emb = build_from_cfg(cfg["text_emb"])
    batch_size = int(cfg.get("batch_size", 8))
    for ep in episodes:
        extract_episode_stage2_emb(
            ep,
            text_emb=text_emb,
            out_dir=out_dir,
            batch_size=batch_size,
        )


def extract_dataset_stage2_label(
    episodes: Iterable[Dict[str, Any]],
    cfg: Dict[str, Any],
    out_dir: Path,
    labeler_name: str,
) -> None:
    """Patch a single labeler into existing stage-2 segments.

    labeler_name is matched as a substring of the labeler's _target_.
    Example: 'cser' matches 'extract.domain_expert.emotion.labeler_cser.CSERLabeler'.
    """
    labeler_cfgs = cfg.get("domain_labelers", [])
    matched = [
        c for c in labeler_cfgs
        if labeler_name.lower() in c.get("_target_", "").lower()
    ]
    if not matched:
        raise ValueError(
            f"No labeler with '{labeler_name}' in _target_ found in config"
        )
    labeler = build_from_cfg(matched[0])
    batch_size = int(cfg.get("batch_size", 8))
    for ep in episodes:
        extract_episode_stage2_label(
            ep,
            labeler=labeler,
            out_dir=out_dir,
            batch_size=batch_size,
        )
