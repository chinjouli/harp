#!/usr/bin/env python3
"""Run HARP inference over a queries JSONL file.

Usage:
    python run_inference.py conf/emotion_hybrid_all.yaml [--resume]
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from tqdm import tqdm

from extract.utils import build_from_cfg, load_config
from inference.agent import HARPAgent, OracleAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-written predictions and append to output",
    )
    parser.add_argument(
        "--max-queries", type=int, default=None,
        help="Stop after this many queries (for testing)",
    )
    parser.add_argument(
        "--query-type", type=str, default=None,
        help="Only process queries with this query_type value (e.g. audio)",
    )
    parser.add_argument(
        "--patch",
        action="store_true",
        help=(
            "Re-run matching queries and replace them in the output file; "
            "preserves all other predictions in original order"
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    inf_cfg = cfg["inference"]
    out_dir = Path(inf_cfg["out_dir"])
    segments_dir = Path(inf_cfg.get("segments_dir", str(out_dir / "segments")))
    faiss_dir = Path(inf_cfg.get("faiss_dir", str(out_dir / "faiss")))

    llm = build_from_cfg(inf_cfg["llm"])
    task = inf_cfg.get("task", "emotion")
    prompts = importlib.import_module(f"inference.prompts.{task}")

    use_vec_cache = bool(inf_cfg.get("use_vec_cache", False))
    audio_vec_cache = domain_vec_cache = None
    if use_vec_cache:
        import numpy as np
        vec_cache_dir = out_dir / "vec_cache"
        audio_vec_cache = dict(np.load(vec_cache_dir / "audio.npz"))
        domain_vec_cache = dict(np.load(vec_cache_dir / "domain_0.npz"))

    label_cache: dict | None = None
    label_cache_path = out_dir / "label_cache.json"
    if label_cache_path.exists():
        with label_cache_path.open(encoding="utf-8") as f:
            label_cache = json.load(f)

    _common = dict(
        llm=llm,
        segments_dir=segments_dir,
        faiss_dir=faiss_dir,
        audio_dir=Path(inf_cfg.get("audio_dir", "")),
        prompts=prompts,
        modalities=inf_cfg.get("modalities", None),
        top_k=int(inf_cfg.get("top_k", 10)),
        max_audio_clip_sec=int(inf_cfg.get("max_audio_clip_sec", 120)),
        strip_domain=bool(inf_cfg.get("strip_domain", False)),
        use_vec_cache=use_vec_cache,
        audio_vec_cache=audio_vec_cache,
        domain_vec_cache=domain_vec_cache,
        task=task,
    )
    if inf_cfg.get("oracle"):
        spk_wav_to_id: dict = {}
        spk_index_path = inf_cfg.get("spk_index")
        if spk_index_path and Path(spk_index_path).exists():
            idx = json.load(open(spk_index_path, encoding="utf-8"))
            spk_wav_to_id = {
                e["wav"]: int(k)
                for k, entries in idx.items()
                for e in entries
            }
        agent: HARPAgent = OracleAgent(
            **_common,
            gt_dir=Path(inf_cfg["gt_dir"]) if "gt_dir" in inf_cfg else None,
            spk_wav_to_id=spk_wav_to_id,
        )
    else:
        agent = HARPAgent(
            **_common,
            text_emb=(
                build_from_cfg(inf_cfg["text_emb"])
                if "text_emb" in inf_cfg else None
            ),
            audio_emb=(
                build_from_cfg(inf_cfg["audio_emb"])
                if "audio_emb" in inf_cfg else None
            ),
            domain_emb=(
                build_from_cfg(inf_cfg["domain_emb"])
                if "domain_emb" in inf_cfg else None
            ),
            asr=(
                build_from_cfg(inf_cfg["asr"])
                if inf_cfg.get("preprocess") and "asr" in inf_cfg else None
            ),
            domain_labeler=(
                build_from_cfg(inf_cfg["domain_labeler"])
                if inf_cfg.get("preprocess") and "domain_labeler" in inf_cfg else None
            ),
            label_cache=label_cache,
            clip_audio_dir=Path(inf_cfg["clip_audio_dir"])
            if "clip_audio_dir" in inf_cfg else None,
        )

    spk_example_dir = Path(cfg["extract"].get("spk_example_dir", ""))
    spk_audio_dir = Path(cfg["extract"].get("spk_audio_dir", ""))
    clip_audio_dir = Path(cfg["extract"].get("clip_audio_dir", ""))

    def _get_spk_wav_map(query_item: dict, episode_id: str) -> dict:
        m: dict = {}
        if spk_audio_dir.exists():
            for field, key in [
                ("audio_spk", "A"), ("audio_spk2", "B"), ("audio_emo", "E")
            ]:
                if query_item.get(field):
                    m[key] = str(spk_audio_dir / query_item[field])
        if clip_audio_dir.exists():
            for sub, key in [("song_a", "A"), ("song_b", "B")]:
                clip = (query_item.get(sub) or {}).get("audio_clip")
                if clip:
                    m[key] = str(clip_audio_dir / "audio" / Path(clip).name)
        if not m:
            p = spk_example_dir / f"{episode_id}_spk_map.json"
            if p.exists():
                clips = json.loads(p.read_text()).get("clips", {})
                m = {
                    k: str(spk_example_dir.parent / v["wav"])
                    for k, v in clips.items()
                    if v.get("wav")
                }
        return m

    queries_path = Path(inf_cfg["queries"])
    out_path = Path(
        inf_cfg.get("predictions", str(out_dir / "predictions.jsonl"))
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # patch mode: load existing predictions, re-run matching queries, rewrite
    if args.patch:
        existing: dict = {}
        existing_order: list = []
        if out_path.exists():
            with out_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    qid = r.get("query_id")
                    if qid not in existing:
                        existing_order.append(qid)
                    existing[qid] = r
            print(f"Patch mode — loaded {len(existing)} existing predictions")

        total = sum(1 for ln in queries_path.open(encoding="utf-8") if ln.strip())
        n_run = 0
        with (
            queries_path.open(encoding="utf-8") as qf,
            tqdm(total=total, desc="inference", unit="query") as bar,
        ):
            for line in qf:
                line = line.strip()
                if not line:
                    continue
                query_item = json.loads(line)
                bar.update(1)
                if args.query_type and query_item.get("query_type") != args.query_type:
                    bar.set_postfix_str("skip")
                    continue
                if args.max_queries and n_run >= args.max_queries:
                    break
                episode_id = str(query_item["episode_id"])
                query_item["speaker_wav_map"] = _get_spk_wav_map(query_item, episode_id)
                result = agent(query_item)
                qid = result.get("query_id")
                if qid not in existing:
                    existing_order.append(qid)
                existing[qid] = result
                n_run += 1
                bar.set_postfix_str(episode_id)

        # rewrite in original order, appending any new ids at the end
        all_ids = existing_order + [k for k in existing if k not in set(existing_order)]
        with out_path.open("w", encoding="utf-8") as of:
            for qid in all_ids:
                of.write(json.dumps(existing[qid], ensure_ascii=False) + "\n")
        print(f"Patched {n_run} predictions → {out_path}")
        return

    done_ids: set = set()
    if out_path.exists():
        if args.resume:
            with out_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        if r.get("query_id") is not None:
                            done_ids.add(r["query_id"])
            print(
                f"Resuming — {len(done_ids)} predictions already done,"
                f" appending to {out_path}"
            )
        else:
            print(
                f"WARNING: {out_path} exists and will be overwritten."
                " Pass --resume to continue."
            )

    total = sum(1 for ln in queries_path.open(encoding="utf-8") if ln.strip())
    file_mode = "a" if args.resume else "w"
    with (
        queries_path.open(encoding="utf-8") as qf,
        out_path.open(file_mode, encoding="utf-8") as of,
        tqdm(total=total, desc="inference", unit="query") as bar,
    ):
        for line in qf:
            line = line.strip()
            if not line:
                continue
            query_item = json.loads(line)
            bar.update(1)
            if args.query_type and query_item.get("query_type") != args.query_type:
                bar.set_postfix_str("skip")
                continue
            if query_item.get("query_id") in done_ids:
                bar.set_postfix_str("skip")
                continue
            if args.max_queries and bar.n > args.max_queries:
                break
            episode_id = str(query_item["episode_id"])
            query_item["speaker_wav_map"] = _get_spk_wav_map(query_item, episode_id)
            result = agent(query_item)
            of.write(json.dumps(result, ensure_ascii=False) + "\n")
            of.flush()
            bar.set_postfix_str(episode_id)


if __name__ == "__main__":
    main()
