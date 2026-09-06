#!/usr/bin/env python3
"""Run the audio baseline (naive or advanced) over a queries JSONL file.

Naive    (--no-localize, default): full episode audio + transcript.
Advanced (--localize):             audio window from localize_time().

Usage:
    python run_baseline.py conf/emotion_hybrid_all.yaml \\
        --queries /path/to/queries.jsonl \\
        --modalities audio trans \\
        --vllm-url http://localhost:8091/v1

    # advanced baseline:
    python run_baseline.py conf/emotion_hybrid_all.yaml --localize
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from extract.utils import load_config
from inference.baseline_agent import AudioBaselineAgent
from inference.llm.qwen3_omni import Qwen3Omni


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--queries",
        help="Queries JSONL. Defaults to inference.queries in config.",
    )
    parser.add_argument(
        "--modalities", nargs="+",
        choices=["audio", "trans"],
        default=["audio", "trans"],
        metavar="MOD",
    )
    parser.add_argument(
        "--localize", action="store_true",
        help="Advanced baseline: call localize_time() to select audio window.",
    )
    parser.add_argument(
        "--vllm-url", default="http://localhost:8091/v1",
    )
    parser.add_argument(
        "--model-id", default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--out-name",
        help=(
            "Output filename relative to out_dir. "
            "Defaults to baseline_predictions.jsonl or "
            "adv_baseline_predictions.jsonl based on --localize."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    inf_cfg = cfg["inference"]
    out_dir = Path(inf_cfg["out_dir"])
    data_dir = Path(cfg["dataset"]["data_dir"])

    llm = Qwen3Omni(
        base_url=args.vllm_url,
        model_id=args.model_id,
        max_tokens=args.max_tokens,
    )
    agent = AudioBaselineAgent(
        llm=llm,
        audio_dir=data_dir / "podcasts_flac",
        gt_seg_dir=out_dir / "gt" / "segments",
        modalities=args.modalities,
        localize=args.localize,
    )

    queries_path = Path(args.queries or inf_cfg["queries"])
    default_name = (
        "adv_baseline_predictions.jsonl"
        if args.localize
        else "baseline_predictions.jsonl"
    )
    out_path = out_dir / (args.out_name or default_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        queries_path.open(encoding="utf-8") as qf,
        out_path.open("w", encoding="utf-8") as of,
    ):
        for raw in qf:
            raw = raw.strip()
            if not raw:
                continue
            item: Dict[str, Any] = json.loads(raw)
            result = agent(item)
            of.write(json.dumps(result, ensure_ascii=False) + "\n")
            ep = result["episode_id"]
            qt = result["query_type"]
            print(f"[{ep}] [{qt}] {result['prediction']}")

    print(f"\nPredictions saved to {out_path}")


if __name__ == "__main__":
    main()
