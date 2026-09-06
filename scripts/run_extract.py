#!/usr/bin/env python3
"""Extract segments and embeddings for all episodes in a dataset.

Usage:
    python run_extract.py conf/emotion_hybrid_all.yaml [--stage 1|2]
                                                    [--embedder text]
                                                    [--labeler cser]

Stage 1: VAD/diar + ASR + speaker embedding  → data/.../cache/
Stage 2: domain label + embedding            → data/.../segments/ + faiss/
Omitting --stage runs both sequentially.

--embedder: run only a single embedder over existing stage-1 cache
            (e.g. --stage 1 --embedder text). Requires --stage 1 or 2.
--labeler:  patch a single labeler into existing stage-2 segments
            (e.g. --stage 2 --labeler cser). Requires --stage 2.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from extract.pipeline import (
    extract_dataset_stage1,
    extract_dataset_stage1_emb,
    extract_dataset_stage2,
    extract_dataset_stage2_emb,
    extract_dataset_stage2_label,
)
from extract.utils import build_from_cfg, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--stage", type=int, choices=[1, 2], default=None,
        help="1=audio/speech, 2=domain. Omit to run both.",
    )
    parser.add_argument(
        "--embedder", default=None,
        help="Run only this embedder on existing cache "
             "(use with --stage 1 or --stage 2). Example: text",
    )
    parser.add_argument(
        "--labeler", default=None,
        help="Patch only this labeler into existing stage-2 segments "
             "(use with --stage 2). Example: cser",
    )
    args = parser.parse_args()

    if args.embedder and args.stage not in (1, 2):
        parser.error("--embedder requires --stage 1 or --stage 2")
    if args.labeler and args.stage != 2:
        parser.error("--labeler requires --stage 2")

    cfg = load_config(args.config)
    ext_cfg = cfg["extract"]
    out_dir = Path(ext_cfg["out_dir"])

    dataset = build_from_cfg(cfg["dataset"])
    if args.stage == 1:
        if args.embedder:
            extract_dataset_stage1_emb(
                dataset, ext_cfg, out_dir, args.embedder
            )
        else:
            extract_dataset_stage1(dataset, ext_cfg, out_dir)
    elif args.stage == 2:
        if args.labeler:
            extract_dataset_stage2_label(
                dataset, ext_cfg, out_dir, args.labeler
            )
        elif args.embedder:
            extract_dataset_stage2_emb(
                dataset, ext_cfg, out_dir, args.embedder
            )
        else:
            extract_dataset_stage2(dataset, ext_cfg, out_dir)
    else:
        eps = list(dataset)
        extract_dataset_stage1(eps, ext_cfg, out_dir)
        extract_dataset_stage2(eps, ext_cfg, out_dir)


if __name__ == "__main__":
    main()
