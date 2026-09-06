#!/usr/bin/env python3
"""Score predictions against ground truth.

Usage:
    python run_eval.py conf/emotion_hybrid_all.yaml [--resume]
    python run_eval.py conf/emotion_hybrid_all.yaml --predictions data/mspemotion/predictions_v1.jsonl

    # closed-API judges (GPT, Gemini) reading pre-extracted GT evidence:
    python run_eval.py conf/emotion_hybrid_all.yaml --closed-api [--resume]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

import importlib

from extract.utils import build_from_cfg, load_config
from evaluate.agent import EvalAgent


def _model_slug(model_id: str) -> str:
    """Turn a model_id into a safe filename component."""
    return re.sub(r"[^a-zA-Z0-9_-]", "", model_id.replace("/", "_"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--predictions",
        help="Override predictions file from config",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-scored predictions and append to scores file",
    )
    parser.add_argument(
        "--patch",
        action="store_true",
        help="Re-score matching queries and replace them in the scores file",
    )
    parser.add_argument(
        "--query-type", type=str, default=None,
        help="Only score queries with this query_type (use with --patch)",
    )
    parser.add_argument(
        "--rationale-only",
        action="store_true",
        default=False,
        help=(
            "Re-score only factual/faithful/rationale; "
            "preserve answer, plan_hit_rate, retrieval_hit_rate. "
            "Implies --patch."
        ),
    )
    parser.add_argument(
        "--closed-api",
        action="store_true",
        default=False,
        help=(
            "Use ClosedAPIEvalAgent: read pre-extracted GT evidence from "
            "data/inf_evidence/ and score only answer + rationale. "
            "Judge is still taken from the config."
        ),
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=None,
        metavar="TARGET",
        help=(
            "Override judge class, e.g. "
            "evaluate.judge.gemini.GeminiJudge"
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        metavar="MODEL_ID",
        help="Override judge model_id (used with --judge)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help=(
            "Submit all judge calls as a single batch API job "
            "(requires --closed-api and a judge with batch() method, "
            "e.g. OpenAIJudge). Incompatible with --resume/--patch."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ev_cfg = cfg["evaluate"]
    out_dir = Path(ev_cfg["out_dir"])

    # CLI flag takes precedence; YAML closed_api: true is the fallback
    closed_api: bool = args.closed_api or bool(ev_cfg.get("closed_api", False))

    judge_cfg = dict(ev_cfg["judge"])
    if args.judge:
        judge_cfg["_target_"] = args.judge
    if args.judge_model:
        judge_cfg["model_id"] = args.judge_model
    judge = build_from_cfg(judge_cfg)
    gt_dir = Path(ev_cfg["gt_dir"]) if "gt_dir" in ev_cfg else None
    conv_dir = ev_cfg.get("conv_dir")
    metadata_path = ev_cfg.get("metadata_path")

    # load task-specific prompts module (e.g. evaluate.prompts.emotion)
    prompts_target = ev_cfg.get("prompts")
    prompts = importlib.import_module(prompts_target) if prompts_target else None

    preds_path = Path(args.predictions or ev_cfg["predictions"])
    queries_path = out_dir / "queries.jsonl"

    evidence_dir = out_dir.parent / "inf_evidence"
    evidence_path = evidence_dir / f"{out_dir.name}-{preds_path.stem}.jsonl"

    # audio variants answer from raw audio without text metadata → skip faithful
    skip_faithful = "audio" in preds_path.stem
    rationale_only: bool = args.rationale_only
    if rationale_only:
        args.patch = True

    if closed_api:
        from evaluate.closed_api_agent import ClosedAPIEvalAgent
        if not evidence_path.exists():
            raise FileNotFoundError(
                f"Pre-extracted GT evidence not found: {evidence_path}\n"
                "Run without --closed-api first to generate it."
            )
        agent = ClosedAPIEvalAgent(
            judge=judge,
            inf_evidence_path=evidence_path,
            prompts=prompts,
            skip_faithful=skip_faithful,
            rationale_only=rationale_only,
        )
    else:
        agent = EvalAgent(
            judge=judge,
            gt_dir=gt_dir,
            prompts=prompts,
            conv_dir=Path(conv_dir) if conv_dir else None,
            metadata_path=Path(metadata_path) if metadata_path else None,
            skip_faithful=skip_faithful,
            rationale_only=rationale_only,
        )

    # scores file mirrors the predictions filename: predictions_v1 → scores_v1
    scores_stem = preds_path.stem.replace("predictions", "scores", 1)
    if closed_api:
        model_id = judge_cfg.get("model_id", "closedapi")
        scores_path = out_dir / f"{scores_stem}_{_model_slug(model_id)}.jsonl"
    else:
        scores_path = out_dir / f"{scores_stem}.jsonl"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    queries: dict = {}
    if queries_path.exists():
        with queries_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    q = json.loads(line)
                    queries[q["query_id"]] = q

    # patch mode: re-score selected query_type, rewrite scores file in place
    if args.patch:
        existing: dict = {}
        existing_order: list = []
        if scores_path.exists():
            with scores_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    qid = r.get("query_id")
                    if qid not in existing:
                        existing_order.append(qid)
                    existing[qid] = r
            print(f"Patch mode — loaded {len(existing)} existing scores")
        ev_existing: dict = {}
        ev_order: list = []
        if not closed_api and evidence_path.exists():
            with evidence_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    qid = r.get("query_id")
                    if qid not in ev_existing:
                        ev_order.append(qid)
                    ev_existing[qid] = r
        total = sum(1 for ln in preds_path.open(encoding="utf-8") if ln.strip())
        n_run = 0
        with (
            preds_path.open(encoding="utf-8") as pf,
            tqdm(total=total, desc="evaluate", unit="pred") as bar,
        ):
            for line in pf:
                line = line.strip()
                if not line:
                    continue
                pred = json.loads(line)
                bar.update(1)
                query_item = queries.get(pred.get("query_id"), pred)
                if args.query_type and query_item.get("query_type") != args.query_type:
                    bar.set_postfix_str("skip")
                    continue
                score = agent(pred, query_item)
                gt_ev = score.pop("_gt_ev", [])
                gt_side = score.pop("_gt_side", "")
                qid = score["query_id"]
                if qid not in existing:
                    existing_order.append(qid)
                if rationale_only and qid in existing:
                    _RAT_FIELDS = ("factual", "faithful",
                                   "rationale", "judge_trace")
                    existing[qid] = {
                        **existing[qid],
                        **{k: score[k] for k in _RAT_FIELDS if k in score},
                    }
                else:
                    existing[qid] = score
                if not closed_api:
                    ev_rec = {"query_id": qid, "episode_id": score["episode_id"],
                              "gt_ev": gt_ev, "gt_side": gt_side}
                    if qid not in ev_existing:
                        ev_order.append(qid)
                    ev_existing[qid] = ev_rec
                n_run += 1
                bar.set_postfix_str(str(pred.get("episode_id", "")))
        all_ids = existing_order + [k for k in existing if k not in set(existing_order)]
        with scores_path.open("w", encoding="utf-8") as sf:
            for qid in all_ids:
                sf.write(json.dumps(existing[qid], ensure_ascii=False) + "\n")
        if not closed_api:
            all_ev_ids = ev_order + [k for k in ev_existing if k not in set(ev_order)]
            with evidence_path.open("w", encoding="utf-8") as ef:
                for qid in all_ev_ids:
                    ef.write(json.dumps(ev_existing[qid], ensure_ascii=False) + "\n")
        print(f"Patched {n_run} scores → {scores_path}")
        return

    # batch mode: collect all prompts, submit one batch job, write scores
    if args.batch:
        if not closed_api:
            raise ValueError("--batch requires --closed-api")
        predictions = [
            json.loads(ln) for ln in preds_path.open(encoding="utf-8")
            if ln.strip()
        ]
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Batch mode — {len(predictions)} predictions → "
            f"{scores_path}"
        )
        scores = agent.batch_score(predictions, queries)
        with scores_path.open("w", encoding="utf-8") as sf:
            for s in scores:
                sf.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Scores written to {scores_path}")
        return

    # resume: collect already-done query_ids from existing scores file
    done_ids: set = set()
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    if scores_path.exists():
        if args.resume:
            with scores_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        if r.get("query_id") is not None:
                            done_ids.add(r["query_id"])
            print(
                f"Resuming — {len(done_ids)} scores already done,"
                f" appending to {scores_path}"
            )
        else:
            print(
                f"WARNING: {scores_path} already exists and will be"
                " overwritten. Pass --resume to continue from where"
                " it left off."
            )

    total = sum(
        1 for ln in preds_path.open(encoding="utf-8") if ln.strip()
    )
    file_mode = "a" if args.resume else "w"
    if closed_api:
        with (
            preds_path.open(encoding="utf-8") as pf,
            scores_path.open(file_mode, encoding="utf-8") as sf,
            tqdm(total=total, desc="evaluate", unit="pred") as bar,
        ):
            for line in pf:
                line = line.strip()
                if not line:
                    continue
                pred = json.loads(line)
                bar.update(1)
                if pred.get("query_id") in done_ids:
                    bar.set_postfix_str("skip")
                    continue
                query_item = queries.get(pred.get("query_id"), pred)
                score = agent(pred, query_item)
                sf.write(json.dumps(score, ensure_ascii=False) + "\n")
                sf.flush()
                bar.set_postfix_str(str(pred.get("episode_id", "")))
    else:
        with (
            preds_path.open(encoding="utf-8") as pf,
            scores_path.open(file_mode, encoding="utf-8") as sf,
            evidence_path.open(file_mode, encoding="utf-8") as ef,
            tqdm(total=total, desc="evaluate", unit="pred") as bar,
        ):
            for line in pf:
                line = line.strip()
                if not line:
                    continue
                pred = json.loads(line)
                bar.update(1)
                if pred.get("query_id") in done_ids:
                    bar.set_postfix_str("skip")
                    continue
                query_item = queries.get(pred.get("query_id"), pred)
                score = agent(pred, query_item)
                gt_ev = score.pop("_gt_ev", [])
                gt_side = score.pop("_gt_side", "")
                sf.write(json.dumps(score, ensure_ascii=False) + "\n")
                sf.flush()
                ev_rec = {
                    "query_id": score["query_id"],
                    "episode_id": score["episode_id"],
                    "gt_ev": gt_ev,
                    "gt_side": gt_side,
                }
                ef.write(json.dumps(ev_rec, ensure_ascii=False) + "\n")
                ef.flush()
                bar.set_postfix_str(str(pred.get("episode_id", "")))

    print(f"Scores written to {scores_path}")


if __name__ == "__main__":
    main()
