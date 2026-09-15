# Scripts

Entry points for the pipeline. Run them from the repo root.

| Script | Does | Details |
|---|---|---|
| `run_extract.py` | Build the index: diarization + ASR + embeddings, then domain labels | [`../extract/`](../extract/README.md) |
| `run_inference.py` | Plan → retrieve → gather → answer over a queries JSONL | [`../inference/`](../inference/README.md) |
| `run_eval.py` | Judge predictions against ground truth | [`../evaluate/`](../evaluate/README.md) |
| `run_score.py` | Aggregate `scores_*.jsonl` by query type | [`../evaluate/`](../evaluate/README.md) |
| `run_judgeeval.py` | Inter-judge agreement: Cohen's κ, Fleiss' κ, leniency | [`../evaluate/`](../evaluate/README.md) |
| `run_baseline.py` | Audio baseline — full episode or a localized window, no retrieval | — |
| `serve_vllm.sh` | Serve a Qwen backbone with vLLM on `$PORT` (default 8091) | [`../inference/`](../inference/README.md) |

Every script takes a config as its first argument; `--help` lists the rest.

```bash
scripts/serve_vllm.sh text          # or: omni
python scripts/run_inference.py conf/emotion_hybrid_all.yaml --resume
```

## Slurm

`slurm/` holds job files for a GPU cluster — a starting point, not a portable interface. `--partition` and `--account` ship as `CHANGE_ME`; set them for your site first.

```bash
sbatch scripts/slurm/extract.sbatch conf/emotion_hybrid_all.yaml 1   # config, stage
sbatch scripts/slurm/infer.sbatch conf/emotion_hybrid_all.yaml --resume
sbatch scripts/slurm/serve_vllm.sbatch omni
```

Each forwards its remaining arguments to the underlying script, so anything valid above works here too.
