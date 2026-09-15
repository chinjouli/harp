# HARP: Agentic Hybrid Retrieval and Analysis for Long-Form Audio

[![arXiv](https://img.shields.io/badge/arXiv-2609.14116-b31b1b)](https://arxiv.org/abs/2609.14116) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-cjli/harp-ffd21e)](https://huggingface.co/cjli/harp) [![GitHub](https://img.shields.io/badge/GitHub-chinjouli/harp-181717?logo=github)](https://github.com/chinjouli/harp) [![License](https://img.shields.io/badge/License-MIT-3da639)](LICENSE)

Long-form audio analysis requires systems to localize and integrate evidence distributed across extended recordings. While existing work primarily retrieves semantic content through structured textual representations, many real-world queries depend on acoustic evidence that is better preserved in continuous representations or raw audio.
HARP (Hybrid Audio Retrieval Pipeline) is an agentic framework and benchmark for systematically studying retrieval and evidence representations in long-audio analysis.
HARP extracts a searchable index from raw audio and lets an LLM agent plan retrieval over it, answering queries that mix **time**, **speaker**, **content**, and **domain info**. The same structure drives an evaluation agent that retrieves ground truth and scores open-ended outputs.


### Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e .

export HARP_DATA_ROOT=/path/to/datasets   # raw corpora
export HARP_OUT_ROOT=/path/to/derived     # dataprep output; defaults to HARP_DATA_ROOT
export HF_TOKEN=...                       # pyannote diarization
export GEMINI_API_KEY=...                 # default judge
```

Run all commands from the repo root. Two optional domain experts need artifacts that are not on PyPI or Hugging Face — a CoughKit checkout, a CSER checkpoint; see [`extract/README.md`](extract/README.md).

### Pipeline

```bash
# 1. extract: diarization + ASR + embeddings, then domain labels
python scripts/run_extract.py conf/emotion_hybrid_all.yaml --stage 1
python scripts/run_extract.py conf/emotion_hybrid_all.yaml --stage 2

# 2. infer: plan → retrieve → gather → answer
python scripts/run_inference.py conf/emotion_hybrid_all.yaml [--resume]

# 3. evaluate: judge predictions against ground truth
python scripts/run_eval.py conf/emotion_hybrid_all.yaml [--resume]

# 4. aggregate scores by query type
python scripts/run_score.py data/mspemotion
```

For a local backbone, start vLLM first and point `llm.base_url` at it: `scripts/serve_vllm.sh [text|omni]`.

Retrieval setups (`text`, `embed`, `hybrid`, `hybrid_all`, `hybrid_audio`) are two keys in the config — see [`conf/README.md`](conf/README.md). The scores this produces are defined in [`evaluate/README.md`](evaluate/README.md).

### Layout

Each directory has its own README with the details.

| Directory | Contents |
|---|---|
| [`conf/`](conf/README.md) | One YAML per task; every knob documented |
| [`extract/`](extract/README.md) | Two-stage indexing; audio experts (generic) and domain experts (per task) |
| [`inference/`](inference/README.md) | `HARPAgent`, retrieval tools, LLM backbones, task prompts |
| [`evaluate/`](evaluate/README.md) | Judges, metrics, ground-truth retrieval |
| [`human_eval/`](human_eval/README.md) | Self-contained HTML annotation UI + scoring |
| [`dataset/`](dataset/README.md) | `EpisodeDataset` loaders and GT processing |
| [`dataprep/`](dataprep/README.md) | Building the corpora and queries from raw sources |
| [`scripts/`](scripts/README.md) | Entry points and Slurm job files |

Each README ends with how to extend that part — a new expert, dataset, backbone, judge or task.

### Data

[`dataprep/`](dataprep/README.md) rebuilds each corpus and its queries in two stages: `audio` downloads the corpus and crops it into the audio HARP runs on, `queries` regenerates the query JSON. Both are deterministic.

```bash
huggingface-cli download cjli/harp --local-dir $HARP_OUT_ROOT   # music + health queries, CSER ckpt
dataprep/run_music.sh audio --src /raw/songeval --out $HARP_OUT_ROOT/songeval
cat $HARP_OUT_ROOT/songeval/queries/*.jsonl > data/songeval/queries.jsonl
```

SongEval and MedMosaic download automatically. MSP needs a signed academic licence and three separate downloads, and its queries are not redistributed — regenerate them locally; see [`dataprep/emotion/`](dataprep/emotion/README.md).

### Citing

```bibtex
@article{li2026harp,
  title   = {{HARP}: Agentic Hybrid Retrieval and Analysis for Long-Form Audio},
  author  = {Li, Chin-Jou and Someki, Masao and Jin, Woojeong and
             Siriwardena, Yashish M. and Laud, Tanmay and Puri, Shanil and
             Watanabe, Shinji},
  journal = {arXiv preprint arXiv:2609.14116},
  year    = {2026}
}
```

The paper evaluates on three corpora — please also cite any you use. BibTeX is in each task's preparation README: [MSP-Podcast + MSP-Conversation](dataprep/emotion/README.md#citation), [SongEval](dataprep/music/README.md#citation), [MedMosaic](dataprep/health/README.md#citation).
