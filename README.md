# HARP

Hybrid audio RAG over long recordings (10 min – 2 h).

HARP answers complex queries that mix four elements — **time**, **speaker/singer**, **content**, and **domain info** (paralinguistic cues) — by extracting a searchable index from raw audio and letting an LLM agent plan retrieval over it. The same structure drives an evaluation agent that retrieves ground truth and scores open-ended outputs.

Three tasks ship: **emotion recognition** (MSP-Podcast + MSP-Conversation), **music evaluation** (SongEval), and **healthcare conversation** (MedMosaic).

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e .

export HARP_DATA_ROOT=/path/to/datasets   # raw corpora
export HARP_OUT_ROOT=/path/to/derived     # dataprep output; defaults to HARP_DATA_ROOT
export HF_TOKEN=...                       # pyannote diarization
export GEMINI_API_KEY=...                 # default judge
```

Two domain experts need artifacts that are not on PyPI or Hugging Face — a CoughKit checkout for health, a CSER checkpoint for emotion. Both are optional; see [`extract/README.md`](extract/README.md).

Run all commands from the repo root.

## Pipeline

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

## Layout

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
| `scripts/` | Entry points; `slurm/` holds site-specific job files |

Run artifacts live under `extract.out_dir` (e.g. `data/mspemotion/`) and are gitignored: `cache/` from stage 1, `segments/` and `faiss/` from stage 2, plus `queries.jsonl`, `predictions*.jsonl`, and `scores*.jsonl`.

## Data preparation

Corpora are not redistributed here. [`dataprep/`](dataprep/README.md) has one runner per task with two stages, run in order:

```bash
uv pip install -e ".[prepare]"
dataprep/run_music.sh audio   --src /data/songeval --out /derived/songeval  # download + build
dataprep/run_music.sh queries --src /data/songeval --out /derived/songeval  # rebuild the JSON
```

`audio` downloads the corpus and merges/crops it into the exact audio HARP runs on; it is deterministic, so you get the same audio we did. `queries` regenerates the query JSON from that audio — only needed to change the benchmark, since the JSON is released.

SongEval (CC BY-NC-SA 4.0) and MedMosaic (CC BY 4.0) download from Hugging Face automatically; [MSP-Podcast](https://lab-msp.com/MSP/MSP-Podcast.html) and [MSP-Conversation](https://lab-msp.com/MSP/MSP-Conversation.html) need a signed academic licence and must be unpacked by hand.

Prepared queries for music and health, plus the CSER checkpoint, are published at [cjli/harp](https://huggingface.co/cjli/harp):

```bash
huggingface-cli download cjli/harp --local-dir $HARP_OUT_ROOT
```

The emotion queries are not: the MSP licence forbids redistribution and the queries quote transcript. Regenerate them with `run_emotion.sh all` once you hold MSP — generation is seeded and deterministic, so you get the same benchmark. Emotion needs three downloads — NaturalVoices for audio and transcripts, MSP-Podcast and MSP-Conversation for the human labels; [`dataprep/emotion/`](dataprep/emotion/README.md) has the layout and says which labels back each query type.

Prepared queries are concatenated into the run directory before inference:

```bash
cat /derived/songeval/queries/*.jsonl > data/songeval/queries.jsonl
```

## Retrieval setups

The configs ship as **`hybrid_all`**. The other setups come from editing `inference.modalities` and `inference.strip_domain`:

| Setup | Retrieval | Evidence shown |
|---|---|---|
| `text` | keyword search + label lookup | metadata |
| `embed` | vector search | metadata |
| `hybrid` | keyword **and** vector search | metadata |
| `hybrid_all` | hybrid search | metadata **and** audio |
| `hybrid_audio` | hybrid search | audio only |

`inference.oracle: true` builds the plan from ground truth instead — an upper bound on answering when retrieval is perfect. See [`conf/README.md`](conf/README.md).

## Scoring

Predictions are open-ended, so an LLM judge scores them. Because the `plan` and `evidence_sets` are saved with each answer, retrieval is scored separately from answering.

| Score | Type | Question |
|---|---|---|
| `answer` | 0/1 | Does the prediction match the GT answer? |
| `factual` | 0/1 | Are the rationale's claims consistent with GT evidence? |
| `faithful` | 0/1 | Does the rationale stay grounded in the retrieved evidence? |
| `rationale` | 0/1 | `factual & faithful` |
| `retrieval_hit_rate` | float | GT evidence segments the retrieval found |
| `plan_hit_rate` | float | GT evidence segments the plan's windows covered |

`run_score.py` aggregates by query type. Use a judge from a different family than the candidate; `run_judgeeval.py` reports inter-judge agreement.

## Config convention

Plain YAML with `_target_` for dynamic class loading (no Hydra); `utils.build_from_cfg` instantiates recursively, passing the remaining keys as constructor kwargs.

```yaml
extract:
  asr:
    _target_: extract.audio_expert.asr.whisper.WhisperASR
    model_name: large-v3-turbo
```

## Extending

- **ASR / diarizer / embedder** → subclass a base in `extract/audio_expert/base.py`, drop the file in the matching subdirectory.
- **New task** → domain experts under `extract/domain_expert/{task}/`, plus `inference/prompts/{task}.py` and `evaluate/prompts/{task}.py`.
- **New dataset** → subclass `EpisodeDataset`.
- **New LLM or judge** → satisfy `LLMBase` / `JudgeBase`, swap `_target_`.

## Citing

HARP builds on three corpora; please cite whichever you use. BibTeX lives in each task's preparation README — [MSP-Podcast + MSP-Conversation](dataprep/emotion/README.md#citation), [SongEval](dataprep/music/README.md#citation), [MedMosaic](dataprep/health/README.md#citation).

## Slurm

`scripts/slurm/` holds job files for a GPU cluster. Set `--partition` and `--account` for your site before use — they ship as `CHANGE_ME`. A starting point, not a portable interface.

```bash
sbatch scripts/slurm/extract.sbatch conf/emotion_hybrid_all.yaml 1
sbatch scripts/slurm/infer.sbatch conf/emotion_hybrid_all.yaml --resume
sbatch scripts/slurm/serve_vllm.sbatch omni
```
