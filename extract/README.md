# Extract

Turns raw audio into a searchable index: one JSONL of segments per episode plus dataset-level FAISS indexes.

```bash
python scripts/run_extract.py conf/emotion_hybrid_all.yaml --stage 1
python scripts/run_extract.py conf/emotion_hybrid_all.yaml --stage 2
```

Omitting `--stage` runs both.

## Two stages

The split exists because the halves differ in cost and in how often they change.

**Stage 1 — generic audio processing** (`extract_episode_stage1`). Diarize, merge and split turns, ASR each turn, embed speaker and text. Task-agnostic and expensive; the same cache serves every task and setup. Writes `cache/{episode_id}_turns.jsonl`, `_spk.npy`, and `_text.npy`.

**Stage 2 — domain experts** (`extract_episode_stage2`). Reads the cache, runs the task's labelers and embedders, writes `segments/{episode_id}.jsonl` and appends to `faiss/`. Cheap to re-run, which is the point: swapping an emotion labeler never re-runs Whisper.

FAISS indexes are dataset-level and append-only (`utils.append_faiss`) — `audio.index`, `text.index`, and one `domain_{i}.index` per domain embedder.

Both stages skip episodes whose output exists, so re-running resumes. To re-run one component against an existing cache:

```bash
python scripts/run_extract.py conf/... --stage 1 --embedder text
python scripts/run_extract.py conf/... --stage 2 --labeler cser
```

`--labeler` matches a substring of the labeler's `_target_`; `--embedder` supports only `text`.

## Segment schema

```jsonc
{
    "seg_id": "MSP-PODCAST_0001_ext_0003", "episode_id": "MSP-PODCAST_0001",
    "start": 12.4, "end": 17.1, "speaker": "SPEAKER_00", "text": "...",
    "domain_labels": { /* task-specific, opaque to the retriever */ },
    "audio_faiss_idx": 42, "text_faiss_idx": 42, "domain_faiss_idxs": [17]
}
```

`dataset/process_gt_*.py` emits the same schema from ground truth, so the retriever reads either path unchanged.

## Two kinds of expert

```
extract/
├── pipeline.py       # stage drivers
├── utils.py          # load_config, build_from_cfg, append_faiss
├── audio_expert/     # generic — any audio task
│   ├── base.py       # ASRBase, DiarBase, EmbBase, TextEmbBase
│   ├── asr/  diar/  emb/
└── domain_expert/    # task-specific — one per new task
    ├── base.py       # DomainLabelerBase, DomainEmbBase
    └── emotion/  music/  health/
```

`audio_expert/` is what makes sense for *any* long-audio task — transcription, speaker segmentation, general embeddings. You add here for a better ASR, not for a new task.

`domain_expert/` is the paralinguistic knowledge that defines a task. **Every new task needs one.**

Both are wired in by `_target_`; nothing imports them by name.

## Adding an audio expert

Subclass the matching class in `audio_expert/base.py` and drop the file in `asr/`, `diar/`, or `emb/`. The `*_batch` methods have loop defaults — override only if your model batches better. Remaining YAML keys become constructor kwargs:

```yaml
extract:
  asr:
    _target_: extract.audio_expert.asr.my_asr.MyASR
    model_name: ...
```

`DiarBase.segment` must return speaker IDs consistent across the whole episode. If there are no speakers to separate, a VAD is a valid `DiarBase` — see `diar/silero.py`, which labels everything `SPEAKER_00`, or `diar/mert.py`, which uses the same interface for musical section boundaries.

## External domain experts

Most experts load from Hugging Face automatically. Two do not, and both are optional — nothing else breaks if you skip the task that uses them.

**CoughKit** (`health/labeler_coughkit_count.py`) is not on PyPI. Clone it and install in place; the bundled XGBoost models resolve relative to the checkout, so an editable install is required:

```bash
git clone https://github.com/bagustris/coughkit
uv pip install -e ./coughkit
```

`CoughKitCounter` imports it lazily inside `__init__`, so the rest of `extract/` works without it.

**CSER** (`emotion/labeler_cser.py`) is a BiLSTM head over WavLM-Large, from [Berkeley-Speech-Group/emo-reasoning](https://github.com/Berkeley-Speech-Group/emo-reasoning). The architecture is reimplemented here, so you do not need to install that repo — only its trained checkpoint. WavLM itself is pulled from Hugging Face (`microsoft/wavlm-large`) on first use.

The shipped config reads it from `$HARP_CSER_CKPT`:

```yaml
extract:
  domain_labelers:
    - _target_: extract.domain_expert.emotion.labeler_cser.CSERLabeler
      ckpt_path: ${HARP_CSER_CKPT}
```

See the emo-reasoning repo for training the checkpoint yourself.

## Adding a domain expert (new task)

Create `domain_expert/{task}/` with a labeler, an embedder, or both — `health` ships with a labeler and none. See `domain_expert/base.py` for the two contracts.

`label()` returns whatever dict the task needs; it lands verbatim in `domain_labels` and only the task's prompt modules interpret it. See `emotion/labeler_ser.py` for a reference schema.

Both slots are **lists**:

```yaml
extract:
  domain_labelers:
    - _target_: extract.domain_expert.mytask.labeler_a.LabelerA
    - _target_: extract.domain_expert.mytask.labeler_b.LabelerB
  domain_embedders:
    - _target_: extract.domain_expert.mytask.embedder_x.EmbedderX
```

Labelers merge left-to-right — a later one's non-`None` values win, which is how emotion combines categorical `SERLabeler` with continuous `CSERLabeler`. Each embedder gets its own FAISS index.

Finish the task by adding `inference/prompts/{task}.py` and `evaluate/prompts/{task}.py`.
