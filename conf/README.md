# Configs

One YAML per task, in four sections read by the matching script: `dataset` and `extract` by `run_extract.py`, `inference` by `run_inference.py`, `evaluate` by `run_eval.py` (which also reads `dataset` for ground truth).

```
conf/
├── emotion_hybrid_all.yaml    # MSP-Podcast + MSP-Conversation
├── music_hybrid_all.yaml      # SongEval
└── health_hybrid_all.yaml     # MedMosaic long-form
```

Values may use `${VAR}`, expanded by `utils.load_config`; an unset variable raises rather than yielding a broken path.

Two roots keep raw and generated data apart, matching `dataprep`'s `--src` / `--out`: `${HARP_DATA_ROOT}` for the downloaded corpus (`msppodcast_full/`, `long_form/`) and `${HARP_OUT_ROOT}` for anything preparation produced (`songeval/tracks`, `test_episodes.jsonl`). `HARP_OUT_ROOT` defaults to `HARP_DATA_ROOT`, so leave it unset if you prepared data in place.

## Retrieval setups

The shipped configs are the **`hybrid_all`** setup. Two keys define all five: `inference.modalities` gates which retrieval tools the agent may call, and `inference.strip_domain` controls whether domain metadata is stripped from the evidence before the model sees it.

| Setup | `modalities` | `strip_domain` |
|---|---|---|
| **text** | `[time, keyword, label]` | `false` |
| **embed** | `[text_emb, audio_emb, domain_emb]` | `false` |
| **hybrid** | both of the above combined | `false` |
| **hybrid_all** | `hybrid + audio` | `false` |
| **hybrid_audio** | `hybrid + audio` | `true` |

`hybrid_all` and `hybrid_audio` retrieve identically and differ only in what the model may see — separating "the system found the right moment" from "the model can hear it".

Give each setup its own `predictions` filename, and keep `inference.predictions` and `evaluate.predictions` in sync.

### Modalities

| Name | Retrieves by | Requires |
|---|---|---|
| `time` | time window | — |
| `keyword` | transcript match | — |
| `label` | discrete domain label | a labeler at extract time |
| `text_emb` | transcript vector | `extract.text_emb` |
| `audio_emb` | generic audio vector | `extract.audio_emb` |
| `domain_emb` | task-specific vector | `extract.domain_embedders` |
| `audio` | attaches the raw clip | an audio-capable `llm` |

A vector modality also needs its embedder declared under `inference` to encode the *query* — omitting it disables that search even if listed. `health` has no `domain_emb` because MedMosaic ships no domain embedder.

`inference.oracle: true` swaps `HARPAgent` for `OracleAgent`, building the plan from ground truth. Pair it with `modalities: [time, audio]`.

## `dataset`

`_target_` picks the class (`dataset/longemo.py`, `songeval.py`, `medmosaic.py`); `data_dir` and `metadata_path` point at the source. Task-specific keys — `split`, `include_ct`, `streaming_json`, `episode_list`, `max_episodes`, `tracks` — are documented on each class.

## `extract`

| Key | Default | Description |
|---|---|---|
| `out_dir` | required | Run directory; holds `cache/`, `segments/`, `faiss/` |
| `batch_size` | `8` | Segments per batch |
| `max_seg_sec` | `null` | Split turns longer than this |
| `spk_audio_dir`, `clip_audio_dir` | — | Query-referenced clips (looked up under `<dir>/audio/`) |

Component slots, each a `_target_` block: `asr`, `diar`, `audio_emb`, optional `text_emb` (stage 1); `domain_labelers` and `domain_embedders`, both **lists** (stage 2). See [`../extract/README.md`](../extract/README.md).

## `inference`

| Key | Default | Description |
|---|---|---|
| `out_dir`, `queries`, `predictions` | required | Run directory and I/O |
| `task` | required | Selects `inference/prompts/{task}.py` |
| `audio_dir` | required | Episode audio for the `audio` modality |
| `llm` | required | Backbone; see [`../inference/README.md`](../inference/README.md) |
| `modalities` | all | Enabled tools |
| `strip_domain` | `false` | Drop domain metadata from evidence |
| `top_k` | `10` | Max evidence segments per set |
| `max_audio_clip_sec` | `120` | Cap on attached clip length |
| `oracle` | `false` | Use the GT-derived plan |
| `use_vec_cache` | `false` | Cache query embeddings across runs |
| `preprocess` | `false` | Re-run ASR / labeling on retrieved clips |
| `segments_dir`, `faiss_dir` | `<out_dir>/…` | Override index locations |
| `spk_index` | — | Speaker-clip index, used by `oracle` |

## `evaluate`

`out_dir`, `predictions` (match `inference.predictions`), `gt_dir`, `prompts` (e.g. `evaluate.prompts.emotion`), `metadata_path` for tasks that need it, and `closed_api: true` to default to the closed-API path.

The default judge is Gemini — a different family from the Qwen candidate, since models favour their own outputs:

```yaml
evaluate:
  judge:
    _target_: evaluate.judge.gemini.GeminiJudge
    model_id: gemini-3.1-flash-lite
```

`--judge` / `--judge-model` override it without editing the config. See [`../evaluate/README.md`](../evaluate/README.md).

## Adding a task

Add a dataset class, domain experts under `extract/domain_expert/{task}/`, and `inference/prompts/{task}.py` + `evaluate/prompts/{task}.py`. Then copy the nearest config, repoint the `_target_`s, and set `task:` and `prompts:`.
