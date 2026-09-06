# Dataset

Loaders that feed prepared corpora into the pipeline.

| File | Purpose |
|---|---|
| `base.py` | `EpisodeDataset` — the interface `run_extract.py` consumes |
| `longemo.py` | MSP-Podcast + MSP-Conversation (`LongEmoDataset`) |
| `songeval.py` | SongEval medleys (`SongEvalDataset`) |
| `medmosaic.py` | MedMosaic long-form (`LongFormDataset`) |
| `process_gt_emotion.py`, `process_gt_music.py` | Convert ground-truth annotations into the segment schema the extractor emits, so the retriever reads either path unchanged |

A loader yields one record per episode — id, audio path, and whatever metadata the task needs. Point `dataset._target_` at the class; remaining YAML keys become constructor arguments. See [`../conf/README.md`](../conf/README.md) for those keys and [`../extract/README.md`](../extract/README.md) for the segment schema.

Building a corpus and its queries from raw sources is a separate stage: [`../dataprep/`](../dataprep/README.md). A loader over an existing corpus is enough to run HARP; preparation code is only needed to rebuild or extend one.
