# Data preparation

Gets each corpus onto disk and rebuilds its query set. Two stages, run in order:

| Stage | Does | Needed |
|---|---|---|
| `audio` | Downloads the corpus and merges/crops it into the audio HARP runs on | once per dataset |
| `queries` | Regenerates the query JSON from that audio | only to change the benchmark |

Prepared queries for music and health are published at [cjli/harp](https://huggingface.co/cjli/harp), so most users download those, run `audio`, and stop. The `audio` stage is deterministic, so the audio it produces matches ours.

```bash
uv pip install -e ".[prepare]"    # pydub, mutagen, pandas; also needs ffmpeg
huggingface-cli download cjli/harp --local-dir $HARP_OUT_ROOT

dataprep/run_music.sh   audio --src /raw/songeval   --out $HARP_OUT_ROOT/songeval
dataprep/run_health.sh  audio --src /raw/medmosaic  --out $HARP_OUT_ROOT/medmosaic
dataprep/run_emotion.sh audio --src /raw/msp        --out $HARP_OUT_ROOT/longemo_dataset
```

Each runner takes `audio`, `queries`, or `all`, and forwards `--src` / `--out` to every step. Individual scripts take the same flags and can be run alone.

| Task | Corpus | Licence | Details |
|---|---|---|---|
| music | SongEval | CC BY-NC-SA 4.0 | [`music/`](music/README.md) |
| health | MedMosaic | CC BY 4.0 | [`health/`](health/README.md) |
| emotion | MSP-Podcast + MSP-Conversation + NaturalVoices | academic licence, signed | [`emotion/`](emotion/README.md) |

Music and health download automatically. MSP cannot: it needs three separate signed downloads, and its queries are **not** redistributed — you regenerate them locally. See [`emotion/`](emotion/README.md).

## Paths

`--src` is the raw corpus, read-only. `--out` is where everything generated goes. Splitting them keeps a download pristine and lets you drop the released JSON into `--out` before running `audio`.

| | Resolution order |
|---|---|
| `--src` | flag → `$HARP_SRC_ROOT/<dataset>` → `$HARP_DATA_ROOT/<dataset>` |
| `--out` | flag → `$HARP_OUT_ROOT/<dataset>` → same as `--src` |

`<dataset>` is `longemo_dataset`, `songeval`, or `medmosaic`. The pipeline configs read the same two roots, so `--out` lines up with what `run_extract.py` later expects. Resolution lives in `paths.py`.

## Re-running

Runners are safe to re-run. Most scripts resume by themselves — they take `--rerun`, or check each episode, track, or clip before writing — so the runner just invokes them. The few that would redo expensive work or overwrite good output (the downloads, `make_jsonl`, `find_engsong`, `make_tracks`, and the two clip-index builds) are guarded on the file they produce; delete that file to force the step again.

## Feeding queries into a run

Preparation writes queries next to the data, one file per episode or track. The pipeline wants a single file named by `inference.queries`:

```bash
cat $HARP_OUT_ROOT/longemo_dataset/queries/*.jsonl > data/mspemotion/queries.jsonl
cat $HARP_OUT_ROOT/songeval/queries/*.jsonl        > data/songeval/queries.jsonl
cp  $HARP_OUT_ROOT/medmosaic/queries.jsonl           data/medmosaic/queries.jsonl
```

Point `extract.spk_audio_dir` and `extract.clip_audio_dir` at the `audio/` directory under `--out`, or copy it into the run directory.

To drop queries from *scoring* rather than from the run, list their IDs in `data/{run_id}/exclude.txt` (`#` comments allowed) — `run_score.py` and `human_eval/score_human.py` read it. Excluding at scoring time keeps predictions comparable across runs.

## Adding a task

Add a directory here with scripts taking `--src` / `--out` via `paths.add_path_args`, a `run_<task>.sh` with the same `audio` / `queries` / `all` stages, and a `README.md`. Then add an `EpisodeDataset` subclass in [`../dataset/`](../dataset/README.md) so extraction can read the result.
