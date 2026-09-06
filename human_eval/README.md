# Human evaluation

A human baseline for the same queries the model answers: `make_eval.py` builds a self-contained HTML annotation page, `score_human.py` scores the JSON it produces.

## Build a page

```bash
python human_eval/make_eval.py \
    --conf conf/emotion_hybrid_all.yaml \
    --episode MSP-PODCAST_0046 \
    --output human_eval/eval_html/eval_MSP-PODCAST_0046.html
```

Paths come from the config (`dataset.data_dir`, `inference.queries`, `inference.audio_dir`); `--flac-dir`, `--audio-dir`, and `--queries` override them. `--episode` takes a comma-separated list for a multi-episode page, and `--max-per-type` caps queries per `query_type`.

The output is one HTML file with the episode audio and every referenced clip inlined as base64 — no server, no assets directory. Annotators open it in a browser and it works offline. That also makes it large (often >100 MB), which is why `eval_html/` and `audio_patch/` are gitignored.

The page gives the annotator a seek bar with query timestamps ticked, a note-taking timeline, MCQ choices, and per-question confidence and difficulty ratings. Submitting downloads a JSON file.

## Inputs

| File | Contents |
|---|---|
| `query_options.json` | MCQ choices per task and query type |
| `instructions.json` | Task instructions shown in the page's modal |
| `emotion_episodes.json`, `health_episodes.json` | Which episodes make up each question set |

## Score

Collect the downloaded JSON files into `human_eval/outcome/`, then:

```bash
python human_eval/score_human.py                 # all outcome files
python human_eval/score_human.py emotion music   # filter by task
```

Each outcome file holds `answers` (one `{query_id, query_type, choice, sure, difficulty}` per query) and the annotator's `notes`. Scoring compares `choice` against the queries JSONL and reports accuracy by query type, honouring `data/{run_id}/exclude.txt` if present. `locate` answers are timestamps, matched within `LOCATE_TOL` seconds.

Run from the repo root — the paths in `score_human.py` are relative to it.
