# Music — SongEval

```bash
dataprep/run_music.sh audio   --src /raw/songeval --out /derived/songeval
dataprep/run_music.sh queries --src /raw/songeval --out /derived/songeval
```

Design notes: [`query_design.md`](query_design.md). Prepared queries are published at [cjli/harp](https://huggingface.co/cjli/harp), so the `queries` stage is only needed to change the benchmark.

## Getting the corpus

[ASLP-lab/SongEval](https://huggingface.co/datasets/ASLP-lab/SongEval), CC BY-NC-SA 4.0. The `audio` stage downloads it, fetching only `metadata.jsonl` and `mp3/`; override the repo with `$SONGEVAL_REPO`.

## audio

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `find_engsong.py` | `mp3/` | `languages.jsonl`, `languages2.jsonl` |
| 2 | `make_tracks.py` | `languages2.jsonl`, `metadata.jsonl`, `mp3/` | `tracks/*.mp3`, `tracks_metadata.jsonl` |
| 3 | `extract_clips.py` | `tracks/`, `tracks_metadata.jsonl` | `audio/snippet_*.wav`, `clip_index.json` |

Step 1 runs Whisper language ID over every mp3 and is by far the slowest part — drop in the released `languages2.jsonl` and start at step 2 to skip it. Step 2 stitches 1006 English songs into 50 hour-long medleys with 3 s crossfades. All three are deterministic (`SEED = 42`), so the medleys and snippet offsets match ours exactly.

## queries

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `find_pairs.py` | `tracks_metadata.jsonl` | `pairs/track_NN.json` |
| 2 | `generate_queries.py` | `pairs/`, `clip_index.json` | `queries/track_NN.jsonl` |

Both take `--track` and `--rerun`. 500 queries: 10 per track, split 3 time + 3 position + 4 audio, each song used once per track.

## Citation

```bibtex
@article{yao2025songeval,
  title   = {{SongEval}: A Benchmark Dataset for Song Aesthetics Evaluation},
  author  = {Yao, Jixun and Ma, Guobin and Xue, Huixin and Chen, Huakang and
             Hao, Chunbo and Jiang, Yuepeng and Liu, Haohe and Yuan, Ruibin and
             Xu, Jin and Xue, Wei and others},
  journal = {arXiv preprint arXiv:2505.10793},
  year    = {2025}
}
```
