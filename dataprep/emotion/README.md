# Emotion — MSP-Podcast + MSP-Conversation

```bash
dataprep/run_emotion.sh audio   --src /raw/msp --out /derived/longemo_dataset
dataprep/run_emotion.sh queries --src /raw/msp --out /derived/longemo_dataset
```

Query types and event shapes: [`query_design.md`](query_design.md).

## Getting the corpus

Three separate downloads, each under its own signed academic licence. NaturalVoices supplies the audio and transcripts; the two v2.0 releases supply the human labels. `run_emotion.sh audio` prints these instructions if `msppodcast_full/` is missing.

| Source | Provides |
|---|---|
| [NaturalVoices](https://lab-msp.com/NaturalVoices/) | `all_data.json` (1.7 GB) — transcripts, timing, diarization; `podcasts_flac/` |
| [MSP-Podcast](https://lab-msp.com/MSP/MSP-Podcast.html) v2.0 | `Labels/labels_{consensus,detailed}.csv`, `Partitions.txt` |
| [MSP-Conversation](https://lab-msp.com/MSP/MSP-Conversation.html) v2.0 | `partitions.txt`, `Time_Labels/`, `Annotations/{Arousal,Valence,Dominance}/` |

Arrange all three under `<src>/msppodcast_full/`, keeping the directory names above.

## audio

| # | Script | Writes |
|---|---|---|
| 1 | `make_jsonl.py` | `jsons/{episode}.jsonl` |
| 2 | `build_test_episodes.py` | `src_dataset/test_episodes.jsonl` |
| 3 | `getspk_example.py --build-index --extract` | `spk_clip_index.json`, `audio/speaker_*.wav` |
| 4 | `getemo_example.py --build-index --extract` | `emo_clip_index.json`, `audio/emotion_*.wav` |

Step 2 keeps the 40 of 58 MSP-Conversation Test1 episodes with little training-set overlap, and writes the `episode_list` the configs point at. Steps 3 and 4 pick the highest-SNR clips per speaker and per emotion, then cut them; drop `--build-index` to reuse an existing index.

## queries

| # | Script | Writes |
|---|---|---|
| 1 | `find_candidates.py` | `candidates/{episode}.json` |
| 2 | `assign_queries.py` | `assignments/{episode}.json` |
| 3 | `generate_queries.py` | `queries/{episode}.jsonl` |

All three take `--episode` for one episode and `--rerun` to overwrite.

**These queries are not redistributed.** The MSP licence forbids it, and 153 of the 466 embed verbatim transcript in `judge_info.seg_text` / `ground_truth.quote`, with 109 more quoting transcript in the query text. Regenerate them once you hold MSP: selection is a pure function of the corpus, `random` is seeded with 42, and both LLM calls decode greedily against a pinned `Qwen/Qwen3.5-0.8B`. Generated topic phrases can drift across transformers versions and hardware, which changes wording but not ground truth.

### Ground truth sources

`all_data.json`'s emotion fields are model output, not annotation, so ground truth uses the human labels — with one exception.

| Query type | n | Source |
|---|---|---|
| `state` | 109 | `labels_consensus.csv` (`EmoClass`, `EmoAct/Val/Dom`) + `labels_detailed.csv` votes → confidence |
| `locate` / `locate_hard` | 108 / 45 | same, plus transcript timing for the quote |
| `change` | 118 | MSP-Conversation `Annotations/{dim}/*.csv`, `Mean` /100 |
| `comparison` | 86 | **`all_data.json`** — automatic per-segment `arousal`/`valence`/`dominance` |

`comparison` averages automatic AVD per speaker over a 60 s window (`find_multi_spk_windows`), because the human labels are per-utterance and miss speakers in some windows. 

AVD from `labels_consensus.csv` is 1–7 SAM, normalised `(x - 1) / 6`; traces are 0–100, divided by 100.

## Citation

```bibtex
@article{11457331,
  author  = {Busso, Carlos and Lotfian, Reza and Sridhar, Kusha and Salman, Ali N. and
             Lin, Wei-Cheng and Goncalves, Lucas and Parthasarathy, Srinivas and
             Naini, Abinay Reddy and Leem, Seong-Gyun and Martinez-Lucas, Luz and
             Chou, Huang-Cheng and Mote, Pravin},
  journal = {IEEE Transactions on Affective Computing},
  title   = {The {MSP-Podcast} Corpus},
  year    = {2026},
  volume  = {17},
  number  = {3},
  pages   = {3065-3083},
  doi     = {10.1109/TAFFC.2026.3678489}
}

@article{martinez2026msp,
  title   = {{MSP-Conversation}: A Corpus for Naturalistic, Time-Continuous Emotion Recognition},
  author  = {Martinez-Lucas, Luz and Mote, Pravin and Naini, Abinay Reddy and
             Abdelwahab, Mohammed and Busso, Carlos},
  journal = {arXiv preprint arXiv:2603.22536},
  year    = {2026}
}
```
