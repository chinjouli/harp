# Query design

## Pipeline overview (episode-by-episode)

1. **Precompute audio pools** — done once across all test episodes
   - `getspk_example.py` — for each global speaker, extract the top-5 clean
     single-speaker clips from other episodes (SNR ≥ 20, 3–8 s, pure speech)
     → `audio/speaker_01.wav …`; index saved to `spk_clip_index.json`
   - `getemo_example.py` — for each non-neutral emotion code, extract 20 clips
     from other test episodes, diverse speakers, high cf
     → `audio/emotion_01.wav …`; index saved to `emo_clip_index.json`

2. **Per-episode candidate finding** (`find_candidates.py`)
   - `st_event`: non-N/X label, cf ≥ 0.50, single speaker, ≥ 2 context segs
   - `ct_trend`: 30 s window, clear directional change in one AVD dim,
     single speaker dominant; also one flat-window candidate per dim
   - `multi_spk_window`: ≥ 2 speakers, inter-speaker dim diff > 0.10

3. **Query type assignment** (`assign_queries.py`) — up to `cap` per type per episode

4. **Query generation** (`generate_queries.py`) — fill template, attach ground
   truth and judge info; for `locate` queries, also pick and store audio clip names

---

## Query types

### state — (speaker, content, time) → emotion

- Candidate shape: `st_event`
- Constraints: cf ≥ 0.50; N and X included
- Query: "Around [time], Speaker X was talking about [topic] and mentioned
  '[quote]'. What emotion(s) does Speaker X show?"
- Ground truth: `{emo, cf, votes, secondary, act, val, dom}`
- Judge: primary label match; partial credit for semantically close secondaries

### change — (speaker, time_range) → trend direction

- Candidate shape: `ct_trend`
- Constraints: half-split |Δ| > 0.08, single speaker dominant
- Includes one flat-window candidate per dim (`direction: "none"`) — 2 of the
  cap slots are reserved for flat windows
- Query: "From [t0] to [t1] seconds, how did Speaker X's [dim] change while
  discussing [topic]?"
- Ground truth: `{direction, delta, half1, half2}`
- Judge: direction correct; half1/half2 verify magnitude

### comparison — (time_range, topic) → speaker

- Candidate shape: `multi_spk_window`
- Constraints: ≥ 2 speakers each with > 10 s coverage; inter-speaker diff > 0.10
  on the strongest dim
- Query: "From [t0] to [t1] seconds discussing [topic], which speaker had higher
  [dim] — Speaker X or Speaker Y?"
- Ground truth: `{winner, dim, spk_val}`
- Judge: speaker ID correct (hard); per-speaker means verify direction (soft)

### locate — (speaker_example, emotion_example, time_range) → time + content

- Candidate shape: `st_event` (cf ≥ 0.50, non-neutral)
- Two audio clips included per query:
  - **Speaker clip** (`audio_spk`): one of the 5 precomputed clips for that
    global speaker, from a different episode
  - **Emotion clip** (`audio_emo`): one of the 20 precomputed clips for that
    emotion code, from a different test episode and a different global speaker
- Query: "Speaker X: [audio_spk]. Between [t0] and [t1] seconds, when did
  Speaker X show the same emotion as in [audio_emo], and what were they
  saying?"
- Ground truth: `{time_ref, t0, t1, quote, emo}`
- Judge: time within ±15 s; quote overlaps the annotated segment

---

## Query record schema

```json
{
  "id": 0,
  "episode": "MSP-PODCAST_0002",
  "query_type": "state | change | comparison | locate",
  "query": "...",
  "time_ref": 183.5,
  "topic": "work stress",
  "audio_spk":  "speaker_03.wav",
  "audio_spk2": "speaker_07.wav",
  "audio_emo":  "emotion_05.wav",
  "ground_truth": {},
  "judge_info": {}
}
```

- `audio_spk` — present for **all** query types (speaker identity clip)
- `audio_spk2` — present only for `comparison` (the second speaker)
- `audio_emo` — present only for `locate` (emotion example clip)

---

## Audio pool index schemas

**`spk_clip_index.json`**
```json
{
  "12345": [
    {"wav": "speaker_01.wav", "ep": "MSP-PODCAST_0003",
     "start": 42.1, "end": 47.8, "snr": 28.3}
  ]
}
```
Key is the global speaker ID (string int). Up to 5 entries per speaker, sorted
by SNR descending.

**`emo_clip_index.json`**
```json
{
  "H": [
    {"wav": "emotion_01.wav", "ep": "MSP-PODCAST_0005",
     "global_spk": 67890, "cf": 0.82, "start": 12.4, "end": 17.1}
  ]
}
```
Keys are emotion codes (H/A/S/U/F/D/C/O — no N or X). Up to 20 entries per
code, from test episodes only, sorted by cf descending.
