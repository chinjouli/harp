# Query design

## Overview

One query type: **compare** — given two songs in the same track, which has
better overall aesthetics?

Songs are identified by one of three reference methods. Each query uses the
same method for both songs (no mixing).

**Scale**: 50 tracks × 10 queries = 500 queries total. Each track has 20
songs; 10 queries → 10 non-overlapping pairs (each song appears exactly
once per track). Split per track: **3 time + 3 position + 4 audio**.

---

## Query type: compare

**Task**: "Which of the two songs sounds better overall?"

**Ground truth**: grand mean across 4 raters × 5 dimensions (Coherence,
Musicality, Memorability, Clarity, Naturalness).

| `winner` | condition |
|----------|-----------|
| `"a"` / `"b"` | `|mean_a - mean_b| ≥ 0.25` |
| `"tie"` | `|mean_a - mean_b| < 0.25` |

`delta` (= `mean_a - mean_b`, signed) is stored so the judge can flag cases
where the model expresses strong confidence in one direction but the true
difference is small (near-tie region).

**Judge**:
- Hard score: predicted winner matches `ground_truth.winner` (tie counts
  as correct only if model also says tie or expresses uncertainty).
- Soft check: `delta` and per-dimension means available to verify whether
  a claimed "strong win" is supported by the data.

---

## Reference methods

### 1. time — rough timestamp

Songs are ~2.5–5 min long. A time reference is a whole or half minute
(e.g., "around 3:00" or "around 3:30"), always a multiple of 30 s. The
referenced song is whichever song contains that timestamp; any point ±30 s
around the ref is guaranteed to fall within the same song.

- `time_ref`: float (seconds), multiple of 30
- Query phrasing: "Around [m:ss] in the track, there is a song. …"

### 2. position — ordinal index in the track

Songs are numbered from 1 in playback order ("the 3rd song in the track").

- `position`: int (1-based)
- Query phrasing: "The [N]th song in the track …"

### 3. audio — 5-second sample clip

A 5 s clip from the interior of the song (drawn from within the same
track), used as an audio fingerprint for retrieval. Clips avoid the 3 s
crossfade boundaries (at least 5 s from `start_time` and `end_time`).

- `audio_clip`: filename of the 5 s wav, e.g. `clip_042a.wav`
- Clip index stored in `clip_index.json` (see below)
- Query phrasing: "Here is a short clip: [audio_clip]. …"

---

## Query record schema

```json
{
  "id": 0,
  "track": "track_03",
  "ref_method": "time | position | audio",
  "query": "...",
  "song_a": {
    "file_name": "mp3/1234.mp3",
    "time_ref": 210.0,
    "position": null,
    "audio_clip": null
  },
  "song_b": {
    "file_name": "mp3/5678.mp3",
    "time_ref": 390.0,
    "position": null,
    "audio_clip": null
  },
  "ground_truth": {
    "winner": "a | b | tie",
    "delta": 0.65,
    "mean_a": 4.25,
    "mean_b": 3.60
  },
  "judge_info": {
    "scores_a": {"Coherence": 4.5, "Musicality": 4.0,
                 "Memorability": 4.5, "Clarity": 4.0, "Naturalness": 4.25},
    "scores_b": {"Coherence": 3.5, "Musicality": 3.75,
                 "Memorability": 3.5, "Clarity": 3.5, "Naturalness": 3.75}
  }
}
```

- Exactly one of `time_ref`, `position`, `audio_clip` is non-null per song,
  matching `ref_method`.
- `delta` = `mean_a - mean_b` (positive → A wins, negative → B wins).
- `scores_a` / `scores_b` give per-dimension means across raters.

---

## Clip index schema (`clip_index.json`)

```json
{
  "clip_042a.wav": {
    "track": "track_03",
    "file_name": "mp3/1234.mp3",
    "start": 45.0,
    "end": 50.0
  }
}
```

Each clip is 5 s long (`end - start = 5`), drawn from within the same
track as the query, at least 5 s from both `start_time` and `end_time`
boundaries of its song.
