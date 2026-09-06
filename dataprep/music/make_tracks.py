#!/usr/bin/env python3
"""
Stitch 1006 filtered English songs into 50 ~1h mix tracks.
Songs are stratified by overall avg score (buckets 1-5) so each
track gets an even score distribution. Transitions use a 3s crossfade.
"""
import argparse
import json
import random
import mutagen.mp3
from pathlib import Path

from dataprep import paths
from collections import defaultdict
from pydub import AudioSegment

CROSSFADE_MS = 3000
N_TRACKS     = 50
SEED         = 42
SCORE_KEYS   = ["Coherence", "Musicality", "Memorability", "Clarity", "Naturalness"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    paths.add_path_args(ap)
    src, out = paths.resolve(ap.parse_args(), "songeval")
    OUT_DIR = out / "tracks"
    OUT_DIR.mkdir(exist_ok=True)

    # --- Load filtered songs with scores and durations ---
    rows = [json.loads(l) for l in open(out / "languages2.jsonl", encoding="utf-8")]
    selected = {
        r["file"] for r in rows
        if r.get("lang2", r["lang"]) == "en" and r.get("prob2", r["prob"]) > 0.5
    }

    meta = {
        Path(m["file_name"]).name: m
        for l in open(src / "metadata.jsonl", encoding="utf-8")
        for m in [json.loads(l)]
    }

    songs = []  # (file, avg_score, per-dim scores)
    for fname in selected:
        if fname not in meta:
            continue
        dur = mutagen.mp3.MP3(str(src / "mp3" / fname)).info.length
        if not (150 <= dur <= 300):
            continue
        ann = meta[fname]["annotation"]
        dim = {k: sum(a[k] for a in ann) / len(ann) for k in SCORE_KEYS}
        avg = sum(dim.values()) / len(dim)
        songs.append({"file": fname, "avg": avg, "dim": dim})

    # --- Bucket by rounded score (1-5) ---
    random.seed(SEED)

    buckets = defaultdict(list)
    for s in songs:
        bucket = max(1, min(5, round(s["avg"])))
        buckets[bucket].append(s)

    # Score bucket sizes: {1: 50, 2: 254, 3: 236, 4: 378, 5: 88}
    print("Score bucket sizes:", {k: len(v) for k, v in sorted(buckets.items())})

    # --- Stratified assignment: round-robin across buckets into N_TRACKS ---
    for v in buckets.values():
        random.shuffle(v)

    tracks = [[] for _ in range(N_TRACKS)]
    bucket_iters = {k: iter(v) for k, v in buckets.items()}
    # flatten buckets in interleaved order so each track gets even distribution
    pool = []
    while any(bucket_iters.values()):
        for k in sorted(bucket_iters):
            try:
                pool.append(next(bucket_iters[k]))
            except StopIteration:
                del bucket_iters[k]

    for i, song in enumerate(pool):
        tracks[i % N_TRACKS].append(song)

    # --- Build each track ---
    for t_idx, track_songs in enumerate(tracks):
        random.shuffle(track_songs)

        track_name = f"track_{t_idx:02d}"
        out_path   = OUT_DIR / f"{track_name}.mp3"
        skip_audio = out_path.exists()

        cursor_ms = 0
        if not skip_audio:
            audio = AudioSegment.empty()

        for song in track_songs:
            seg    = AudioSegment.from_mp3(str(src / "mp3" / song["file"]))
            dur_ms = len(seg)

            if not skip_audio:
                if len(audio) == 0:
                    audio = seg
                else:
                    audio = audio.append(seg, crossfade=CROSSFADE_MS)
                    cursor_ms -= CROSSFADE_MS // 2

            elif song is not track_songs[0]:
                # mirror the cursor adjustment without loading audio
                cursor_ms -= CROSSFADE_MS // 2

            end_ms = cursor_ms + dur_ms
            song["start_ms"] = cursor_ms
            song["end_ms"]   = end_ms
            cursor_ms = end_ms - CROSSFADE_MS // 2

        if not skip_audio:
            audio.export(out_path, format="mp3", bitrate="192k")
            total_min = len(audio) / 60000
            print(f"{track_name}: {len(track_songs)} songs, {total_min:.1f} min")
        else:
            print(f"{track_name}: skipped (mp3 exists), boundaries recalculated")

    out_jsonl = open(out / "tracks_metadata.jsonl", "w", encoding="utf-8")
    for t_idx, track_songs in enumerate(tracks):
        track_name = f"track_{t_idx:02d}"
        for song in track_songs:
            m = meta[song["file"]]
            entry = {
                "track_name": track_name,
                "file_name":  m["file_name"],
                "start_time": round(song["start_ms"] / 1000, 3),
                "end_time":   round(song["end_ms"]   / 1000, 3),
                "gender":     m["gender"],
                "annotation": m["annotation"],
            }
            out_jsonl.write(json.dumps(entry, ensure_ascii=False) + "\n")
    out_jsonl.close()
    print(f"\nDone. {N_TRACKS} tracks saved to {OUT_DIR}/")
    print("Metadata saved to tracks_metadata.jsonl")


if __name__ == "__main__":
    main()
