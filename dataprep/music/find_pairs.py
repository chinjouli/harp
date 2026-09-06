"""Find and pair songs for each track; assign reference methods.

Reads tracks_metadata.jsonl, outputs pairs/track_NN.json.

10 queries per track: 3 time + 3 position + 4 audio.
Songs are shuffled then paired consecutively (each song used once).

Run from the repo root:
    python dataprep/music/find_pairs.py
    python dataprep/music/find_pairs.py --track track_00
    python dataprep/music/find_pairs.py --rerun
"""
import argparse
import json
import math
import random

from dataprep import paths

N_QUERIES   = 10
SEED        = 42
REF_METHODS = ["time"] * 3 + ["position"] * 3 + ["audio"] * 4
DIMS        = ("Coherence", "Musicality", "Memorability",
               "Clarity", "Naturalness")
TIE_THRESH  = 0.25

def _wire(args) -> None:
    """Resolve path constants from --src / --out."""
    src, out = paths.resolve(args, "songeval")
    globals().update(
        TRACKS_META=out / "tracks_metadata.jsonl",
        OUT_DIR=out / "pairs",
    )


def grand_mean(annotation):
    vals = [r[d] for r in annotation for d in DIMS]
    return sum(vals) / len(vals)


def dim_means(annotation):
    return {d: round(sum(r[d] for r in annotation) / len(annotation), 3)
            for d in DIMS}


def pick_time_ref(start, end):
    # 30s-aligned stamp where [ref-30, ref+30] ⊆ [start, end]
    lo   = math.ceil((start + 30) / 30) * 30
    hi   = math.floor((end   - 30) / 30) * 30
    opts = list(range(int(lo), int(hi) + 1, 30))
    return opts[len(opts) // 2]


def song_positions(track_songs):
    ordered = sorted(track_songs, key=lambda s: s["start_time"])
    return {s["file_name"]: i + 1 for i, s in enumerate(ordered)}


def make_pair(song_a, song_b, method, positions):
    mean_a = grand_mean(song_a["annotation"])
    mean_b = grand_mean(song_b["annotation"])
    delta  = round(mean_a - mean_b, 4)
    winner = ("tie" if abs(delta) < TIE_THRESH
              else ("a" if delta > 0 else "b"))

    sa = {"file_name":  song_a["file_name"],
          "start_time": song_a["start_time"],
          "end_time":   song_a["end_time"]}
    sb = {"file_name":  song_b["file_name"],
          "start_time": song_b["start_time"],
          "end_time":   song_b["end_time"]}

    if method == "time":
        sa["time_ref"] = pick_time_ref(song_a["start_time"],
                                       song_a["end_time"])
        sb["time_ref"] = pick_time_ref(song_b["start_time"],
                                       song_b["end_time"])
    elif method == "position":
        sa["position"] = positions[song_a["file_name"]]
        sb["position"] = positions[song_b["file_name"]]
    # audio: clip filenames resolved later in generate_queries.py

    return {
        "ref_method":   method,
        "song_a":       sa,
        "song_b":       sb,
        "ground_truth": {"winner": winner, "delta": delta,
                         "mean_a": round(mean_a, 4),
                         "mean_b": round(mean_b, 4)},
        "judge_info":   {"scores_a": dim_means(song_a["annotation"]),
                         "scores_b": dim_means(song_b["annotation"])},
    }


def process_track(track_name, track_songs, seed):
    rng       = random.Random(seed)
    songs     = list(track_songs)
    rng.shuffle(songs)
    positions = song_positions(track_songs)

    pairs = [
        make_pair(songs[i], songs[i + 1], REF_METHODS[i // 2], positions)
        for i in range(0, N_QUERIES * 2, 2)
    ]
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_path_args(ap)
    ap.add_argument("--track", default=None)
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()
    _wire(args)

    OUT_DIR.mkdir(exist_ok=True)

    songs    = [json.loads(l) for l in open(TRACKS_META, encoding="utf-8")]
    by_track = {}
    for s in songs:
        by_track.setdefault(s["track_name"], []).append(s)

    tracks = [args.track] if args.track else sorted(by_track)
    if not args.rerun:
        tracks = [t for t in tracks
                  if not (OUT_DIR / f"{t}.json").exists()]

    if not tracks:
        print("Nothing to do.")
        return

    for track_name in tracks:
        track_songs = by_track.get(track_name, [])
        if len(track_songs) < N_QUERIES * 2:
            print(f"  {track_name}: only {len(track_songs)} songs, skipping")
            continue
        track_seed = SEED + abs(hash(track_name)) % (2 ** 31)
        pairs = process_track(track_name, track_songs, track_seed)
        with open(OUT_DIR / f"{track_name}.json", "w",
                  encoding="utf-8") as f:
            json.dump({"track": track_name, "pairs": pairs},
                      f, ensure_ascii=False, indent=2)
        counts = {"time": 0, "position": 0, "audio": 0}
        for p in pairs:
            counts[p["ref_method"]] += 1
        print(f"  {track_name}: {len(pairs)} pairs  "
              f"[time:{counts['time']}  position:{counts['position']}  "
              f"audio:{counts['audio']}]")

    print("Done.")


if __name__ == "__main__":
    main()
