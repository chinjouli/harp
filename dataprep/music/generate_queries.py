"""Generate compare queries for each medley.

Reads pairs/track_NN.json + clip_index.json,
outputs queries/track_NN.jsonl.

Run from the repo root:
    python dataprep/music/generate_queries.py
    python dataprep/music/generate_queries.py --track track_00
    python dataprep/music/generate_queries.py --rerun
"""
import argparse
import json

from dataprep import paths

def _wire(args) -> None:
    """Resolve path constants from --src / --out."""
    src, out = paths.resolve(args, "songeval")
    globals().update(
        PAIRS_DIR=out / "pairs",
        OUT_DIR=out / "queries",
        CLIP_INDEX=out / "clip_index.json",
        TRACKS_META=out / "tracks_metadata.jsonl",
    )


def fmt_time(secs):
    half = round(secs / 30)
    mins = half // 2
    return (f"around {mins} minute{'s' if mins != 1 else ''}"
            if half % 2 == 0 else f"around {mins} and a half minutes")


def ordinal(n):
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def find_clip(clip_index, track, file_name):
    for name, info in clip_index.items():
        if info["track"] == track and info["file_name"] == file_name:
            return name
    return None


def make_query(pair, track, clip_index):
    method = pair["ref_method"]
    sa, sb = pair["song_a"], pair["song_b"]
    ref_a, ref_b = {}, {}

    if method == "time":
        ta = fmt_time(sa["time_ref"])
        tb = fmt_time(sb["time_ref"])
        q  = (f"Compare the two songs at around {ta} and {tb}. "
              f"Which of these two songs has better overall aesthetics?")
        ref_a["time_ref"] = sa["time_ref"]
        ref_b["time_ref"] = sb["time_ref"]

    elif method == "position":
        ta = ordinal(sa["position"])
        tb = ordinal(sb["position"])
        q  = (f"Compare the {ta} and {tb} songs in this medley. "
              f"Which has better overall aesthetics?")
        ref_a["position"] = sa["position"]
        ref_b["position"] = sb["position"]

    elif method == "audio":
        clip_a = find_clip(clip_index, track, sa["file_name"])
        clip_b = find_clip(clip_index, track, sb["file_name"])
        if not clip_a or not clip_b:
            raise ValueError(
                f"Missing clip for {sa['file_name']} or {sb['file_name']}")
        q  = (f"Song A: [A's audio example]. Song B: [B's audio example]. "
              f"Both clips come from this medley. "
              f"Which song has better overall aesthetics?")
        ref_a["audio_clip"] = clip_a
        ref_b["audio_clip"] = clip_b

    else:
        raise ValueError(f"Unknown ref_method: {method}")

    return q, ref_a, ref_b


def process_track(track_name, clip_index):
    src = PAIRS_DIR / f"{track_name}.json"
    if not src.exists():
        print(f"  {track_name}: missing pairs, skipping.")
        return 0

    data  = json.load(open(src, encoding="utf-8"))
    OUT_DIR.mkdir(exist_ok=True)
    count = 0
    with open(OUT_DIR / f"{track_name}.jsonl", "w", encoding="utf-8") as f:
        for i, pair in enumerate(data["pairs"]):
            try:
                q, ref_a, ref_b = make_query(pair, track_name, clip_index)
            except ValueError as e:
                print(f"  {track_name} pair {i} skipped: {e}")
                continue
            sa, sb = pair["song_a"], pair["song_b"]
            rec = {
                "episode_id":        track_name,
                "query_type":   pair["ref_method"],
                "query":        q,
                "song_a":       {"file_name": sa["file_name"], **ref_a},
                "song_b":       {"file_name": sb["file_name"], **ref_b},
                "ground_truth": pair["ground_truth"],
                "judge_info":   pair["judge_info"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_path_args(ap)
    ap.add_argument("--track", default=None)
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()
    _wire(args)

    clip_index = (json.load(open(CLIP_INDEX, encoding="utf-8"))
                  if CLIP_INDEX.exists() else {})

    songs  = [json.loads(l) for l in open(TRACKS_META, encoding="utf-8")]
    tracks = sorted({s["track_name"] for s in songs})
    if args.track:
        tracks = [args.track]
    if not args.rerun:
        tracks = [t for t in tracks
                  if not (OUT_DIR / f"{t}.jsonl").exists()]

    if not tracks:
        print("Nothing to do.")
        return

    print(f"Processing {len(tracks)} tracks ...")
    total = sum(process_track(t, clip_index) for t in tracks)
    print(f"Done. {total} queries written.")


if __name__ == "__main__":
    main()
