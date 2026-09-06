"""Extract 5-second audio snippets, one per song across all tracks.

Output: audio/snippet_NNN.wav  +  clip_index.json

Run from the repo root:
    python dataprep/music/extract_clips.py
    python dataprep/music/extract_clips.py --rerun
"""
import argparse
import json

from dataprep import paths

from pydub import AudioSegment

CLIP_DUR_S   = 5.0
BOUNDARY_PAD = 5.0  # seconds to stay away from crossfade edges

def _wire(args) -> None:
    """Resolve path constants from --src / --out."""
    src, out = paths.resolve(args, "songeval")
    globals().update(
        TRACKS_META=out / "tracks_metadata.jsonl",
        TRACKS_DIR=out / "tracks",
        AUDIO_DIR=out / "audio",
        CLIP_INDEX=out / "clip_index.json",
    )


def clip_start(song_start, song_end):
    interior_start = song_start + BOUNDARY_PAD
    interior_end   = song_end   - BOUNDARY_PAD
    mid = (interior_start + interior_end) / 2
    return max(interior_start, mid - CLIP_DUR_S / 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_path_args(ap)
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()
    _wire(args)

    AUDIO_DIR.mkdir(exist_ok=True)

    index = {}
    if not args.rerun and CLIP_INDEX.exists():
        index = json.load(open(CLIP_INDEX, encoding="utf-8"))
    done = {(v["track"], v["file_name"]) for v in index.values()}

    songs    = [json.loads(l) for l in open(TRACKS_META, encoding="utf-8")]
    by_track = {}
    for s in songs:
        by_track.setdefault(s["track_name"], []).append(s)

    counter = max(
        (int(k.split("_")[1].split(".")[0]) for k in index),
        default=0
    ) + 1

    for track_name in sorted(by_track):
        track_songs = sorted(by_track[track_name],
                             key=lambda s: s["start_time"])
        pending = [s for s in track_songs
                   if (track_name, s["file_name"]) not in done
                   or args.rerun]
        if not pending:
            print(f"  {track_name}: already done")
            continue

        mp3 = TRACKS_DIR / f"{track_name}.mp3"
        if not mp3.exists():
            print(f"  {track_name}: mp3 not found, skipping")
            continue

        print(f"  {track_name}: loading ...", end=" ", flush=True)
        audio = AudioSegment.from_mp3(mp3)

        for song in pending:
            cs      = clip_start(song["start_time"], song["end_time"])
            t0_ms   = int(cs * 1000)
            t1_ms   = int((cs + CLIP_DUR_S) * 1000)
            clip    = audio[t0_ms:t1_ms]
            name    = f"snippet_{counter:03d}.wav"
            clip.export(AUDIO_DIR / name, format="wav")
            index[name] = {
                "track":     track_name,
                "file_name": song["file_name"],
                "start":     round(cs, 2),
                "end":       round(cs + CLIP_DUR_S, 2),
            }
            counter += 1

        print(f"{len(pending)} clips")

    with open(CLIP_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(index)} entries → {CLIP_INDEX}")


if __name__ == "__main__":
    main()
