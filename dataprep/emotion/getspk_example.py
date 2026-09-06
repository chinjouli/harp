"""Build speaker example clip pool.

For each global speaker, extracts the top-5 highest-SNR clean clips from
episodes other than the one being queried.

Output:
  audio/speaker_0001.wav …   extracted clips (sequential numbering)
  spk_clip_index.json        {global_spk_id: [{wav, ep, start, end, snr}]}

Usage:
  # --build-index scans all JSONL and saves the index; --extract cuts the wavs
  python dataprep/emotion/getspk_example.py --build-index
  python dataprep/emotion/getspk_example.py --extract
"""
import argparse
import json
import subprocess
from collections import defaultdict

from dataprep import paths


SNR_MIN = 20.0
DUR_MIN = 3.0
DUR_MAX = 8.0
SPK_CAP = 5

def _wire(args) -> None:
    """Resolve path constants from --src / --out."""
    src, out = paths.resolve(args, "longemo_dataset")
    globals().update(
        JSONL_DIR=out / "jsons",
        FLAC_DIR=src / "msppodcast_full/podcasts_flac",
        AUDIO_DIR=out / "audio",
        INDEX_FILE=out / "spk_clip_index.json",
        EPISODES_FILE=out / "src_dataset/test_episodes.jsonl",
    )


def is_good_clip(seg):
    dur = seg["end"] - seg["start"]
    return (len(seg.get("global_speaker2", [])) == 1
            and seg.get("num_speakers", 2) == 1
            and seg.get("speech_classification", "") == "Speech"
            and seg.get("SNR", 0) >= SNR_MIN
            and DUR_MIN <= dur <= DUR_MAX)


def build_index():
    # collect global speaker IDs from test episodes only
    test_eps = {
        f"MSP-PODCAST_{json.loads(l)['conversation'].split('_')[-1]}"
        for l in open(EPISODES_FILE, encoding="utf-8")
        if l.strip() and not l.startswith("//")
    }
    target_spks = set()
    for ep in test_eps:
        fp = JSONL_DIR / f"{ep}.jsonl"
        if not fp.exists():
            continue
        for line in open(fp, encoding="utf-8"):
            for g in json.loads(line).get("global_speaker2", []):
                target_spks.add(g)
    print(f"Scanning {JSONL_DIR} for {len(target_spks)} speakers …")

    all_clips = defaultdict(list)
    files = sorted(JSONL_DIR.glob("*.jsonl"))
    for i, fp in enumerate(files, 1):
        ep = fp.stem
        for line in open(fp, encoding="utf-8"):
            s = json.loads(line)
            g = s.get("global_speaker2", [])
            if len(g) != 1 or g[0] not in target_spks or not is_good_clip(s):
                continue
            all_clips[g[0]].append({
                "ep": ep, "start": s["start"], "end": s["end"],
                "snr": s["SNR"], "gender": s.get("gender", "?"),
            })
        if i % 500 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", flush=True)

    index = {}
    for spk, clips in all_clips.items():
        by_ep      = defaultdict(list)
        for c in clips:
            by_ep[c["ep"]].append(c)
        best_per_ep = sorted(
            [max(v, key=lambda c: c["snr"]) for v in by_ep.values()],
            key=lambda c: -c["snr"])
        index[str(spk)] = best_per_ep[:SPK_CAP]

    AUDIO_DIR.mkdir(exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"Index saved → {INDEX_FILE}  ({len(index)} speakers)")
    return index


def extract_clips(index):
    AUDIO_DIR.mkdir(exist_ok=True)
    counter = 1
    updated = {}
    for spk, clips in index.items():
        updated_clips = []
        for c in clips:
            flac = FLAC_DIR / f"{c['ep']}.flac"
            if not flac.exists():
                print(f"  Missing FLAC: {flac}")
                continue
            wav_name = f"speaker_{counter:04d}.wav"
            out_path = AUDIO_DIR / wav_name
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", str(c["start"]), "-to", str(c["end"]),
                "-i", str(flac), str(out_path), "-y",
            ], check=True)
            c["wav"] = wav_name
            updated_clips.append(c)
            counter += 1
        updated[spk] = updated_clips

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False)
    print(f"Extracted {counter - 1} clips → {AUDIO_DIR}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_path_args(ap)
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()
    _wire(args)

    if not args.build_index and not args.extract:
        ap.print_help()
        return
    if args.build_index:
        build_index()
    if args.extract:
        if not INDEX_FILE.exists():
            raise FileNotFoundError("Run --build-index first")
        extract_clips(json.load(open(INDEX_FILE, encoding="utf-8")))


if __name__ == "__main__":
    main()
