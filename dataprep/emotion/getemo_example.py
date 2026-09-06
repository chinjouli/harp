"""Build emotion example clip pool from test episodes.

For each non-neutral emotion code, collects the 20 highest-cf clean clips
from test episodes. Clips from the same episode as the query are excluded
at query-generation time.

Output:
  audio/emotion_0001.wav …   extracted clips (sequential numbering)
  emo_clip_index.json        {emo_code: [{wav, ep, global_spk, cf, start, end}]}

Usage:
  python dataprep/emotion/getemo_example.py --build-index   # pick clips, save index
  python dataprep/emotion/getemo_example.py --extract       # cut wavs from index
"""
import argparse
import csv
import json
import subprocess
from collections import defaultdict

from dataprep import paths


EMO_CAP = 20
SNR_MIN = 20.0
DUR_MIN = 3.0
DUR_MAX = 8.0
NEUTRAL = {"N", "X"}

EMO_CODE = {"Neutral": "N", "Angry": "A", "Sad": "S", "Happy": "H",
            "Surprise": "U", "Fear": "F", "Disgust": "D",
            "Contempt": "C", "Other": "O"}

def _wire(args) -> None:
    """Resolve path constants from --src / --out."""
    src, out = paths.resolve(args, "longemo_dataset")
    globals().update(
        POD_V2=src / "msppodcast_full/msp_podcast_v2.0",
        FLAC_DIR=src / "msppodcast_full/podcasts_flac",
        JSONL_DIR=out / "jsons",
        AUDIO_DIR=out / "audio",
        INDEX_FILE=out / "emo_clip_index.json",
        EPISODES_FILE=out / "src_dataset/test_episodes.jsonl",
    )


def build_index():
    test_eps = {
        f"MSP-PODCAST_{json.loads(l)['conversation'].split('_')[-1]}"
        for l in open(EPISODES_FILE, encoding="utf-8")
        if l.strip() and not l.startswith("//")
    }

    # ep -> {n: {emo, spkr, votes}}
    labels = defaultdict(dict)
    with open(POD_V2 / "Labels/labels_consensus.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ep = "_".join(row["FileName"].split("_")[:2])
            if ep not in test_eps:
                continue
            emo = row["EmoClass"]
            if emo in NEUTRAL:
                continue
            n = row["FileName"][len(ep) + 1:-4]
            labels[ep][n] = {"emo": emo, "spkr": row["SpkrID"],
                             "votes": defaultdict(int)}

    with open(POD_V2 / "Labels/labels_detailed.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ep = "_".join(row["FileName"].split("_")[:2])
            n  = row["FileName"][len(ep) + 1:-4]
            if ep in labels and n in labels[ep]:
                em = EMO_CODE.get(row["EmoClass_Major"], row["EmoClass_Major"])
                labels[ep][n]["votes"][em] += 1

    # best clean clip per global speaker per episode (for matching)
    best_clip = {}   # (ep, global_spk) -> best seg dict
    for ep in test_eps:
        fp = JSONL_DIR / f"{ep}.jsonl"
        if not fp.exists():
            continue
        for line in open(fp, encoding="utf-8"):
            s   = json.loads(line)
            g   = s.get("global_speaker2", [])
            dur = s["end"] - s["start"]
            if (len(g) == 1 and s.get("num_speakers", 2) == 1
                    and s.get("speech_classification", "") == "Speech"
                    and DUR_MIN <= dur <= DUR_MAX
                    and s.get("SNR", 0) >= SNR_MIN):
                key = (ep, g[0])
                if key not in best_clip or s["SNR"] > best_clip[key]["SNR"]:
                    best_clip[key] = s

    emo_pool = defaultdict(list)
    for ep, segs in labels.items():
        for n, lab in segs.items():
            votes = lab["votes"]
            tot   = sum(votes.values())
            if tot == 0:
                continue
            cf = round(votes.get(lab["emo"], 0) / tot, 3)
            if cf < 0.50:
                continue
            spkr = lab.get("spkr", "")
            if not spkr.isdigit():
                continue
            global_spk = int(spkr)
            seg = best_clip.get((ep, global_spk))
            if seg is None:
                continue
            emo_pool[lab["emo"]].append({
                "ep": ep, "global_spk": global_spk, "cf": cf,
                "start": seg["start"], "end": seg["end"], "snr": seg["SNR"],
                "act": round(seg.get("arousal", 0.5), 3),
                "val": round(seg.get("valence", 0.5), 3),
                "dom": round(seg.get("dominance", 0.5), 3),
            })

    index = {}
    for emo, clips in emo_pool.items():
        clips.sort(key=lambda c: (-c["cf"], -c["snr"]))
        index[emo] = clips[:EMO_CAP]

    AUDIO_DIR.mkdir(exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"Index saved → {INDEX_FILE}")
    for emo, clips in index.items():
        print(f"  {emo}: {len(clips)} clips")
    return index


def extract_clips(index):
    AUDIO_DIR.mkdir(exist_ok=True)
    counter = 1
    updated = {}
    for emo, clips in index.items():
        updated_clips = []
        for c in clips:
            flac = FLAC_DIR / f"{c['ep']}.flac"
            if not flac.exists():
                print(f"  Missing FLAC: {flac}")
                continue
            wav_name = f"emotion_{counter:04d}.wav"
            out_path = AUDIO_DIR / wav_name
            if not out_path.exists():
                subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-ss", str(c["start"]), "-to", str(c["end"]),
                    "-i", str(flac), str(out_path), "-y",
                ], check=True)
            c["wav"] = wav_name
            updated_clips.append(c)
            counter += 1
        updated[emo] = updated_clips

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
            raise FileNotFoundError("Run --build-index first, or supply the index")
        extract_clips(json.load(open(INDEX_FILE, encoding="utf-8")))


if __name__ == "__main__":
    main()
