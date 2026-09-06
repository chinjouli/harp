#!/usr/bin/env python3
"""Language-ID the raw SongEval mp3s to find the English ones.

Two passes: 0-30 s for every track, then 30-60 s for uncertain rows.
Writes languages.jsonl then languages2.jsonl.

Run from the repo root:
    python dataprep/music/find_engsong.py
"""
import argparse
import json

import whisper

from dataprep import paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    paths.add_path_args(ap)
    src, out = paths.resolve(ap.parse_args(), "songeval")

    model = whisper.load_model("large-v3")


    # Round 1: predict LID on the 0-30s chunk
    out = open(out / "languages.jsonl", "a", encoding="utf-8")

    for mp3 in sorted((src / "mp3").glob("*.mp3")):
        audio = whisper.load_audio(str(mp3))
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)
        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)
        row = {"file": mp3.name, "lang": lang, "prob": round(probs[lang], 4)}
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        print(row)

    out.close()


    # Round 2: re-check uncertain rows on the 30-60s chunk
    SAMPLE_RATE = whisper.audio.SAMPLE_RATE  # 16000
    CHUNK = SAMPLE_RATE * 30  # samples per 30s

    rows = [json.loads(l) for l in open(out / "languages.jsonl", encoding="utf-8")]
    out2 = open(out / "languages2.jsonl", "w", encoding="utf-8")

    for row in rows:
        needs_rerun = row["lang"] not in ("zh", "en") or row["prob"] < 0.5
        if not needs_rerun:
            out2.write(json.dumps(row, ensure_ascii=False) + "\n")
            out2.flush()
            continue
        audio = whisper.load_audio(str(src / "mp3" / row["file"]))
        chunk2 = whisper.pad_or_trim(audio[CHUNK: CHUNK * 2])
        mel = whisper.log_mel_spectrogram(chunk2, n_mels=128).to(model.device)
        _, probs = model.detect_language(mel)
        lang2 = max(probs, key=probs.get)
        row2 = {**row, "lang2": lang2, "prob2": round(probs[lang2], 4)}
        out2.write(json.dumps(row2, ensure_ascii=False) + "\n")
        out2.flush()
        print(row2)

    out2.close()


if __name__ == "__main__":
    main()
