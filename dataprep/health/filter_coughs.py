"""
Score and filter Coswara cough recordings by audio quality.
Outputs a CSV ranked by quality score; top N are flagged as selected.
"""

import warnings
import numpy as np
import pandas as pd
import librosa
from itertools import groupby

import argparse

from dataprep import paths

warnings.filterwarnings("ignore")

SNR_POOL = 2000   # take top N by SNR before applying hard rejections
FRAME_LEN = 512   # ~10ms at 48kHz
HOP_LEN = 256

def _wire(args) -> None:
    """Resolve path constants from --src / --out."""
    src, out = paths.resolve(args, "medmosaic")
    globals().update(
        DATA_ROOT=src / "Coswara-Data/Extracted_data",
        OUTPUT_CSV=out / "cough_quality.csv",
    )


def estimate_snr(y, sr):
    """SNR: ratio of loud (cough) frames to quiet (noise floor) frames."""
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LEN, hop_length=HOP_LEN)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-9)
    noise_floor = np.percentile(rms_db, 10)
    signal_peak = np.percentile(rms_db, 90)
    return float(signal_peak - noise_floor)


def clipping_ratio(y):
    """Fraction of samples above 0.90 (aggressive clipping threshold)."""
    return float(np.mean(np.abs(y) > 0.90))


def near_saturation_ratio(y):
    """
    Fraction of samples above 0.80 AND longest consecutive run.
    Catches mic overload distortion before hard clipping occurs.
    """
    hi = np.abs(y) > 0.80
    runs = [sum(1 for _ in g) for k, g in groupby(hi) if k]
    max_run = max(runs) if runs else 0
    return float(np.mean(hi)), max_run


def active_ratio(y, sr):
    """Fraction of frames above an energy threshold (should be 5-50%)."""
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LEN, hop_length=HOP_LEN)[0]
    threshold = np.percentile(rms, 30)
    return float(np.mean(rms > threshold * 3))


def count_cough_onsets(y, sr):
    """Number of detected onset impulses — should be >= 1."""
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=HOP_LEN,
        backtrack=True, units="frames"
    )
    return int(len(onset_frames))


def noise_flatness(y, sr):
    """
    Spectral flatness of quiet frames. High flatness = white noise.
    Low is better (tonal bg is worse than white noise for cough isolation,
    but white noise is also bad). We want low flatness in quiet segments.
    """
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LEN, hop_length=HOP_LEN)[0]
    quiet_mask = rms < np.percentile(rms, 25)
    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=FRAME_LEN, hop_length=HOP_LEN
    )[0]
    if quiet_mask.shape[0] > flatness.shape[0]:
        quiet_mask = quiet_mask[: flatness.shape[0]]
    else:
        flatness = flatness[: quiet_mask.shape[0]]
    if quiet_mask.sum() == 0:
        return float(np.mean(flatness))
    return float(np.mean(flatness[quiet_mask]))


def score_file(path):
    try:
        y, sr = librosa.load(str(path), sr=None, mono=True)
    except Exception as e:
        return None, {"error": str(e)}

    if len(y) < sr * 0.5:  # reject clips shorter than 0.5s
        return None, {"error": "too short"}

    snr = estimate_snr(y, sr)
    clip = clipping_ratio(y)
    near_sat, max_sat_run = near_saturation_ratio(y)
    active = active_ratio(y, sr)
    onsets = count_cough_onsets(y, sr)
    flatness = noise_flatness(y, sr)
    duration = len(y) / sr

    # Hard reject conditions
    if clip > 0.005:        # >0.5% hard-clipped samples
        reject = "clipping"
    elif near_sat > 0.0005: # >0.05% near mic saturation (>0.90 amplitude)
        reject = "saturation"
    elif onsets < 1:        # no detectable cough impulse
        reject = "no_onset"
    elif active > 0.8:      # nearly always loud = noise/music
        reject = "always_active"
    elif snr < 5:           # essentially no SNR
        reject = "low_snr"
    else:
        reject = ""

    # Composite quality score (higher = better)
    active_penalty = abs(active - 0.25) * 10  # ideal active ~25%
    score = snr - clip * 500 - active_penalty - flatness * 20

    return score, {
        "path": str(path.resolve()),
        "cough_type": path.stem,
        "date": path.parts[-3],
        "patient_id": path.parts[-2],
        "duration_s": round(duration, 2),
        "snr_db": round(snr, 2),
        "clipping_ratio": round(clip, 5),
        "near_sat_ratio": round(near_sat, 6),
        "max_sat_run": max_sat_run,
        "active_ratio": round(active, 3),
        "onset_count": onsets,
        "noise_flatness": round(flatness, 4),
        "score": round(score, 3),
        "reject_reason": reject,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    paths.add_path_args(ap)
    _wire(ap.parse_args())
    wavs = sorted(DATA_ROOT.glob("*/*/cough*.wav"))
    print(f"Found {len(wavs)} cough wav files")

    rows = []
    for i, p in enumerate(wavs):
        if i % 500 == 0:
            print(f"  {i}/{len(wavs)}...")
        score, row = score_file(p)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Step 1: rank all files by SNR, take top SNR_POOL as candidates
    df = df.sort_values("snr_db", ascending=False).reset_index(drop=True)
    df["in_snr_pool"] = df.index < SNR_POOL

    # Step 2: selected = in pool AND not hard-rejected
    df["selected"] = df["in_snr_pool"] & (df["reject_reason"].fillna("") == "")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults written to {OUTPUT_CSV}")
    print(f"  Total files:      {len(df)}")
    print(f"  Top {SNR_POOL} SNR pool:  {df['in_snr_pool'].sum()}")
    rejected_in_pool = df["in_snr_pool"] & (df["reject_reason"].fillna("") != "")
    print(f"  Rejected in pool: {rejected_in_pool.sum()}")
    print(f"  Selected:         {df['selected'].sum()}")
    print(f"\nReject reason counts (in SNR pool):")
    print(df[rejected_in_pool]["reject_reason"].value_counts())
    print(f"\nStats for selected files:")
    sel = df[df["selected"]]
    print(sel[["snr_db", "clipping_ratio", "near_sat_ratio",
               "active_ratio", "onset_count"]].describe().round(4))


if __name__ == "__main__":
    main()
