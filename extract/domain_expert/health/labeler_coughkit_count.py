from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torchaudio

from extract.domain_expert.base import DomainLabelerBase


def _to_mono_16k(wave: np.ndarray, sr: int) -> np.ndarray:
    t = torch.from_numpy(np.asarray(wave, dtype=np.float32))
    if t.ndim == 2:
        t = t.mean(dim=0)
    if sr != 16000:
        t = torchaudio.functional.resample(t, sr, 16000)
    return t.numpy()


class CoughKitCounter(DomainLabelerBase):
    """Cough event counter via CoughKit energy segmentation + XGBoost.

    Splits the segment into high-energy sub-windows, classifies each,
    and counts those that exceed the threshold.

    Schema:
        {primary: "{n} cough"|"no cough",
         max_cough_prob: float}  # peak sub-window cough probability
    """

    def __init__(self, threshold: float = 0.5) -> None:
        from coughkit import load_cough_classifier, load_scaler
        self._model = load_cough_classifier()
        self._scaler = load_scaler()
        self._threshold = threshold

    def label(self, wave: np.ndarray, sr: int) -> Dict[str, Any]:
        from coughkit.segmentation import segment_cough
        from coughkit.dsp import classify_cough

        wav16 = _to_mono_16k(wave, sr)
        segments, _ = segment_cough(wav16, 16000)

        probs = [
            float(classify_cough(seg, 16000, self._model, self._scaler))
            for seg in segments
        ]
        count = sum(1 for p in probs if p >= self._threshold)
        max_prob = float(max(probs)) if probs else 0.0

        return {
            "primary": f"{count} cough" if count > 0 else "no cough",
            "max_cough_prob": max_prob,
        }
