from __future__ import annotations

from typing import List

import numpy as np
import torch
import torchaudio
import whisper

from extract.audio_expert.base import ASRBase


def _to_16k(wave: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return wave
    t = torch.from_numpy(wave).float().unsqueeze(0)
    t = torchaudio.functional.resample(t, sr, 16000)
    return t.squeeze(0).numpy()


class WhisperASR(ASRBase):
    def __init__(self, model_name: str = "large-v3-turbo") -> None:
        self._model = whisper.load_model(model_name)

    def transcribe(self, wave: np.ndarray, sr: int) -> str:
        wave16 = _to_16k(wave, sr)
        result = self._model.transcribe(
            wave16, fp16=torch.cuda.is_available()
        )
        return str(result.get("text", "")).strip()

    def transcribe_batch(
        self, waves: List[np.ndarray], srs: List[int]
    ) -> List[str]:
        # whisper has no native batching; loop is fine for short segments
        return [self.transcribe(w, s) for w, s in zip(waves, srs)]
