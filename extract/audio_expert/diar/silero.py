from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torchaudio

from extract.audio_expert.base import DiarBase

# speaker label is a placeholder; pipeline.py replaces it after clustering
_DEFAULT_SPEAKER = "SPEAKER_00"


class SileroVAD(DiarBase):
    """Voice activity detection using Silero VAD.

    Returns speech segments without speaker labels (all tagged
    SPEAKER_00). pipeline.py assigns real speaker IDs after
    agglomerative clustering of speaker embeddings.
    """

    def __init__(self, min_silence_sec: float = 0.5) -> None:
        self.min_silence_sec = min_silence_sec
        from silero_vad import load_silero_vad
        self._model = load_silero_vad()

    def segment(
        self, wave: np.ndarray, sr: int
    ) -> List[Dict[str, Any]]:
        """Return [{speaker, start, end}] for all speech regions."""
        from silero_vad import get_speech_timestamps

        t = torch.from_numpy(wave).float()
        if sr != 16000:
            t = torchaudio.functional.resample(t, sr, 16000)
            effective_sr = 16000
        else:
            effective_sr = sr

        raw = get_speech_timestamps(
            t,
            self._model,
            sampling_rate=effective_sr,
            return_seconds=True,
        )
        if not raw:
            return []

        merged = [dict(raw[0])]
        for seg in raw[1:]:
            if seg["start"] - merged[-1]["end"] < self.min_silence_sec:
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(dict(seg))

        return [
            {
                "speaker": _DEFAULT_SPEAKER,
                "start": float(s["start"]),
                "end": float(s["end"]),
            }
            for s in merged
        ]
