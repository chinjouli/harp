from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import torch

from extract.audio_expert.base import DiarBase


class PyannoteDiar(DiarBase):
    """Speaker diarization via pyannote; returns real speaker IDs directly."""

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-community-1",
    ) -> None:
        from pyannote.audio import Pipeline

        token = os.environ["HF_TOKEN"]
        self._pipeline = Pipeline.from_pretrained(
            model_name, token=token
        )
        if torch.cuda.is_available():
            self._pipeline.to(torch.device("cuda"))

    def segment(
        self, wave: np.ndarray, sr: int
    ) -> List[Dict[str, Any]]:
        waveform = torch.from_numpy(
            np.asarray(wave, dtype=np.float32)
        ).unsqueeze(0)  # (1, samples)
        output = self._pipeline(
            {"waveform": waveform, "sample_rate": sr}
        )
        annotation = (
            output.speaker_diarization
            if hasattr(output, "speaker_diarization")
            else output
        )
        return [
            {
                "speaker": speaker,
                "start": float(segment.start),
                "end": float(segment.end),
            }
            for segment, _, speaker in annotation.itertracks(
                yield_label=True
            )
        ]
