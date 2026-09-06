from __future__ import annotations

import os
from typing import List

import numpy as np
import torch

from extract.audio_expert.base import EmbBase


class PyannoteEmb(EmbBase):
    """Per-segment speaker embeddings (512-d) via pyannote/embedding."""

    def __init__(
        self,
        model_name: str = "pyannote/embedding",
    ) -> None:
        from pyannote.audio import Inference, Model

        cache = os.path.expanduser("~/.cache/huggingface/token")
        token = os.environ.get("HF_TOKEN") or (
            open(cache).read().strip() if os.path.exists(cache) else None
        )
        model = Model.from_pretrained(model_name, token=token)
        if torch.cuda.is_available():
            model = model.to("cuda")
        self._inference = Inference(model, window="whole")

    def embed(self, wave: np.ndarray, sr: int) -> np.ndarray:
        waveform = torch.from_numpy(
            np.asarray(wave, dtype=np.float32)
        ).unsqueeze(0)  # (1, samples)
        vec = self._inference({"waveform": waveform, "sample_rate": sr})
        return np.asarray(vec, dtype=np.float32).flatten()

    def embed_batch(
        self, waves: List[np.ndarray], srs: List[int]
    ) -> np.ndarray:
        return np.stack(
            [self.embed(w, s) for w, s in zip(waves, srs)]
        )
