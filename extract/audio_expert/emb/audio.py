from __future__ import annotations

import tempfile
from typing import List

import numpy as np
import soundfile as sf
import torch

from extract.audio_expert.base import EmbBase


class OpenBEATsEmb(EmbBase):
    """General-purpose audio embeddings (1024-d) via OpenBEATs.

    Suitable for music, bioacoustics, environmental sound, and speech.
    Patch embeddings are mean-pooled across time to produce one vector.
    """

    def __init__(
        self,
        checkpoint: str = "espnet/OpenBEATS-Large-i2-as20k",
        chunk_seconds: float = 10.0,
    ) -> None:
        from openbeats.model import OpenBeats

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = OpenBeats.from_pretrained(checkpoint, device=device)
        self._chunk_seconds = chunk_seconds

    def embed(self, wave: np.ndarray, sr: int) -> np.ndarray:
        return self.embed_batch([wave], [sr])[0]

    def embed_batch(
        self, waves: List[np.ndarray], srs: List[int]
    ) -> np.ndarray:
        results: List[np.ndarray] = []
        for wave, sr in zip(waves, srs):
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                sf.write(tmp.name, wave, sr)
                out = self._model.encode_file(
                    tmp.name, chunk_seconds=self._chunk_seconds
                )
            patches = out["patch_embeddings"]  # (num_patches, 1024)
            vec = np.asarray(patches, dtype=np.float32).mean(axis=0)
            results.append(vec)
        return np.stack(results)
