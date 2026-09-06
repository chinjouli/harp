from __future__ import annotations

from typing import List

import numpy as np
import torch
import torchaudio

from extract.domain_expert.base import DomainEmbBase

_TARGET_SR = 24000  # MERT training sample rate


def _to_24k(wave: np.ndarray, sr: int) -> np.ndarray:
    t = torch.from_numpy(np.asarray(wave, dtype=np.float32))
    if t.ndim == 2:
        t = t.mean(dim=0)
    if sr != _TARGET_SR:
        t = torchaudio.functional.resample(t, sr, _TARGET_SR)
    return t.numpy()


class MERTEmb(DomainEmbBase):
    """Music embeddings (768-d) from MERT, mean-pooled across time.

    layer_idx selects which transformer layer to use (-1 = last).
    Different layers perform differently by task; tune via config.
    """

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        layer_idx: int = -1,
    ) -> None:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self._layer_idx = layer_idx
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self._device).eval()

    @torch.inference_mode()
    def embed(self, wave: np.ndarray, sr: int) -> np.ndarray:
        return self.embed_batch([wave], [sr])[0]

    @torch.inference_mode()
    def embed_batch(
        self, waves: List[np.ndarray], srs: List[int]
    ) -> np.ndarray:
        resampled = [_to_24k(w, s) for w, s in zip(waves, srs)]
        inputs = self._processor(
            resampled,
            sampling_rate=_TARGET_SR,
            return_tensors="pt",
            padding=True,
        ).to(self._device)
        outputs = self._model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of (batch, time, 768) per layer
        hidden = outputs.hidden_states[self._layer_idx]
        pooled = hidden.mean(dim=1)  # (batch, 768)
        return pooled.cpu().numpy().astype(np.float32)
