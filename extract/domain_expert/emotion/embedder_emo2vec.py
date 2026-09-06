from __future__ import annotations

from typing import List

import numpy as np
import torch
import torchaudio

from extract.domain_expert.base import DomainEmbBase


def _to_mono_16k(wave: np.ndarray, sr: int) -> np.ndarray:
    t = torch.from_numpy(np.asarray(wave, dtype=np.float32))
    if t.ndim == 2:
        t = t.mean(dim=0)
    if sr != 16000:
        t = torchaudio.functional.resample(t, sr, 16000)
    return t.numpy()


class Emo2VecEmb(DomainEmbBase):
    """emotion2vec utterance-level embeddings via funasr."""

    def __init__(
        self, model_name: str = "emotion2vec/emotion2vec_base"
    ) -> None:
        import logging
        logging.getLogger("funasr").setLevel(logging.WARNING)
        logging.getLogger("modelscope").setLevel(logging.WARNING)
        from funasr import AutoModel as FunASRModel
        self._model = FunASRModel(
            model=model_name, hub="hf", disable_update=True
        )

    def embed(self, wave: np.ndarray, sr: int) -> np.ndarray:
        wav16 = _to_mono_16k(wave, sr)
        res = self._model.generate(
            input=wav16,
            granularity="utterance",
            extract_embedding=True,
        )
        return np.asarray(res[0]["feats"], dtype=np.float32)

    def embed_batch(
        self, waves: List[np.ndarray], srs: List[int]
    ) -> np.ndarray:
        return np.stack(
            [self.embed(w, s) for w, s in zip(waves, srs)]
        )
