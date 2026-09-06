from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from extract.audio_expert.base import TextEmbBase


class SentenceEmb(TextEmbBase):
    """Sentence-transformer text embeddings (L2-normalized, float32)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = SentenceTransformer(model_name, device=self.device)

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)
