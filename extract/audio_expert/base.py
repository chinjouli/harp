from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class ASRBase(ABC):
    @abstractmethod
    def transcribe(self, wave: np.ndarray, sr: int) -> str: ...

    def transcribe_batch(
        self, waves: List[np.ndarray], srs: List[int]
    ) -> List[str]:
        return [self.transcribe(w, s) for w, s in zip(waves, srs)]


class DiarBase(ABC):
    @abstractmethod
    def segment(
        self, wave: np.ndarray, sr: int
    ) -> List[Dict[str, Any]]:
        """Return speaker-labeled turns: [{speaker, start, end}].

        Clustering is internal — caller receives global speaker IDs
        that are consistent across the full episode.
        """
        ...


class EmbBase(ABC):
    @abstractmethod
    def embed(self, wave: np.ndarray, sr: int) -> np.ndarray: ...

    def embed_batch(
        self, waves: List[np.ndarray], srs: List[int]
    ) -> np.ndarray:
        return np.stack(
            [self.embed(w, s) for w, s in zip(waves, srs)]
        )


class TextEmbBase(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])
