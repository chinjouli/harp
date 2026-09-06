from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class DomainLabelerBase(ABC):
    @abstractmethod
    def label(self, wave: np.ndarray, sr: int) -> Dict[str, Any]:
        """Return task-specific label dict.

        Emotion schema:
            {primary: str, scores: {cls: float},
             secondary: str|None, votes: dict|None,
             valence: float|None, arousal: float|None,
             dominance: float|None}

        Other tasks define their own schema; the retriever treats
        domain_labels as opaque.
        """
        ...


class DomainEmbBase(ABC):
    @abstractmethod
    def embed(self, wave: np.ndarray, sr: int) -> np.ndarray: ...

    def embed_batch(
        self, waves: List[np.ndarray], srs: List[int]
    ) -> np.ndarray:
        return np.stack(
            [self.embed(w, s) for w, s in zip(waves, srs)]
        )
