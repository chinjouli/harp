from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio
from transformers import AutoModelForAudioClassification

from extract.domain_expert.base import DomainLabelerBase


def _to_mono_16k(wave: np.ndarray, sr: int) -> torch.Tensor:
    t = torch.from_numpy(wave).float()
    if t.ndim == 2:
        t = t.mean(dim=0)
    if sr != 16000:
        t = torchaudio.functional.resample(t, sr, 16000)
    return t


class SERLabeler(DomainLabelerBase):
    """Categorical emotion labeler using a WavLM-based SER model.

    Returns domain_labels schema compatible with emotion task:
        {primary, scores, secondary=None, votes=None,
         valence=None, arousal=None, dominance=None}
    """

    def __init__(
        self,
        model_tag: str = (
            "3loi/SER-Odyssey-Baseline-WavLM-Categorical"
        ),
        max_sec: float = 15.0,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = AutoModelForAudioClassification.from_pretrained(
            model_tag, trust_remote_code=True
        )
        self._model.to(self.device).eval()
        self._mean = torch.as_tensor(
            self._model.config.mean, dtype=torch.float32
        )
        self._std = torch.as_tensor(
            self._model.config.std, dtype=torch.float32
        )
        self._max_samples = int(max_sec * 16000)
        self._id2label: Dict[int, str] = self._model.config.id2label

    @torch.inference_mode()
    def label(self, wave: np.ndarray, sr: int) -> Dict[str, Any]:
        wav = _to_mono_16k(wave, sr)[: self._max_samples]
        mean = self._mean.to(self.device)
        std = self._std.to(self.device)
        normed = ((wav.to(self.device) - mean) / (std + 1e-6)).unsqueeze(0)
        mask = torch.ones(1, normed.size(1), device=self.device)
        out = self._model(normed, mask)
        logits = out.logits if hasattr(out, "logits") else out
        probs = torch.softmax(logits[0], dim=0).cpu().numpy()

        scores = {
            self._id2label[i]: float(p)
            for i, p in enumerate(probs)
        }
        primary = max(scores, key=lambda k: scores[k])
        return {
            "primary": primary,
            "scores": scores,
            "secondary": None,
            "votes": None,
            "valence": None,
            "arousal": None,
            "dominance": None,
        }
