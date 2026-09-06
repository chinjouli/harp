from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as TAT

from extract.domain_expert.base import DomainLabelerBase

_DEFAULT_CKPT = os.path.expandvars(
    "${HARP_DATA_ROOT}/emo-reasoning/checkpoints/cser/epoch_200.pt"
)
_CHUNK_SEC = 30
_TARGET_SR = 16000
_FRAMES_PER_SEC = 50


class _CSERM(nn.Module):
    """BiLSTM head over WavLM-Large 1-Hz features → VAD per second."""

    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 2,
        num_bilstm: int = 2,
    ) -> None:
        super().__init__()
        self.bilstm_1 = nn.LSTM(
            input_size=1024,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.5,
            batch_first=True,
        )
        if num_bilstm >= 2:
            self.bilstm_2 = nn.LSTM(
                input_size=hidden_size * 2,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=True,
                dropout=0.5,
                batch_first=True,
            )
        self._num_bilstm = num_bilstm
        self.output_layer = nn.Linear(hidden_size * 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.bilstm_1(x)
        if self._num_bilstm >= 2:
            out, _ = self.bilstm_2(out)
        return self.output_layer(out)


def _to_mono_16k(wave: np.ndarray, sr: int) -> torch.Tensor:
    t = torch.from_numpy(wave).float()
    if t.ndim == 2:
        t = t.mean(dim=1)
    if sr != _TARGET_SR:
        t = TAT.Resample(sr, _TARGET_SR)(t.unsqueeze(0)).squeeze(0)
    return t


def _extract_wavlm(
    waveform: torch.Tensor, wavlm, device
) -> torch.Tensor:
    # chunk to avoid OOM; returns [T_seconds, 1024]
    chunk_samples = _CHUNK_SEC * _TARGET_SR
    parts = []
    for chunk in waveform.split(chunk_samples):
        inp = chunk.unsqueeze(0).to(device)
        with torch.no_grad():
            out = wavlm(input_values=inp)
        parts.append(out.last_hidden_state[0].cpu())
    frames = torch.cat(parts, dim=0)
    n_real = int(waveform.shape[0] / _TARGET_SR * _FRAMES_PER_SEC) + 1
    frames = frames[:n_real]
    t, d = frames.shape
    pad = (_FRAMES_PER_SEC - t % _FRAMES_PER_SEC) % _FRAMES_PER_SEC
    if pad:
        frames = torch.cat([frames, torch.zeros(pad, d)])
    n_sec = frames.shape[0] // _FRAMES_PER_SEC
    return frames.view(n_sec, _FRAMES_PER_SEC, d).mean(dim=1)  # [T, 1024]


class CSERLabeler(DomainLabelerBase):
    """Continuous VAD labeler: WavLM-Large + BiLSTM (CSER).

    Returns per-dimension mean and std over the segment in [0, 1].
    primary/scores/secondary/votes are None (continuous model).
    valence/arousal/dominance are stored as {"mean": float, "std": float}.
    """

    def __init__(
        self,
        ckpt_path: str = _DEFAULT_CKPT,
        wavlm_model: str = "microsoft/wavlm-large",
        hidden_size: int = 512,
        num_layers: int = 2,
        num_bilstm: int = 2,
        device: Optional[str] = None,
    ) -> None:
        from transformers import WavLMModel

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._wavlm = (
            WavLMModel.from_pretrained(wavlm_model).to(self.device).eval()
        )
        self._cser = _CSERM(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_bilstm=num_bilstm,
        )
        self._cser.load_state_dict(
            torch.load(ckpt_path, map_location=self.device)
        )
        self._cser.to(self.device).eval()

    @torch.inference_mode()
    def label(self, wave: np.ndarray, sr: int) -> Dict[str, Any]:
        waveform = _to_mono_16k(wave, sr)
        features = _extract_wavlm(waveform, self._wavlm, self.device)
        pred = (
            self._cser(features.unsqueeze(0).to(self.device))[0]
            .cpu()
            .clamp(0, 1)
            .numpy()
        )  # [T, 3]: arousal, valence, dominance per second

        def _stats(col: np.ndarray) -> Dict[str, float]:
            return {
                "mean": float(col.mean()),
                "std": float(col.std()),
                "start": float(col[0]),
                "end": float(col[-1]),
            }

        return {
            "primary": None,
            "scores": None,
            "secondary": None,
            "votes": None,
            "arousal": _stats(pred[:, 0]),
            "valence": _stats(pred[:, 1]),
            "dominance": _stats(pred[:, 2]),
        }
