from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from extract.domain_expert.base import DomainLabelerBase


_METRICS = ["Naturalness", "Clarity", "Musicality", "Coherence", "Memorability"]
_TARGET_SR = 16000
_TARGET_SAMPLES = _TARGET_SR * 30  # 30 s


class _MusicAestheticsMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(23040, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 256), nn.ReLU(), nn.LayerNorm(256),
        )
        self.heads = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(256, 64), nn.ReLU(),
                nn.Linear(64, 1),  # raw output on 1–5 scale; no Sigmoid
            )
            for m in _METRICS
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.bottleneck(x)
        return {m: self.heads[m](shared) for m in _METRICS}


def _pool_whisper(hidden: torch.Tensor) -> torch.Tensor:
    """(1, 1500, 768) → (1, 23040) via mean/max/min pooling over 10 segments."""
    # 1500 frames / 10 segments = 150 frames each
    feats = hidden.view(1, 10, 150, 768)
    mean_p = feats.mean(dim=2)
    max_p = feats.max(dim=2).values
    min_p = feats.min(dim=2).values
    return torch.cat([mean_p, max_p, min_p], dim=2).view(1, -1)  # (1, 23040)


def _resample_and_pad(wave: np.ndarray, sr: int) -> np.ndarray:
    t = torch.from_numpy(np.asarray(wave, dtype=np.float32))
    if t.ndim == 2:
        t = t.mean(dim=0)
    if sr != _TARGET_SR:
        t = torchaudio.functional.resample(t, sr, _TARGET_SR)
    samples = t.numpy()
    if len(samples) >= _TARGET_SAMPLES:
        # center crop
        start = (len(samples) - _TARGET_SAMPLES) // 2
        samples = samples[start: start + _TARGET_SAMPLES]
    else:
        samples = np.pad(samples, (0, _TARGET_SAMPLES - len(samples)))
    return samples


class MusicAestheticsLabeler(DomainLabelerBase):
    """Scores music on five aesthetic dimensions using laion/music-aesthetics."""

    WINDOW_SEC: float = 30.0  # pipeline uses speaker-span windows of this size

    def __init__(
        self,
        whisper_repo: str = "laion/music-whisper",
        aesthetics_repo: str = "laion/music-aesthetics",
    ) -> None:
        from huggingface_hub import hf_hub_download
        from transformers import WhisperModel, WhisperProcessor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._processor = WhisperProcessor.from_pretrained(whisper_repo)
        encoder = WhisperModel.from_pretrained(whisper_repo).encoder
        self._encoder = encoder.to(self._device).eval()

        self._model = _MusicAestheticsMoE().to(self._device)
        self._model.load_state_dict(
            torch.load(
                hf_hub_download(aesthetics_repo, "stage1_bottleneck.pt"),
                map_location=self._device,
            )
        )
        for metric in _METRICS:
            self._model.heads[metric].load_state_dict(
                torch.load(
                    hf_hub_download(aesthetics_repo, f"expert_{metric}.pt"),
                    map_location=self._device,
                )
            )
        self._model.eval()

    @torch.inference_mode()
    def label(self, wave: np.ndarray, sr: int) -> Dict[str, Any]:
        audio = _resample_and_pad(wave, sr)

        inputs = self._processor(
            audio, sampling_rate=_TARGET_SR, return_tensors="pt"
        )
        hidden = self._encoder(
            inputs.input_features.to(self._device)
        ).last_hidden_state  # (1, 1500, 768)

        feat = _pool_whisper(hidden)  # (1, 23040)
        raw = self._model(feat)
        scores = {m: float(v.item()) for m, v in raw.items()}
        overall = float(np.mean(list(scores.values())))

        if overall >= 3.5:
            primary = "High"
        elif overall >= 2.5:
            primary = "Mid"
        else:
            primary = "Low"

        return {
            "primary": primary,
            "scores": {**scores, "Overall": overall},
        }

    def label_episode(
        self,
        wave: np.ndarray,
        sr: int,
        segments: List[Dict[str, Any]],
    ) -> None:
        """Score the full audio in WINDOW_SEC chunks per speaker tag and
        update domain_labels in each segment dict in-place.

        Groups by speaker so each song/section is windowed independently.
        All segments of a speaker receive the mean score across windows.
        """
        from collections import defaultdict
        spk_segs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for seg in segments:
            spk_segs[seg.get("speaker", "")].append(seg)

        for spk_list in spk_segs.values():
            t_start = min(float(s["start"]) for s in spk_list)
            t_end = max(float(s["end"]) for s in spk_list)

            # score non-overlapping WINDOW_SEC chunks over the speaker span
            results: List[Dict[str, Any]] = []
            t = t_start
            while t < t_end:
                w0, w1 = t, min(t + self.WINDOW_SEC, t_end)
                chunk = wave[int(w0 * sr): int(w1 * sr)]
                results.append(self.label(chunk, sr))
                t += self.WINDOW_SEC

            # average per-metric scores across windows
            avg_scores: Dict[str, float] = {
                m: float(np.mean([r["scores"][m] for r in results]))
                for m in results[0]["scores"]
            }
            overall = avg_scores["Overall"]
            primary = "High" if overall >= 3.5 else "Mid" if overall >= 2.5 else "Low"
            song_label = {"primary": primary, "scores": avg_scores}

            for seg in spk_list:
                dl = seg.setdefault("domain_labels", {})
                dl.update(song_label)
