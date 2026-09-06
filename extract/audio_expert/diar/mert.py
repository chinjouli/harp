from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torchaudio
from sklearn.preprocessing import normalize

from extract.audio_expert.base import DiarBase

_TARGET_SR = 24000  # MERT training rate


def _to_24k(wave: np.ndarray, sr: int) -> np.ndarray:
    t = torch.from_numpy(np.asarray(wave, dtype=np.float32))
    if t.ndim == 2:
        t = t.mean(dim=0)
    if sr != _TARGET_SR:
        t = torchaudio.functional.resample(t, sr, _TARGET_SR)
    return t.numpy()


class MERTDiar(DiarBase):
    """Music track segmentation via MERT embeddings + change-point detection.

    Embeds overlapping windows with MERT, finds boundaries with ruptures
    KernelCPD, then clusters segments by cosine similarity to assign
    song IDs (returned as 'speaker' for pipeline compatibility).

    Args:
        model_name:    HuggingFace MERT checkpoint.
        layer_idx:     Which hidden layer to pool (-1 = last).
        win_sec:       Embedding window length in seconds.
        hop_sec:       Hop between consecutive windows.
        min_seg_sec:   Minimum segment duration enforced by ruptures.
        sim_threshold: Cosine similarity above which two segments get
                       the same song ID (agglomerative distance threshold
                       = 1 - sim_threshold).
        penalty:       ruptures KernelCPD pen value; lower = more splits.
    """

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        layer_idx: int = -1,
        win_sec: float = 10.0,
        hop_sec: float = 2.0,
        min_seg_sec: float = 90.0,
        penalty: float = 12.0,
    ) -> None:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self._layer_idx = layer_idx
        self._win_sec = win_sec
        self._hop_sec = hop_sec
        self._min_seg_sec = min_seg_sec
        self._penalty = penalty
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model = (
            AutoModel.from_pretrained(model_name, trust_remote_code=True)
            .to(self._device)
            .eval()
        )

    @torch.inference_mode()
    def _embed_windows(
        self, wave: np.ndarray
    ) -> tuple[np.ndarray, List[float]]:
        """Slide a window over wave; return (n_wins, 768) and start times."""
        win = int(self._win_sec * _TARGET_SR)
        hop = int(self._hop_sec * _TARGET_SR)
        n = len(wave)
        starts = list(range(0, max(1, n - win + 1), hop))

        vecs: List[np.ndarray] = []
        for s in starts:
            chunk = wave[s: s + win]
            inp = self._processor(
                chunk, sampling_rate=_TARGET_SR, return_tensors="pt"
            ).to(self._device)
            hidden = self._model(
                **inp, output_hidden_states=True
            ).hidden_states[self._layer_idx]  # (1, T, 768)
            vecs.append(
                hidden.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            )

        return np.stack(vecs), [s / _TARGET_SR for s in starts]

    def segment(
        self, wave: np.ndarray, sr: int
    ) -> List[Dict[str, Any]]:
        import ruptures as rpt

        wave24 = _to_24k(wave, sr)
        total_sec = len(wave24) / _TARGET_SR

        embs, win_starts = self._embed_windows(wave24)
        n_wins = len(embs)

        if n_wins == 1:
            return [{"speaker": "SONG_01", "start": 0.0, "end": total_sec}]

        # Normalize so rbf kernel approximates cosine similarity
        embs_n = normalize(embs, norm="l2")
        min_size = max(1, int(self._min_seg_sec / self._hop_sec))
        bpts = rpt.KernelCPD(kernel="rbf", min_size=min_size).fit_predict(
            embs_n, pen=self._penalty
        )
        # bpts = [..., n_wins]; prepend 0 to get boundary pairs
        bounds = [0] + bpts

        # Per-segment mean embeddings and time ranges
        seg_embs: List[np.ndarray] = []
        seg_times: List[tuple[float, float]] = []
        for i in range(len(bounds) - 1):
            s_idx, e_idx = bounds[i], bounds[i + 1]
            seg_embs.append(embs_n[s_idx:e_idx].mean(axis=0))
            t_start = win_starts[s_idx]
            t_end = win_starts[e_idx] if e_idx < n_wins else total_sec
            seg_times.append((t_start, t_end))

        # Sequential IDs: each detected segment is a distinct song
        return [
            {
                "speaker": f"SONG_{i+1:02d}",
                "start": float(t_start),
                "end": float(t_end),
            }
            for i, (t_start, t_end) in enumerate(seg_times)
        ]
