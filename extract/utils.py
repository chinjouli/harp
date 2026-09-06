from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import soundfile as sf
import yaml


def _expand(node: Any) -> Any:
    """Recursively expand ${VAR} in config strings; error on unset vars."""
    if isinstance(node, str):
        out = os.path.expandvars(node)
        if "${" in out:
            raise KeyError(f"Unset environment variable in config: {node}")
        return out
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v) for v in node]
    return node


def load_config(path: str) -> Dict[str, Any]:
    # dataprep writes derived files under HARP_OUT_ROOT; when it was not split
    # out from the source tree the two roots are the same.
    if "HARP_OUT_ROOT" not in os.environ and "HARP_DATA_ROOT" in os.environ:
        os.environ["HARP_OUT_ROOT"] = os.environ["HARP_DATA_ROOT"]
    with open(path, "r", encoding="utf-8") as f:
        return _expand(yaml.safe_load(f))


def load_class(target: str) -> Type:
    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_from_cfg(cfg: Dict[str, Any]) -> Any:
    """Instantiate a class from a config dict with _target_ key."""
    cfg = dict(cfg)
    cls = load_class(cfg.pop("_target_"))
    # recursively build nested configs
    kwargs = {
        k: build_from_cfg(v)
        if isinstance(v, dict) and "_target_" in v
        else v
        for k, v in cfg.items()
    }
    return cls(**kwargs)


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    wave, sr = sf.read(path, always_2d=False, dtype="float32")
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    return wave, int(sr)


def make_seg_id(episode_id: str, idx: int, prefix: str = "ext") -> str:
    return f"{episode_id}_{prefix}_{idx:04d}"


def append_faiss(
    vectors: np.ndarray,
    metadata: List[Dict[str, Any]],
    index_path: Path,
    meta_path: Path,
) -> None:
    """Append vectors to a dataset-level FAISS index (IndexFlatIP, L2-normed)."""
    import faiss

    vectors = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)

    if index_path.exists():
        index = faiss.read_index(str(index_path))
        if index.d != vectors.shape[1]:
            raise ValueError(
                f"Dimension mismatch: index={index.d},"
                f" vectors={vectors.shape[1]}"
            )
    else:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(vectors.shape[1])

    start_idx = int(index.ntotal)
    index.add(vectors)

    tmp = index_path.with_suffix(index_path.suffix + ".tmp")
    faiss.write_index(index, str(tmp))
    tmp.replace(index_path)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("a", encoding="utf-8") as f:
        for i, entry in enumerate(metadata):
            entry["faiss_idx"] = start_idx + i
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
