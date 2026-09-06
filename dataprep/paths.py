"""Source and output roots for the preparation scripts.

Every script reads a raw corpus and writes derived artifacts. Both locations
are arguments so the two can be separated: keep the downloaded corpus
read-only and send everything generated to its own directory.

Resolution order, highest first:
  --src / --out
  $HARP_SRC_ROOT/<dataset>  /  $HARP_OUT_ROOT/<dataset>
  $HARP_DATA_ROOT/<dataset> (both; the in-place default)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def add_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--src", default=None,
        help="Raw corpus directory (default: $HARP_SRC_ROOT or $HARP_DATA_ROOT)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Where to write derived files (default: same as --src)",
    )


def _root(explicit: str | None, env: str, dataset: str) -> Path | None:
    if explicit:
        return Path(explicit)
    base = os.environ.get(env) or os.environ.get("HARP_DATA_ROOT")
    return Path(base) / dataset if base else None


def resolve(args: argparse.Namespace, dataset: str) -> tuple[Path, Path]:
    """Return (src, out) for one dataset, creating out if needed."""
    src = _root(args.src, "HARP_SRC_ROOT", dataset)
    if src is None:
        raise SystemExit(
            f"No source directory for {dataset}: pass --src or set "
            "HARP_SRC_ROOT / HARP_DATA_ROOT"
        )
    if not src.is_dir():
        raise SystemExit(f"Source directory not found: {src}")

    out = _root(args.out, "HARP_OUT_ROOT", dataset) or src
    out.mkdir(parents=True, exist_ok=True)
    return src, out
