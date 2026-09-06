#!/usr/bin/env bash
# MedMosaic long-form preparation.
#
#   audio    download the corpus; long_form/ needs no further processing
#   queries  write queries.jsonl from the RES annotations
#   all      audio, then queries
#
# Usage: dataprep/run_health.sh audio|queries|all [--src DIR] [--out DIR]
# Steps that resume on their own (a --rerun flag or a per-item exists check)
# are just invoked; the guards below cover the steps that would otherwise redo
# expensive work or overwrite good output.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

STAGE="${1:-}"
if [ "$STAGE" != "audio" ] && [ "$STAGE" != "queries" ] && [ "$STAGE" != "all" ]; then
  echo "usage: run_health.sh audio|queries|all [--src DIR] [--out DIR]" >&2
  exit 1
fi
shift

# --src / --out are forwarded to every script; read them here too, for the
# download step and the skip checks.
ARGS=("$@")
SRC=""
OUT=""
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[i]}" in
    --src) SRC="${ARGS[i + 1]:-}" ;;
    --out) OUT="${ARGS[i + 1]:-}" ;;
  esac
done
if [ -z "$SRC" ]; then
  BASE="${HARP_SRC_ROOT:-${HARP_DATA_ROOT:-}}"
  if [ -z "$BASE" ]; then
    echo "pass --src or set HARP_SRC_ROOT / HARP_DATA_ROOT" >&2
    exit 1
  fi
  SRC="$BASE/medmosaic"
fi
if [ -z "$OUT" ]; then
  OUT="$SRC"
fi

if [ "$STAGE" = "audio" ] || [ "$STAGE" = "all" ]; then
  if compgen -G "$SRC/long_form/*.wav" > /dev/null; then
    echo "skip  download        $SRC/long_form already populated"
  else
    echo "run   download MedMosaic -> $SRC"
    huggingface-cli download "${MEDMOSAIC_REPO:-icml-anon-submission/medmosaic-dataset}" \
        --repo-type dataset --local-dir "$SRC" \
        --include "long_form/*"
  fi
  echo "      long_form/ is used as-is; no merging or cropping needed"
fi

if [ "$STAGE" = "queries" ] || [ "$STAGE" = "all" ]; then
  echo "run   generate reported/observed queries (fast, deterministic)"
  python dataprep/health/generate_queries.py "$@"
fi

echo "Done."
