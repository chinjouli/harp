#!/usr/bin/env bash
# SongEval preparation.
#
#   audio    download the corpus, then build the 50 medleys and snippets
#   queries  pair songs and write the query files
#   all      audio, then queries
#
# Usage: dataprep/run_music.sh audio|queries|all [--src DIR] [--out DIR]
# Steps that resume on their own (a --rerun flag or a per-item exists check)
# are just invoked; the guards below cover the steps that would otherwise redo
# expensive work or overwrite good output.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

STAGE="${1:-}"
if [ "$STAGE" != "audio" ] && [ "$STAGE" != "queries" ] && [ "$STAGE" != "all" ]; then
  echo "usage: run_music.sh audio|queries|all [--src DIR] [--out DIR]" >&2
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
  SRC="$BASE/songeval"
fi
if [ -z "$OUT" ]; then
  OUT="$SRC"
fi

if [ "$STAGE" = "audio" ] || [ "$STAGE" = "all" ]; then
  if compgen -G "$SRC/mp3/*.mp3" > /dev/null; then
    echo "skip  download        $SRC/mp3 already populated"
  else
    echo "run   download SongEval -> $SRC"
    huggingface-cli download "${SONGEVAL_REPO:-ASLP-lab/SongEval}" \
        --repo-type dataset --local-dir "$SRC" \
        --include "metadata.jsonl" "mp3/*"
  fi

  if [ -f "$OUT/languages2.jsonl" ]; then
    echo "skip  language ID     $OUT/languages2.jsonl exists"
  else
    echo "run   language ID over every mp3 (slow)"
    python dataprep/music/find_engsong.py "$@"
  fi

  if [ -f "$OUT/tracks_metadata.jsonl" ]; then
    echo "skip  make tracks     $OUT/tracks_metadata.jsonl exists"
  else
    echo "run   stitch songs into 50 medley tracks"
    python dataprep/music/make_tracks.py "$@"
  fi

  echo "run   extract 5 s reference snippets (resumes)"
  python dataprep/music/extract_clips.py "$@"
fi

if [ "$STAGE" = "queries" ] || [ "$STAGE" = "all" ]; then
  echo "run   pair songs per track (resumes per track)"
  python dataprep/music/find_pairs.py "$@"

  echo "run   generate queries (resumes per track)"
  python dataprep/music/generate_queries.py "$@"
fi

echo "Done."
