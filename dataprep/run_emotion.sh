#!/usr/bin/env bash
# LongEmo preparation (MSP-Podcast + MSP-Conversation).
#
#   audio    check the licensed corpus is present, then cut the example clips
#   queries  find candidates and write the query files
#   all      audio, then queries
#
# Usage: dataprep/run_emotion.sh audio|queries|all [--src DIR] [--out DIR]
# Steps that resume on their own (a --rerun flag or a per-item exists check)
# are just invoked; the guards below cover the steps that would otherwise redo
# expensive work or overwrite good output.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

STAGE="${1:-}"
if [ "$STAGE" != "audio" ] && [ "$STAGE" != "queries" ] && [ "$STAGE" != "all" ]; then
  echo "usage: run_emotion.sh audio|queries|all [--src DIR] [--out DIR]" >&2
  exit 1
fi
shift

# --src / --out are forwarded to every script; read them here too, for the
# corpus check and the skip checks.
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
  SRC="$BASE/longemo_dataset"
fi
if [ -z "$OUT" ]; then
  OUT="$SRC"
fi

if [ "$STAGE" = "audio" ] || [ "$STAGE" = "all" ]; then
  if [ ! -d "$SRC/msppodcast_full" ]; then
    cat >&2 <<MSG
MSP-Podcast and MSP-Conversation cannot be downloaded automatically: they are
released under an academic licence signed per person.

    https://lab-msp.com/MSP/MSP-Podcast.html
    https://lab-msp.com/MSP/MSP-Conversation.html

Unpack your copy so that $SRC/msppodcast_full/ contains msp_podcast_v2.0/,
msp_conversation_v2.0/, podcasts_flac/ and all_data.json.
MSG
    exit 1
  fi
  echo "ok    corpus           $SRC/msppodcast_full found"

  if compgen -G "$OUT/jsons/*.jsonl" > /dev/null; then
    echo "skip  split episodes  $OUT/jsons already populated"
  else
    echo "run   split all_data.json into per-episode JSONL"
    python dataprep/emotion/make_jsonl.py "$@"
  fi

  echo "run   select test episodes (fast, deterministic)"
  python dataprep/emotion/build_test_episodes.py "$@"

  if [ -f "$OUT/spk_clip_index.json" ]; then
    echo "skip  speaker index   $OUT/spk_clip_index.json exists"
  else
    echo "run   index speaker example clips"
    python dataprep/emotion/getspk_example.py --build-index "$@"
  fi

  if compgen -G "$OUT/audio/speaker_*.wav" > /dev/null; then
    echo "skip  speaker clips   $OUT/audio already has speaker_*.wav"
  else
    echo "run   cut speaker example clips"
    python dataprep/emotion/getspk_example.py --extract "$@"
  fi

  if [ -f "$OUT/emo_clip_index.json" ]; then
    echo "skip  emotion index   $OUT/emo_clip_index.json exists"
  else
    echo "run   index emotion example clips"
    python dataprep/emotion/getemo_example.py --build-index "$@"
  fi

  echo "run   cut emotion example clips (skips clips already cut)"
  python dataprep/emotion/getemo_example.py --extract "$@"
fi

if [ "$STAGE" = "queries" ] || [ "$STAGE" = "all" ]; then
  echo "run   find candidate events (resumes per episode)"
  python dataprep/emotion/find_candidates.py "$@"

  echo "run   assign query types (resumes per episode)"
  python dataprep/emotion/assign_queries.py "$@"

  echo "run   generate queries (resumes per episode)"
  python dataprep/emotion/generate_queries.py "$@"
fi

echo "Done."
