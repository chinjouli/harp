#!/bin/bash
# Serve a Qwen backbone with vLLM on $PORT (default 8091).
# Usage: scripts/serve_vllm.sh [text|omni]
set -e
MODE=${1:-text}
PORT=${PORT:-8091}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$MODE" in
  text)
    vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --max-model-len 16384
    ;;
  omni)
    vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --port "$PORT" \
        --tensor-parallel-size 2 \
        --max-model-len 32768 \
        --no-enable-prefix-caching --enforce-eager \
        --disable-custom-all-reduce
    ;;
  *)
    echo "usage: $0 [text|omni]" >&2; exit 1
    ;;
esac
