#!/bin/bash
#SBATCH --job-name=harp-extract
#SBATCH --partition=CHANGE_ME    # e.g. gpuA100x4,gpuA40x4
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err

# Usage: sbatch scripts/slurm/extract.sbatch conf/emotion_hybrid_all.yaml 1 [extra args]
set -e
CONFIG=${1:?usage: extract.sbatch CONFIG STAGE [extra args]}
STAGE=${2:?usage: extract.sbatch CONFIG STAGE [extra args]}
shift 2

mkdir -p logs
source .venv/bin/activate
export HF_TOKEN=${HF_TOKEN:-$(cat ~/.cache/huggingface/token)}
python scripts/run_extract.py "$CONFIG" --stage "$STAGE" "$@"
