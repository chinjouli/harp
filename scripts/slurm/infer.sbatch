#!/bin/bash
#SBATCH --job-name=harp-infer
#SBATCH --partition=CHANGE_ME    # e.g. gpuA100x4,gpuA40x4
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err

# Usage: sbatch scripts/slurm/infer.sbatch conf/music_hybrid_all.yaml [--resume]
set -e
CONFIG=${1:?usage: infer.sbatch CONFIG [extra args]}
shift

mkdir -p logs
source .venv/bin/activate
python scripts/run_inference.py "$CONFIG" "$@"
