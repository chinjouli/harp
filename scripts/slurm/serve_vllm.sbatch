#!/bin/bash
#SBATCH --job-name=harp-vllm
#SBATCH --partition=CHANGE_ME    # e.g. gpuA100x4,gpuA40x4
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err

# Usage: sbatch scripts/slurm/serve_vllm.sbatch [text|omni]
set -e
mkdir -p logs
source .venv/bin/activate
exec scripts/serve_vllm.sh "${1:-text}"
