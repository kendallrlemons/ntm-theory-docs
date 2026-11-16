#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --job-name=prepare_fah_growth
#SBATCH --cpus-per-task=1
#SBATCH --output=prepare_fah_growth_%j.out
#SBATCH --error=prepare_fah_growth_%j.err

set -eo pipefail
source ~/.bashrc
module load cuda/12.0
conda activate gnn_rbfe
set -u
cd /data1/choderaj/lemonsk/FEC_Sandbox
export PYTHONUNBUFFERED=1
python prepare_fah_growth.py \
  --benchmark_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_benchmark_v2/known_reference \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_growth_v1
