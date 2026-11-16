#!/usr/bin/env bash
#SBATCH --time=144:00:00
#SBATCH --job-name=prep_hypersphere
#SBATCH --cpus-per-task=8
#SBATCH --output=prep_hypersphere_%j.out
#SBATCH --error=prep_hypersphere_%j.err

set -eo pipefail

# Initialize the cluster environment before enabling unset-variable checks.
source ~/.bashrc
module load cuda/12.0
conda activate gnn_rbfe
set -u

# Directory containing prepare_fah_gnn.py and its Python dependencies.
cd /data1/choderaj/lemonsk/FEC_Sandbox

python -u prepare_fah_gnn.py \
  --benchmark_dirs \
    /data1/choderaj/lemonsk/FEC_Sandbox/fah_benchmark_v2/known_reference \
    /data1/choderaj/lemonsk/FEC_Sandbox/fah_benchmark_v2/strict_network \
  --architecture hypersphere \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_gnn_graphs_v1/hypersphere \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
