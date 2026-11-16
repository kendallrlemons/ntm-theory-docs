#!/usr/bin/env bash
#SBATCH --time=144:00:00
#SBATCH --job-name=fit_fah_campaign_pilot
#SBATCH --cpus-per-task=4
#SBATCH --output=fit_fah_campaign_pilot_%j.out
#SBATCH --error=fit_fah_campaign_pilot_%j.err

set -eo pipefail
source ~/.bashrc
module load cuda/12.0
conda activate gnn_rbfe
set -u
cd /data1/choderaj/lemonsk/FEC_Sandbox
export PYTHONUNBUFFERED=1

python run_fah_campaign_pilot.py \
  --benchmark_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_campaign_pilot_v1/strict_network \
  --parent_benchmark_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_benchmark_v2/strict_network \
  --parent_results_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_results_strict_network_v2 \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_campaign_pilot_results_v1/strict_network/trees \
  --threads "${SLURM_CPUS_PER_TASK:-4}"
