#!/usr/bin/env bash
#SBATCH --time=144:00:00
#SBATCH --job-name=prepare_fah_campaign_pilot
#SBATCH --cpus-per-task=1
#SBATCH --output=prepare_fah_campaign_pilot_%j.out
#SBATCH --error=prepare_fah_campaign_pilot_%j.err

set -eo pipefail
source ~/.bashrc
module load cuda/12.0
conda activate gnn_rbfe
set -u
cd /data1/choderaj/lemonsk/FEC_Sandbox
export PYTHONUNBUFFERED=1

python prepare_fah_campaign_pilot.py \
  --parent_benchmark_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_benchmark_v2/strict_network \
  --generations /data1/choderaj/lemonsk/FEC_Sandbox/moonshot_edges_v2/generations.csv \
  --manifest /data1/choderaj/lemonsk/FEC_Sandbox/moonshot_edges_v2/manifest.json \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_campaign_pilot_v1/strict_network
