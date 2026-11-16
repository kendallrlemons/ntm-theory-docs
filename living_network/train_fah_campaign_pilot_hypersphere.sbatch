#!/usr/bin/env bash
#SBATCH --time=144:00:00
#SBATCH --job-name=train_fah_campaign_pilot_hypersphere
#SBATCH --cpus-per-task=4
#SBATCH --output=train_fah_campaign_pilot_hypersphere_%j.out
#SBATCH --error=train_fah_campaign_pilot_hypersphere_%j.err
#SBATCH --gres=gpu:1

set -eo pipefail
source ~/.bashrc
module load cuda/12.0
conda activate gnn_rbfe
set -u
cd /data1/choderaj/lemonsk/FEC_Sandbox
export PYTHONUNBUFFERED=1

python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0), flush=True)'
for task in pilot_complex pilot_solvent; do
  for mode in structure structure_plus_pilot; do
    python run_fah_gnn.py \
      --benchmark_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_campaign_pilot_v1/strict_network \
      --parent_benchmark_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_benchmark_v2/strict_network \
      --reference_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_campaign_pilot_results_v1/strict_network/trees \
      --graph_cache /data1/choderaj/lemonsk/FEC_Sandbox/fah_gnn_graphs_v1/hypersphere \
      --architecture hypersphere --task "$task" --mode "$mode" \
      --output_dir "/data1/choderaj/lemonsk/FEC_Sandbox/fah_campaign_pilot_results_v1/strict_network/hypersphere/$task/$mode" \
      --threads "${SLURM_CPUS_PER_TASK:-4}"
  done
done
